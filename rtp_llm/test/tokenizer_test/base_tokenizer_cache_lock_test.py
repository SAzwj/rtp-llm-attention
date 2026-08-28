import hashlib
import importlib.util
import json
import multiprocessing
import os
import queue
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

BASE_TOKENIZER_RUNFILE = (
    "rtp_llm/frontend/tokenizer_factory/tokenizers/base_tokenizer.py"
)


def _base_tokenizer_path() -> Path:
    test_srcdir = os.environ.get("TEST_SRCDIR")
    if test_srcdir:
        workspace = os.environ.get("TEST_WORKSPACE", "rtp_llm")
        return Path(test_srcdir) / workspace / BASE_TOKENIZER_RUNFILE
    return Path(__file__).resolve().parents[2] / BASE_TOKENIZER_RUNFILE.removeprefix(
        "rtp_llm/"
    )


def _load_base_tokenizer(path: str):
    spec = importlib.util.spec_from_file_location("rtp_base_tokenizer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BaseTokenizer


def _install_fake_transformers(cache_root: str, callback):
    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, tokenizer_path, **kwargs):
            return callback(tokenizer_path, kwargs)

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer
    transformers.__version__ = "4.0.0"
    transformers_utils = types.ModuleType("transformers.utils")
    transformers_hub = types.ModuleType("transformers.utils.hub")
    transformers_hub.HF_MODULES_CACHE = cache_root
    transformers.utils = transformers_utils
    transformers_utils.hub = transformers_hub

    packaging = types.ModuleType("packaging")
    packaging_version = types.ModuleType("packaging.version")
    packaging_version.parse = lambda unused: types.SimpleNamespace(major=4)
    packaging.version = packaging_version

    sys.modules["transformers"] = transformers
    sys.modules["transformers.utils"] = transformers_utils
    sys.modules["transformers.utils.hub"] = transformers_hub
    sys.modules["packaging"] = packaging
    sys.modules["packaging.version"] = packaging_version


def _tokenizer_worker(
    base_tokenizer_path,
    tokenizer_path,
    cache_root,
    role,
    mode,
    start_barrier,
    active,
    max_active,
    state_lock,
    both_entered,
    release,
    entered,
    result_queue,
):
    if start_barrier is not None:
        start_barrier.wait(timeout=10)

    if mode == "forbid_lock":
        fcntl = types.ModuleType("fcntl")
        fcntl.LOCK_EX = 2
        fcntl.LOCK_UN = 8

        def fail_flock(unused_fd, unused_operation):
            raise AssertionError("non-remote tokenizer unexpectedly acquired a lock")

        fcntl.flock = fail_flock
        sys.modules["fcntl"] = fcntl

    def from_pretrained(path, unused_kwargs):
        if mode == "block" and role == "blocker":
            entered.set()
            release.wait(timeout=30)
        elif mode == "error" and role == "failing":
            entered.set()
            time.sleep(0.2)
            raise ValueError("malformed tokenizer source")
        elif mode in {"measure", "overlap"}:
            with state_lock:
                active.value += 1
                max_active.value = max(max_active.value, active.value)
                if active.value == 2:
                    both_entered.set()
            if mode == "overlap":
                release.wait(timeout=5)
            else:
                time.sleep(0.3)
            with state_lock:
                active.value -= 1
        return types.SimpleNamespace(name=Path(path).name)

    _install_fake_transformers(cache_root, from_pretrained)
    BaseTokenizer = _load_base_tokenizer(base_tokenizer_path)
    try:
        tokenizer = BaseTokenizer(tokenizer_path)
        result_queue.put(
            {
                "role": role,
                "status": "success",
                "name": tokenizer.get_real_tokenizer().name,
            }
        )
    except Exception as error:
        result_queue.put(
            {
                "role": role,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )


class BaseTokenizerCacheLockTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.base_tokenizer_path = str(_base_tokenizer_path())

    def tearDown(self):
        self.temp_dir.cleanup()

    def _model_dir(self, name: str, remote_code: bool = True) -> Path:
        model_dir = self.root / name
        model_dir.mkdir()
        config = {"tokenizer_class": "FakeTokenizer"}
        if remote_code:
            config["auto_map"] = {
                "AutoTokenizer": ["tokenization_fake.FakeTokenizer", None]
            }
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        return model_dir

    def _state(self, context):
        return {
            "active": context.Value("i", 0),
            "max_active": context.Value("i", 0),
            "state_lock": context.Lock(),
            "both_entered": context.Event(),
            "release": context.Event(),
            "entered": context.Event(),
            "result_queue": context.Queue(),
        }

    def _process(
        self,
        context,
        state,
        tokenizer_path,
        cache_root,
        role,
        mode,
        start_barrier=None,
    ):
        return context.Process(
            target=_tokenizer_worker,
            args=(
                self.base_tokenizer_path,
                str(tokenizer_path),
                cache_root,
                role,
                mode,
                start_barrier,
                state["active"],
                state["max_active"],
                state["state_lock"],
                state["both_entered"],
                state["release"],
                state["entered"],
                state["result_queue"],
            ),
        )

    def _join(self, *processes):
        for process in processes:
            process.join(timeout=10)
            self.assertFalse(process.is_alive(), f"child did not exit: {process.pid}")
            self.assertEqual(process.exitcode, 0)

    def test_same_model_and_cache_are_serialized(self):
        model_dir = self._model_dir("same-model")
        previous_cwd = os.getcwd()
        os.chdir(self.root)
        try:
            for start_method in ("spawn", "fork"):
                if start_method not in multiprocessing.get_all_start_methods():
                    continue
                with self.subTest(start_method=start_method):
                    context = multiprocessing.get_context(start_method)
                    state = self._state(context)
                    barrier = context.Barrier(2)
                    processes = [
                        self._process(
                            context,
                            state,
                            model_dir,
                            "relative-cache",
                            f"worker-{index}",
                            "measure",
                            barrier,
                        )
                        for index in range(2)
                    ]
                    for process in processes:
                        process.start()
                    self._join(*processes)
                    results = [state["result_queue"].get(timeout=2) for _ in range(2)]
                    self.assertTrue(
                        all(result["status"] == "success" for result in results)
                    )
                    self.assertEqual(state["max_active"].value, 1)
        finally:
            os.chdir(previous_cwd)

    def test_different_keys_do_not_block_each_other(self):
        context = multiprocessing.get_context("spawn")
        scenarios = [
            (self._model_dir("model-a"), self._model_dir("model-b"), "cache", "cache"),
            (
                self._model_dir("same-model"),
                self.root / "same-model",
                "cache-a",
                "cache-b",
            ),
        ]
        previous_cwd = os.getcwd()
        os.chdir(self.root)
        try:
            for first_model, second_model, first_cache, second_cache in scenarios:
                with self.subTest(
                    first_model=first_model.name,
                    first_cache=first_cache,
                    second_cache=second_cache,
                ):
                    state = self._state(context)
                    barrier = context.Barrier(2)
                    first = self._process(
                        context,
                        state,
                        first_model,
                        first_cache,
                        "first",
                        "overlap",
                        barrier,
                    )
                    second = self._process(
                        context,
                        state,
                        second_model,
                        second_cache,
                        "second",
                        "overlap",
                        barrier,
                    )
                    first.start()
                    second.start()
                    overlapped = state["both_entered"].wait(timeout=3)
                    state["release"].set()
                    self._join(first, second)
                    self.assertTrue(overlapped)
                    self.assertEqual(state["max_active"].value, 2)
        finally:
            os.chdir(previous_cwd)

    def test_exception_releases_lock_without_masking_error(self):
        context = multiprocessing.get_context("spawn")
        state = self._state(context)
        model_dir = self._model_dir("exception-model")
        cache_root = str(self.root / "cache")
        failing = self._process(
            context, state, model_dir, cache_root, "failing", "error"
        )
        failing.start()
        self.assertTrue(state["entered"].wait(timeout=3))
        following = self._process(
            context, state, model_dir, cache_root, "following", "simple"
        )
        following.start()
        self._join(failing, following)
        results = [state["result_queue"].get(timeout=2) for _ in range(2)]
        error = next(result for result in results if result["status"] == "error")
        self.assertEqual(error["error_type"], "ValueError")
        self.assertEqual(error["error"], "malformed tokenizer source")
        self.assertEqual(sum(result["status"] == "success" for result in results), 1)

    def test_process_death_releases_lock(self):
        context = multiprocessing.get_context("spawn")
        state = self._state(context)
        model_dir = self._model_dir("death-model")
        cache_root = str(self.root / "cache")
        blocker = self._process(
            context, state, model_dir, cache_root, "blocker", "block"
        )
        blocker.start()
        self.assertTrue(state["entered"].wait(timeout=3))
        follower = self._process(
            context, state, model_dir, cache_root, "follower", "simple"
        )
        follower.start()
        with self.assertRaises(queue.Empty):
            state["result_queue"].get(timeout=0.5)
        blocker.terminate()
        blocker.join(timeout=5)
        self.assertFalse(blocker.is_alive())
        self._join(follower)
        result = state["result_queue"].get(timeout=2)
        self.assertEqual(result["status"], "success")

    def test_read_only_cache_root_uses_fallback(self):
        context = multiprocessing.get_context("spawn")
        state = self._state(context)
        model_dir = self._model_dir("fallback-model")
        cache_root = f"/proc/rtp-llm-tokenizer-lock-test-{os.getpid()}"
        process = self._process(
            context, state, model_dir, cache_root, "fallback", "simple"
        )
        process.start()
        self._join(process)
        result = state["result_queue"].get(timeout=2)
        self.assertEqual(result["status"], "success")
        cache_path = Path(cache_root).resolve()
        tokenizer_path = model_dir.resolve()
        lock_scope = b"\0".join((os.fsencode(cache_path), os.fsencode(tokenizer_path)))
        lock_key = hashlib.sha256(lock_scope).hexdigest()
        fallback_lock = Path(tempfile.gettempdir()).joinpath(
            f"rtp_llm_tokenizer_locks_{os.getuid()}", f"{lock_key}.lock"
        )
        self.assertTrue(fallback_lock.is_file())

    def test_non_remote_tokenizer_does_not_use_file_lock(self):
        context = multiprocessing.get_context("spawn")
        state = self._state(context)
        model_dir = self._model_dir("local-model", remote_code=False)
        process = self._process(
            context,
            state,
            model_dir,
            f"/proc/rtp-llm-unused-lock-test-{os.getpid()}",
            "local",
            "forbid_lock",
        )
        process.start()
        self._join(process)
        result = state["result_queue"].get(timeout=2)
        self.assertEqual(result["status"], "success")


if __name__ == "__main__":
    unittest.main()
