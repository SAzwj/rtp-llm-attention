import filecmp
import importlib.metadata
import os
import pickle
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.cuda_graph import CudaGraphSelectionMode
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector import (
    benchmark_sequence_cap,
    dynamic_backend_allowed,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.selector import select_stable
from rtp_llm.ops import HWKernelConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs


class DecodeSchedulerTest(unittest.TestCase):
    def test_native_modules_come_from_declared_runfiles(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        for name in ("libth_transformer_config", "librtp_compute_ops"):
            self.assertTrue(
                os.path.samefile(sys.modules[name].__file__, root / (name + ".so"))
            )
        # gpu_lock stages declared wheel packages into stable JIT paths.
        # Check content identity rather than mistaking the copy for ambient torch.
        declared = importlib.metadata.distribution("torch")
        loaded = Path(torch.__file__).parent
        for relative in (
            "__init__.py",
            Path(torch._C.__file__).name,
            "lib/libtorch_cpu.so",
            "lib/libtorch_cuda.so",
        ):
            source = declared.locate_file("torch/" + str(relative))
            self.assertTrue(str(source).startswith(os.environ["TEST_SRCDIR"] + os.sep))
            self.assertTrue(filecmp.cmp(loaded / relative, source, shallow=False))

    def test_dynamic_config_survives_spawn_pickle(self):
        hw = HWKernelConfig()
        self.assertFalse(hw.enable_dynamic_decode_backend)
        hw.enable_dynamic_decode_backend = True
        self.assertTrue(pickle.loads(pickle.dumps(hw)).enable_dynamic_decode_backend)

    def test_dynamic_backend_arch_filter(self):
        self.assertTrue(dynamic_backend_allowed("XQADecodeImpl", 9))
        self.assertFalse(dynamic_backend_allowed("XQADecodeImpl", 10))
        self.assertTrue(dynamic_backend_allowed("FlashInferTRTLLMDecodeImpl", 10))
        self.assertTrue(dynamic_backend_allowed("PyFlashinferDecodeImpl", 10))

    def test_threshold_and_registry_tiebreak(self):
        self.assertEqual(select_stable({"A": [100], "B": [97]}, ["A", "B"], "A"), "A")
        self.assertEqual(
            select_stable({"A": [100], "B": [80], "C": [79]}, ["A", "B", "C"], "A"), "B"
        )

    def test_benchmark_cap_respects_physical_capacity_and_memory(self):
        self.assertEqual(benchmark_sequence_cap(2048, 2, 64, 8, 4096, 2**30), 255)
        self.assertLess(benchmark_sequence_cap(2048, 2, 64, 8, 4096, 0), 0)
        self.assertEqual(benchmark_sequence_cap(128, 2, 64, 8, 4096, 2**30), 128)


class DecodePlanTest(unittest.TestCase):
    def setUp(self):
        self.model = GptModelBase(
            SimpleNamespace(
                num_layers=1, vocab_size=10, capture_aux_hidden_layer_ids=[]
            ),
            SimpleNamespace(get_attn_tp_size=lambda: 1),
            object(),
            8,
        )
        from rtp_llm.models_py.modules.factory.attention.attn_factory import (
            DECODE_MHA_IMPS,
        )

        self.backend = DECODE_MHA_IMPS[-1]
        self.model.set_decode_backend_gate([self.backend.__name__])
        self.inputs = PyAttentionInputs()
        self.inputs.input_lengths = torch.ones(2, dtype=torch.int32)
        self.inputs.is_prefill = False

    def test_enable_requires_gate(self):
        self.model.set_decode_backend_gate(None)
        with self.assertRaisesRegex(RuntimeError, "before precision"):
            self.model.configure_dynamic_decode(True)

    def test_plan_selected_once_per_batch_and_applied(self):
        self.model.configure_dynamic_decode(True)
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector"
        with patch(
            module + ".select_decode_backend", return_value=self.backend
        ) as select, patch(
            module + ".instantiate_decode_backend", return_value="actual"
        ) as instantiate:
            for _ in range(2):
                self.assertEqual(
                    self.model.prepare_fmha_impl(
                        SimpleNamespace(attention_inputs=self.inputs), True
                    ),
                    "actual",
                )
            self.inputs.input_lengths = torch.ones(4, dtype=torch.int32)
            self.model.prepare_fmha_impl(
                SimpleNamespace(attention_inputs=self.inputs), True
            )
            self.assertEqual(select.call_count, 2)
            self.assertEqual(instantiate.call_count, 3)
            self.assertEqual(set(self.model.decode_backend_plan), {2, 4})

    def test_eager_and_prefill_keep_mainline_routing(self):
        self.model.configure_dynamic_decode(True)
        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            return_value="static",
        ):
            self.assertEqual(
                self.model.prepare_fmha_impl(
                    SimpleNamespace(attention_inputs=self.inputs)
                ),
                "static",
            )
            self.inputs.is_prefill = True
            self.assertEqual(
                self.model.prepare_fmha_impl(
                    SimpleNamespace(attention_inputs=self.inputs),
                    True,
                    CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH,
                ),
                "static",
            )
        self.assertFalse(self.model.decode_backend_plan)


if __name__ == "__main__":
    unittest.main()
