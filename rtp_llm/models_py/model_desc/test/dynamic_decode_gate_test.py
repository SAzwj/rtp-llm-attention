import copy
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.accuracy.attention_record import (
    AttentionForwardRecord,
    AttentionLayerRecord,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.golden_sdpa_impl import (
    GoldenSDPAImpl,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.recording_wrapper import (
    RecordingWrapper,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.decode_gate import (
    build_decode_gate,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.decode_gate_calibration import (
    calibrate,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder import (
    GateRecorder,
    _GoldenFeedback,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.precision_metrics import (
    DecodeGateConfig,
    evaluate_precision,
)
from rtp_llm.ops import KvCacheDataType, RuntimeConfig


def records_fixture():
    q = torch.ones(2, 2, 4, dtype=torch.bfloat16)
    layer = AttentionLayerRecord(
        0,
        q,
        q[:, :1].clone(),
        q[:, :1].clone(),
        q.reshape(2, 8).clone(),
        torch.ones(2, 2),
    )
    records = {
        "s": {
            "golden": [AttentionForwardRecord("golden", "decode_0", {0: layer})],
            "Good": [
                AttentionForwardRecord("Good", "decode_0", {0: copy.deepcopy(layer)})
            ],
        }
    }
    return records, {("s", "decode_0"): ((0, 2), ("Good",))}


class DecodePrecisionGateTest(unittest.TestCase):
    def test_runtime_config_survives_spawn_pickle(self):
        runtime = RuntimeConfig()
        self.assertFalse(runtime.enable_accuracy_check)
        runtime.enable_accuracy_check = True
        self.assertTrue(pickle.loads(pickle.dumps(runtime)).enable_accuracy_check)

    def test_complete_bf16_and_fp8(self):
        records, expected = records_fixture()
        for fp8 in (False, True):
            passed, details = build_decode_gate(records, expected, (0,), fp8)
            self.assertEqual(passed, frozenset({"Good"}))
            self.assertEqual(len(details["Good"]), 1)

    def test_missing_candidate_is_structural_failure(self):
        records, expected = records_fixture()
        records["s"]["Good"].clear()
        with self.assertRaisesRegex(ValueError, "incomplete candidate"):
            build_decode_gate(records, expected, (0,), False)

    def test_missing_layer_is_structural_failure(self):
        records, expected = records_fixture()
        with self.assertRaisesRegex(ValueError, "incomplete golden"):
            build_decode_gate(records, expected, (0, 1), False)

    def test_duplicate_phase_rejected(self):
        records, expected = records_fixture()
        records["s"]["Good"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_decode_gate(records, expected, (0,), False)

    def test_unaligned_inputs_rejected(self):
        records, expected = records_fixture()
        records["s"]["Good"][0].layer_records[0].q.zero_()
        with self.assertRaisesRegex(ValueError, "unaligned q"):
            build_decode_gate(records, expected, (0,), False)

    def test_nonfinite_never_passes(self):
        for value in (float("nan"), float("inf")):
            records, expected = records_fixture()
            records["s"]["Good"][0].layer_records[0].output[0, 0] = value
            passed, _ = build_decode_gate(records, expected, (0,), True)
            self.assertFalse(passed)

    def test_bad_candidate_filtered(self):
        records, expected = records_fixture()
        records["s"]["Good"][0].layer_records[0].output.mul_(2)
        for fp8 in (False, True):
            self.assertFalse(build_decode_gate(records, expected, (0,), fp8)[0])

    def test_empty_schedule_rejected(self):
        with self.assertRaisesRegex(ValueError, "no scenarios"):
            build_decode_gate({}, {}, (0,), False)

    def test_low_signal_uses_contribution_budget(self):
        golden = torch.ones(1, 128, dtype=torch.bfloat16)
        noise = torch.ones_like(golden) * 0.08
        noise[:, ::2] *= -1
        # R=1 and S=4 select the low-signal branch independent of candidate error.
        contribution = torch.full((1, 2), 4.0)
        fp8 = evaluate_precision(golden + noise, golden, contribution, True)
        self.assertEqual(fp8["low_signal_sequences"], [0])
        self.assertTrue(fp8["overall_pass"])

    def test_reliable_signal_uses_classic_metrics(self):
        golden = torch.ones(1, 128, dtype=torch.bfloat16)
        noise = torch.ones_like(golden) * 0.08
        noise[:, ::2] *= -1
        contribution = torch.ones((1, 2))
        verdict = evaluate_precision(golden + noise, golden, contribution, True)
        self.assertEqual(verdict["low_signal_sequences"], [])
        self.assertFalse(verdict["overall_pass"])
        self.assertFalse(
            evaluate_precision(golden * 1.4, golden, contribution, True)["overall_pass"]
        )

    def test_zero_output_is_valid(self):
        self.assertTrue(
            evaluate_precision(
                torch.zeros(1, 8),
                torch.zeros(1, 8),
                torch.zeros(1, 2),
                False,
            )["overall_pass"]
        )

    def test_signal_class_does_not_depend_on_candidate_error(self):
        golden = torch.ones(1, 8, dtype=torch.bfloat16)
        contribution = torch.full((1, 2), 4.0)
        config = DecodeGateConfig(gain_limit=100.0, budget_coefficient=1.0)
        for candidate in (golden, golden * 100):
            verdict = evaluate_precision(candidate, golden, contribution, True, config)
            self.assertEqual(verdict["low_signal_sequences"], [0])

    def test_golden_contribution_matches_weighted_v_formula(self):
        impl = object.__new__(GoldenSDPAImpl)
        impl.head_num = 2
        impl.kv_head_num = 1
        impl.scale = 1.0
        q = torch.ones(1, 2, 4)
        k = torch.zeros(8, 1, 4)
        v = torch.ones(8, 1, 4)
        contribution = impl._attention_contribution(q, k, v)
        torch.testing.assert_close(contribution, torch.full((1, 2), 1.0 / 8**0.5))

    def test_calibration_requires_normal_pass_and_fault_rejection(self):
        normal = torch.ones(1, 8, dtype=torch.bfloat16)
        records = [
            dict(
                candidate=normal,
                reference=normal,
                contribution=torch.ones(1, 2),
                fp8=fp8,
                kind="normal",
                cohort=f"normal-{fp8}",
            )
            for fp8 in (False, True)
        ] + [
            dict(
                candidate=normal * 2,
                reference=normal,
                contribution=torch.ones(1, 2),
                fp8=fp8,
                kind="scale",
                cohort=f"scale-{fp8}",
            )
            for fp8 in (False, True)
        ]
        selected, _ = calibrate(
            records,
            [DecodeGateConfig()],
            min_normal_pass=1.0,
            min_fault_detection=1.0,
        )
        self.assertIn("fp8", selected)

    def test_golden_feedback_records_actual_output(self):
        configs = SimpleNamespace(head_num=2, kv_head_num=1, size_per_head=4)
        inner = SimpleNamespace(
            forward=lambda qkv, cache, layer: torch.full((2, 8), 9.0)
        )
        wrapper = RecordingWrapper(inner, configs, object())
        records, _ = records_fixture()
        feedback = _GoldenFeedback(wrapper, records["s"]["golden"][0])
        output = feedback.forward(torch.ones(2, 16), None)
        self.assertTrue(torch.equal(output, torch.ones(2, 8, dtype=torch.bfloat16)))
        self.assertTrue(
            torch.equal(feedback.layer_records[0].output, torch.full((2, 8), 9.0))
        )

    def test_world_gate_failure_does_not_install_names(self):
        class Good:
            pass

        records, expected = records_fixture()
        model = SimpleNamespace(
            prepare_fmha_impl=lambda: None,
            layer_num=1,
            config=SimpleNamespace(
                getAttentionConfigs=lambda size: SimpleNamespace(
                    dtype=torch.bfloat16,
                    kv_cache_dtype=KvCacheDataType.BASE,
                    rope_config=None,
                    max_seq_len=128,
                )
            ),
            parallelism_config=SimpleNamespace(
                world_size=2, get_attn_tp_size=lambda: 1
            ),
            set_decode_backend_gate=lambda names: None,
        )
        with patch(
            "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_rope_cache_once"
        ):
            recorder = GateRecorder(model)
        recorder.records, recorder.expected = records, expected
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder"
        original_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)

        def other_rank_failed(mask, group):
            mask[0] += 0
            mask[1:] += 1

        with patch(module + ".DECODE_MHA_IMPS", [Good]), patch(
            module + ".torch.tensor", side_effect=cpu_tensor
        ), patch(module + ".all_reduce", side_effect=other_rank_failed):
            with self.assertRaisesRegex(RuntimeError, "at least one rank"):
                recorder.finalize()


if __name__ == "__main__":
    unittest.main()
