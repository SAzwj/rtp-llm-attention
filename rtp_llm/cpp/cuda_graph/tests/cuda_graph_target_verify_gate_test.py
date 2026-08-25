import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, PyModelOutputs


class GraphSafeModel:
    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        return PyModelOutputs(inputs.input_hiddens + 1)


def build_inputs(token_count: int, *, is_prefill: bool, is_target_verify: bool):
    inputs = PyModelInputs()
    inputs.input_ids = torch.arange(token_count, dtype=torch.int32, device="cuda")
    inputs.input_hiddens = torch.zeros(
        (token_count, 4), dtype=torch.float16, device="cuda"
    )

    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = is_prefill
    attention_inputs.is_target_verify = is_target_verify
    attention_inputs.input_lengths = torch.tensor([token_count], dtype=torch.int32)
    inputs.attention_inputs = attention_inputs
    return inputs


class TestCudaGraphTargetVerifyGate(unittest.TestCase):
    def test_target_verify_runner_only_accepts_verify_prefill(self):
        runner = CudaGraphRunner()
        runner.init_decode(
            GraphSafeModel(),
            4,
            64,
            64,
            64,
            [1],
            True,
            3,
        )

        ordinary_prefill = build_inputs(30, is_prefill=True, is_target_verify=False)
        self.assertFalse(runner.canRun(ordinary_prefill))

        non_prefill_verify = build_inputs(3, is_prefill=False, is_target_verify=True)
        self.assertFalse(runner.canRun(non_prefill_verify))

        target_verify = build_inputs(3, is_prefill=True, is_target_verify=True)
        self.assertTrue(runner.canRun(target_verify))


if __name__ == "__main__":
    unittest.main()
