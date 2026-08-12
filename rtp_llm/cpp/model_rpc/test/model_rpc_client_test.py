import asyncio
import struct
import sys
from unittest.mock import MagicMock, patch

# Mock the ops module to avoid CUDA dependency in this unit test
# This MUST be at the very top before any other imports, even before unittest
mock_ops = MagicMock()
mock_comm = MagicMock()
mock_nccl_op = MagicMock()
mock_compute_ops = MagicMock()
mock_comm.nccl_op = mock_nccl_op
mock_ops.comm = mock_comm
mock_ops.compute_ops = mock_compute_ops
sys.modules["rtp_llm.ops"] = mock_ops
sys.modules["rtp_llm.ops.comm"] = mock_comm
sys.modules["rtp_llm.ops.compute_ops"] = mock_compute_ops
sys.modules["rtp_llm.ops.comm.nccl_op"] = mock_nccl_op

import logging
import os
import unittest
from dataclasses import asdict
from typing import AsyncGenerator
from unittest import TestCase, main

import grpc
import torch
from grpc import StatusCode

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.log_config import setup_logging
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    ModelRpcClient,
    StreamState,
    _engine_reported_finished,
    _record_client_span_latency,
    _record_client_span_usage,
    _request_completed_normally,
    trans_input,
    trans_output,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    GenerateInputPB,
    GenerateOutputsPB,
    TensorPB,
)
from rtp_llm.telemetry import CURRENT_TRACE_STATE
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutputs,
    RequestInfo,
)


class FakeStub:

    async def GenerateStreamCall(self, input: GenerateInputPB, timeout=None):
        # 1. 第一个响应：包含第一个生成的 token
        outputs_pb1 = GenerateOutputsPB()
        output_pb1 = outputs_pb1.flatten_output
        output_pb1.output_ids.data_type = TensorPB.DataType.INT32
        output_pb1.output_ids.shape.extend([1, 1])
        output_pb1.output_ids.int32_data = struct.pack("<i", 0)
        aux_info = output_pb1.aux_info.add()
        aux_info.iter_count = 1
        aux_info.output_len = 1
        output_pb1.logits.data_type = TensorPB.DataType.FP32
        output_pb1.logits.shape.extend([1, 1, 2])
        output_pb1.logits.fp32_data = struct.pack("<ff", 0.0, 0.0)
        output_pb1.finished.extend([False])
        yield outputs_pb1

        # 2. 第二个响应：包含累积的两个 token
        outputs_pb2 = GenerateOutputsPB()
        output_pb2 = outputs_pb2.flatten_output
        output_pb2.output_ids.data_type = TensorPB.DataType.INT32
        output_pb2.output_ids.shape.extend([1, 2])
        output_pb2.output_ids.int32_data = struct.pack("<ii", 0, 1)
        aux_info2 = output_pb2.aux_info.add()
        aux_info2.iter_count = 2
        aux_info2.output_len = 2
        aux_info2.speculative_verify_rounds = 3
        aux_info2.speculative_accepted_token_num = 9
        aux_info2.speculative_proposed_draft_tokens = 12
        aux_info2.context_execute_time_us = 100
        aux_info2.generate_execute_time_us = 200
        aux_info2.context_execute_time_with_cache_us = 80
        output_pb2.logits.data_type = TensorPB.DataType.FP32
        output_pb2.logits.shape.extend([1, 1, 2])
        output_pb2.logits.fp32_data = struct.pack("<ff", 0.1, 0.2)
        output_pb2.finished.extend([False])
        yield outputs_pb2

        # 3. 最终响应：标记结束，并携带最后一个状态
        outputs_pb3 = GenerateOutputsPB()
        output_pb3_item = outputs_pb3.flatten_output
        output_pb3_item.CopyFrom(output_pb2)
        output_pb3_item.finished[0] = True
        yield outputs_pb3


class FakeModelRpcClient(ModelRpcClient):

    def __init__(self):
        # Call parent __init__ with minimal required parameters
        super().__init__(
            [],  # addresses: empty list for fake client
            {},  # client_config: empty dict for fake client
            0,  # max_rpc_timeout_ms
            False,  # decode_entrance
        )
        self.stub = FakeStub()

    async def enqueue(
        self, input_py: GenerateInput
    ) -> AsyncGenerator[GenerateOutputs, None]:
        input_pb = trans_input(input_py)
        stream_state = StreamState()

        async for response_pb in self.stub.GenerateStreamCall(input_pb):
            yield trans_output(input_py, response_pb, stream_state)


class ModelRpcClientTest(TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # self.client = FakeModelRpcClient()

    @staticmethod
    async def _run(client, input):
        responses = []
        async for res in client.enqueue(input):
            responses.extend(res.generate_outputs)
        return responses

    def test_frontend_metric_envelope_stays_out_of_public_aux_info(self):
        input_py = GenerateInput(
            request_id=123,
            token_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
            mm_inputs=[],
            generate_config=GenerateConfig(aux_info=True),
        )
        outputs_pb = GenerateOutputsPB()
        outputs_pb.frontend_metric_only = True
        outputs_pb.frontend_context_token_num.value = 11
        outputs_pb.frontend_context_token_num_with_cache.value = 13
        outputs_pb.frontend_context_execute_time_us.value = 101
        outputs_pb.frontend_context_execute_time_with_cache_us.value = 81
        outputs_pb.frontend_generate_token_num.value = 17
        outputs_pb.frontend_generate_execute_time_us.value = 201
        outputs_pb.flatten_output.finished.append(False)
        outputs_pb.flatten_output.aux_info.add().output_len = 4

        output = trans_output(input_py, outputs_pb, StreamState())

        self.assertTrue(output.frontend_metric_only)
        self.assertEqual(output.frontend_context_token_num, 11)
        self.assertEqual(output.frontend_context_token_num_with_cache, 13)
        self.assertEqual(output.frontend_context_execute_time_us, 101)
        self.assertEqual(output.frontend_context_execute_time_with_cache_us, 81)
        self.assertEqual(output.frontend_generate_token_num, 17)
        self.assertEqual(output.frontend_generate_execute_time_us, 201)
        self.assertNotIn(
            "frontend_metric_only",
            asdict(output.generate_outputs[0].aux_info),
        )
        self.assertNotIn(
            "frontend_generate_token_num",
            asdict(output.generate_outputs[0].aux_info),
        )
        self.assertNotIn(
            "frontend_context_token_num",
            asdict(output.generate_outputs[0].aux_info),
        )
        self.assertNotIn(
            "frontend_generate_execute_time_us",
            asdict(output.generate_outputs[0].aux_info),
        )

    def test_trans_input_serializes_think_terminate_token_id(self):
        input_py = GenerateInput(
            request_id=123,
            token_ids=torch.tensor([1, 2]),
            mm_inputs=[],
            generate_config=GenerateConfig(think_terminate_token_id=42),
        )

        self.assertEqual(
            trans_input(input_py).generate_config.think_terminate_token_id, 42
        )

    @unittest.skip("need fix")
    def test_generate_stream(self):
        client = FakeModelRpcClient()
        generate_config: GenerateConfig = GenerateConfig(using_hf_sampling=False)
        input = GenerateInput(
            token_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            generate_config=generate_config,
        )
        res = asyncio.run(self._run(client, input))
        self.assertEqual(len(res), 3)
        self.assertEqual(list(res[0].output_ids.shape), [1, 1])
        self.assertEqual(res[0].output_ids.tolist(), [[0]])
        self.assertEqual(res[0].finished, False)
        self.assertEqual(res[0].aux_info.iter_count, 2)
        self.assertEqual(res[0].aux_info.output_len, 1)

        self.assertEqual(list(res[1].output_ids.shape), [1, 2])
        self.assertEqual(res[1].output_ids.tolist(), [[0, 1]])
        self.assertEqual(res[1].finished, False)
        self.assertEqual(res[1].aux_info.iter_count, 3)
        self.assertEqual(res[1].aux_info.output_len, 2)
        self.assertEqual(res[1].aux_info.speculative_verify_rounds, 3)
        self.assertEqual(res[1].aux_info.speculative_accepted_token_num, 9)
        self.assertEqual(res[1].aux_info.speculative_proposed_draft_tokens, 12)
        self.assertEqual(res[1].aux_info.context_execute_time_us, 100)
        self.assertEqual(res[1].aux_info.generate_execute_time_us, 200)
        self.assertEqual(res[1].aux_info.context_execute_time_with_cache_us, 80)

        self.assertEqual(res[2].finished, True)

    def test_generate_stream_with_logits_index(self):
        client = FakeModelRpcClient()
        generate_config: GenerateConfig = GenerateConfig(
            return_logits=True,
            logits_index=1,
            return_incremental=True,
            is_streaming=True,
        )
        input = GenerateInput(
            token_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            generate_config=generate_config,
            request_id=123,
            mm_inputs=[],
        )
        res = asyncio.run(self._run(client, input))

        self.assertEqual(len(res), 3)

        # res[0] 是第一个token
        self.assertTrue(hasattr(res[0], "logits"))
        self.assertIsNotNone(res[0].logits)
        logits_0 = res[0].logits.tolist()
        self.assertAlmostEqual(logits_0[0][0], 0.0, places=6)
        self.assertAlmostEqual(logits_0[0][1], 0.0, places=6)

        # res[1] 是第二个token
        self.assertTrue(hasattr(res[1], "logits"))
        self.assertIsNotNone(res[1].logits)
        logits_1 = res[1].logits.tolist()
        self.assertAlmostEqual(logits_1[0][0], 0.1, places=6)
        self.assertAlmostEqual(logits_1[0][1], 0.2, places=6)

        # res[2] 是完成标记，包含指定位置token的logits
        self.assertTrue(res[2].finished)
        self.assertTrue(hasattr(res[2], "logits"))
        self.assertIsNotNone(res[2].logits)
        logits_2 = res[2].logits.tolist()
        self.assertAlmostEqual(logits_2[0][0], 0.0, places=6)
        self.assertAlmostEqual(logits_2[0][1], 0.0, places=6)

    def test_trans_input_request_info(self):
        input_pb = trans_input(
            GenerateInput(
                token_ids=torch.tensor([1, 2, 3]),
                generate_config=GenerateConfig(trace_id="trace-from-config"),
                request_id=123,
                mm_inputs=[],
                headers={"x-request-id": "header-request-id"},
                request_info=RequestInfo(
                    frontend_ip="10.0.0.1",
                    dash_ip="10.0.0.2",
                    trace_id="trace-from-info",
                    request_id="source-request-id",
                    source_role="frontend",
                ),
            )
        )

        self.assertEqual(input_pb.request_info.frontend_ip, "10.0.0.1")
        self.assertEqual(input_pb.request_info.dash_ip, "10.0.0.2")
        self.assertEqual(input_pb.request_info.trace_id, "trace-from-info")
        self.assertEqual(input_pb.request_info.request_id, "source-request-id")
        self.assertEqual(input_pb.request_info.source_role, "frontend")

    def test_compact_logprobs_config_and_output_roundtrip(self):
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=2),
            request_id=123,
            mm_inputs=[],
        )
        input_pb = trans_input(input_py)
        self.assertTrue(input_pb.generate_config.return_logprobs)
        self.assertEqual(input_pb.generate_config.top_logprobs, 2)

        outputs_pb = GenerateOutputsPB()
        output_pb = outputs_pb.flatten_output
        output_pb.finished.append(False)
        output_pb.output_ids.data_type = TensorPB.DataType.INT32
        output_pb.output_ids.shape.extend([1, 2])
        output_pb.output_ids.int32_data = struct.pack("<ii", 10, 11)

        output_pb.token_logprobs.data_type = TensorPB.DataType.FP32
        output_pb.token_logprobs.shape.extend([1, 2])
        output_pb.token_logprobs.fp32_data = struct.pack("<ff", -0.1, -0.2)

        output_pb.top_logprob_token_ids.data_type = TensorPB.DataType.INT32
        output_pb.top_logprob_token_ids.shape.extend([1, 2, 2])
        output_pb.top_logprob_token_ids.int32_data = struct.pack(
            "<iiii", 10, 12, 11, 13
        )

        output_pb.top_logprobs.data_type = TensorPB.DataType.FP32
        output_pb.top_logprobs.shape.extend([1, 2, 2])
        output_pb.top_logprobs.fp32_data = struct.pack("<ffff", -0.1, -1.1, -0.2, -1.2)
        output_pb.logprobs_offsets.append(0)
        output_pb.logprobs_counts.append(2)

        result = trans_output(input_py, outputs_pb, StreamState())
        self.assertEqual(len(result.generate_outputs), 1)
        output = result.generate_outputs[0]
        self.assertEqual(output.token_logprobs.shape, torch.Size([2]))
        self.assertEqual(output.top_logprob_token_ids.shape, torch.Size([2, 2]))
        self.assertEqual(output.top_logprobs.shape, torch.Size([2, 2]))
        self.assertEqual(output.logprobs_offset, 0)
        self.assertEqual(output.logprobs_count, 2)
        self.assertTrue(
            torch.equal(
                output.top_logprob_token_ids,
                torch.tensor([[10, 12], [11, 13]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.allclose(
                output.token_logprobs,
                torch.tensor([-0.1, -0.2], dtype=torch.float32),
            )
        )

    def test_compact_logprobs_zero_top_k_output_roundtrip(self):
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=0),
            request_id=123,
            mm_inputs=[],
        )

        outputs_pb = GenerateOutputsPB()
        output_pb = outputs_pb.flatten_output
        output_pb.finished.append(False)
        output_pb.output_ids.data_type = TensorPB.DataType.INT32
        output_pb.output_ids.shape.extend([1, 2])
        output_pb.output_ids.int32_data = struct.pack("<ii", 10, 11)

        output_pb.token_logprobs.data_type = TensorPB.DataType.FP32
        output_pb.token_logprobs.shape.extend([1, 2])
        output_pb.token_logprobs.fp32_data = struct.pack("<ff", -0.1, -0.2)

        output_pb.top_logprob_token_ids.data_type = TensorPB.DataType.INT32
        output_pb.top_logprob_token_ids.shape.extend([1, 2, 0])
        output_pb.top_logprobs.data_type = TensorPB.DataType.FP32
        output_pb.top_logprobs.shape.extend([1, 2, 0])
        output_pb.logprobs_offsets.append(0)
        output_pb.logprobs_counts.append(2)

        result = trans_output(input_py, outputs_pb, StreamState())

        self.assertEqual(len(result.generate_outputs), 1)
        output = result.generate_outputs[0]
        self.assertEqual(output.token_logprobs.shape, torch.Size([2]))
        self.assertEqual(output.top_logprob_token_ids.shape, torch.Size([2, 0]))
        self.assertEqual(output.top_logprob_token_ids.dtype, torch.int32)
        self.assertEqual(output.top_logprobs.shape, torch.Size([2, 0]))
        self.assertEqual(output.top_logprobs.dtype, torch.float32)

    def test_compact_logprobs_boundary_uses_count_to_remove_rpc_padding(self):
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=1),
            request_id=123,
            mm_inputs=[],
        )

        outputs_pb = GenerateOutputsPB()
        output_pb = outputs_pb.flatten_output
        output_pb.finished.extend([False, False])
        output_pb.output_ids.data_type = TensorPB.DataType.INT32
        output_pb.output_ids.shape.extend([2, 1, 5])
        output_pb.output_ids.int32_data = struct.pack(
            "<10i", 10, 11, 128822, 271, 20, 30, 31, 32, 0, 0
        )

        # Row 0 owns two real content rows plus one transport padding row;
        # row 1 owns all three rows.
        output_pb.token_logprobs.data_type = TensorPB.DataType.FP32
        output_pb.token_logprobs.shape.extend([2, 3])
        output_pb.token_logprobs.fp32_data = struct.pack(
            "<6f", -0.13, -0.20, 0.0, -0.30, -0.31, -0.32
        )
        output_pb.top_logprob_token_ids.data_type = TensorPB.DataType.INT32
        output_pb.top_logprob_token_ids.shape.extend([2, 3, 1])
        output_pb.top_logprob_token_ids.int32_data = struct.pack(
            "<6i", 271, 20, 0, 30, 31, 32
        )
        output_pb.top_logprobs.data_type = TensorPB.DataType.FP32
        output_pb.top_logprobs.shape.extend([2, 3, 1])
        output_pb.top_logprobs.fp32_data = struct.pack(
            "<6f", -0.13, -0.20, 0.0, -0.30, -0.31, -0.32
        )
        output_pb.logprobs_offsets.extend([3, 0])
        output_pb.logprobs_counts.extend([2, 3])

        result = trans_output(input_py, outputs_pb, StreamState())

        first, second = result.generate_outputs
        self.assertEqual(first.logprobs_offset, 3)
        self.assertEqual(first.logprobs_count, 2)
        self.assertEqual(first.token_logprobs.shape, torch.Size([2]))
        self.assertTrue(
            torch.allclose(
                first.token_logprobs,
                torch.tensor([-0.13, -0.20], dtype=torch.float32),
            )
        )
        self.assertEqual(first.top_logprob_token_ids.tolist(), [[271], [20]])
        self.assertEqual(second.logprobs_offset, 0)
        self.assertEqual(second.logprobs_count, 3)
        self.assertEqual(second.token_logprobs.shape, torch.Size([3]))

    def test_compact_logprobs_thinking_only_metadata_survives_without_tensors(self):
        input_py = GenerateInput(
            token_ids=torch.tensor([1, 2, 3]),
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=0),
            request_id=123,
            mm_inputs=[],
        )

        outputs_pb = GenerateOutputsPB()
        output_pb = outputs_pb.flatten_output
        output_pb.finished.append(False)
        output_pb.output_ids.data_type = TensorPB.DataType.INT32
        output_pb.output_ids.shape.extend([1, 3])
        output_pb.output_ids.int32_data = struct.pack("<3i", 10, 11, 128822)
        output_pb.logprobs_offsets.append(3)
        output_pb.logprobs_counts.append(0)

        result = trans_output(input_py, outputs_pb, StreamState())

        output = result.generate_outputs[0]
        self.assertEqual(output.logprobs_offset, 3)
        self.assertEqual(output.logprobs_count, 0)
        self.assertIsNone(output.token_logprobs)
        self.assertIsNone(output.top_logprob_token_ids)
        self.assertIsNone(output.top_logprobs)

    def test_trans_input_request_info_fallback(self):
        input_pb = trans_input(
            GenerateInput(
                token_ids=torch.tensor([1, 2, 3]),
                generate_config=GenerateConfig(trace_id="trace-from-config"),
                request_id=123,
                mm_inputs=[],
                headers={"x-request-id": "header-request-id"},
            )
        )

        self.assertEqual(input_pb.request_info.trace_id, "trace-from-config")
        self.assertEqual(input_pb.request_info.request_id, "header-request-id")

    def test_trans_input_request_info_trace_header_fallback(self):
        traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
        input_pb = trans_input(
            GenerateInput(
                token_ids=torch.tensor([1, 2, 3]),
                generate_config=GenerateConfig(),
                request_id=123,
                mm_inputs=[],
                headers={"traceparent": traceparent},
            )
        )

        self.assertEqual(
            input_pb.request_info.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736"
        )
        self.assertEqual(
            input_pb.request_info.request_id, "4bf92f3577b34da6a3ce929d0e0e4736"
        )


class _FakeAux:
    def __init__(
        self, input_len, output_len, first_token_cost_time=8.5, cost_time=20.0
    ):
        self.input_len = input_len
        self.output_len = output_len
        self.first_token_cost_time = first_token_cost_time
        self.cost_time = cost_time


class _FakeOut:
    def __init__(
        self,
        finished,
        input_len=8,
        output_len=3,
        first_token_cost_time=8.5,
        cost_time=20.0,
    ):
        self.finished = finished
        self.aux_info = _FakeAux(
            input_len, output_len, first_token_cost_time, cost_time
        )


class _AsyncReturn:
    def __init__(self, value):
        self._value = value

    async def __call__(self, *args, **kwargs):
        return self._value


class _FakeClientSpan:
    """Mirrors tracing.ClientSpanHandle: idempotent finish, writes dropped after."""

    def __init__(self):
        self.attributes = {}
        self.status = None
        self.error_type = None
        self.finished = False

    def set_attribute(self, key, value):
        if not self.finished:
            self.attributes[key] = value

    def finish(self, error=None, error_type=""):
        if self.finished:
            return
        self.finished = True
        if error is not None or error_type:
            self.status = "ERROR"
            self.error_type = error_type or type(error).__name__
        else:
            self.status = "OK"


class _FakeTraceState:
    """Mirrors the request completion contract used during stream teardown."""

    def __init__(self, settled_ok=None, renderer_completed=False):
        self.settled_ok = settled_ok
        self.renderer_completed = renderer_completed

    def set_attribute(self, key, value):
        pass


class _FakeRpcError(grpc.RpcError):
    def __init__(self, status):
        self._status = status

    def code(self):
        return self._status

    def details(self):
        return "injected terminal RPC error"

    def trailing_metadata(self):
        return {}


class _SpanAwareStub:
    """Yields `total` responses; the last one carries the engine finished flag."""

    def __init__(self, total, finish_last=True, terminal_error=None):
        self._total = total
        self._finish_last = finish_last
        self._terminal_error = terminal_error
        self.iterator = None

    def GenerateStreamCall(self, input_pb, timeout=None, metadata=None):
        total, finish_last, terminal_error = (
            self._total,
            self._finish_last,
            self._terminal_error,
        )

        class _Iterator:
            def __init__(self):
                self.cancelled = False
                self.code_waited = False
                self.code_resolved = False
                self.events = []
                self._terminal_status = None
                self._terminal_ready = asyncio.Event()

            def __aiter__(self):
                return self._gen()

            def cancel(self):
                self.events.append("cancel")
                if self._terminal_ready.is_set():
                    return False
                self.cancelled = True
                self._terminal_status = StatusCode.CANCELLED
                self._terminal_ready.set()
                return True

            async def code(self):
                self.code_waited = True
                self.events.append("code")
                await self._terminal_ready.wait()
                self.code_resolved = True
                return self._terminal_status

            async def _gen(self):
                for i in range(total):
                    outputs_pb = GenerateOutputsPB()
                    output_pb = outputs_pb.flatten_output
                    output_pb.output_ids.data_type = TensorPB.DataType.INT32
                    output_pb.output_ids.shape.extend([1, i + 1])
                    output_pb.output_ids.int32_data = struct.pack(
                        "<" + "i" * (i + 1), *range(i + 1)
                    )
                    aux_info = output_pb.aux_info.add()
                    aux_info.iter_count = i + 1
                    aux_info.input_len = 8
                    aux_info.output_len = i + 1
                    output_pb.finished.extend([finish_last and i == total - 1])
                    if finish_last and i == total - 1:
                        # The real server can settle independently while the
                        # Python message iterator remains suspended at yield.
                        self._terminal_status = StatusCode.OK
                        asyncio.get_running_loop().call_soon(self._terminal_ready.set)
                    yield outputs_pb
                if terminal_error is not None:
                    self._terminal_status = terminal_error.code()
                    self._terminal_ready.set()
                    raise terminal_error
                self._terminal_status = StatusCode.OK
                self._terminal_ready.set()

        self.iterator = _Iterator()
        return self.iterator


class ClientSpanSettlementTest(TestCase):
    """Regression guard for the CLIENT span settlement timing and status.

    render_response_stream breaks out of its `async for` as soon as every
    sequence has a finish_reason, so enqueue() stays suspended on its last
    yield until aclose()/GC injects GeneratorExit. Settling the span there
    marked successful requests Cancelled, dropped the usage attributes and
    pushed the span end past the root span.

    The renderer now closes its owned upstream explicitly. On engine completion,
    enqueue waits for grpc.aio's terminal status before ending the CLIENT span.
    A teardown after the root already settled is retained only as a fail-open
    fallback for other consumers.
    """

    USAGE_KEYS = (
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.prompt_tokens",
        "gen_ai.usage.completion_tokens",
        "gen_ai.usage.total_tokens",
    )

    def _build_client(
        self, span, total, finish_last=True, trace_state=None, terminal_error=None
    ):
        client = ModelRpcClient(["127.0.0.1:1234"], {}, 0, False)
        stub = _SpanAwareStub(total, finish_last, terminal_error)
        client._channel_pool = MagicMock()
        client._channel_pool.get = _AsyncReturn(MagicMock())
        token = CURRENT_TRACE_STATE.set(trace_state)
        self.addCleanup(CURRENT_TRACE_STATE.reset, token)
        patcher_stub = patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=stub,
        )
        patcher_span = patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.start_client_span",
            return_value=(span, []),
        )
        self.addCleanup(patcher_stub.stop)
        self.addCleanup(patcher_span.stop)
        patcher_stub.start()
        patcher_span.start()
        client._test_stub = stub
        return client

    @staticmethod
    def _make_input():
        return GenerateInput(
            token_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            generate_config=GenerateConfig(is_streaming=True),
            request_id=7,
            mm_inputs=[],
        )

    def test_engine_reported_finished_predicate(self):
        self.assertFalse(_engine_reported_finished(None))
        self.assertFalse(_engine_reported_finished(GenerateOutputs()))
        outputs = GenerateOutputs()
        outputs.generate_outputs = [_FakeOut(True), _FakeOut(False)]
        self.assertFalse(_engine_reported_finished(outputs))
        outputs.generate_outputs = [_FakeOut(True), _FakeOut(True)]
        self.assertTrue(_engine_reported_finished(outputs))

    def test_usage_attributes_skip_non_positive(self):
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [_FakeOut(True, input_len=0, output_len=5)]
        _record_client_span_usage(span, outputs)
        self.assertEqual(span.attributes, {})
        outputs.generate_outputs = [_FakeOut(True, input_len=8, output_len=3)]
        _record_client_span_usage(span, outputs)
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)
        self.assertEqual(span.attributes["gen_ai.usage.total_tokens"], 11)

    def test_usage_attributes_sum_all_choices(self):
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [
            _FakeOut(True, input_len=8, output_len=3),
            _FakeOut(True, input_len=8, output_len=5),
        ]

        _record_client_span_usage(span, outputs)

        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 8)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 8)
        self.assertEqual(span.attributes["gen_ai.usage.prompt_tokens"], 8)
        self.assertEqual(span.attributes["gen_ai.usage.completion_tokens"], 8)
        self.assertEqual(span.attributes["gen_ai.usage.total_tokens"], 16)

    def test_usage_attributes_skip_inconsistent_choices(self):
        for choices in (
            [
                _FakeOut(True, input_len=8, output_len=3),
                _FakeOut(True, input_len=9, output_len=5),
            ],
            [
                _FakeOut(True, input_len=8, output_len=3),
                _FakeOut(True, input_len=8, output_len=0),
            ],
        ):
            with self.subTest(choices=choices):
                span = _FakeClientSpan()
                outputs = GenerateOutputs()
                outputs.generate_outputs = choices
                _record_client_span_usage(span, outputs)
                self.assertEqual(span.attributes, {})

    def test_engine_latency_attributes_from_single_sequence_stream(self):
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [
            _FakeOut(
                True,
                output_len=5,
                first_token_cost_time=8.5,
                cost_time=20.0,
            )
        ]

        _record_client_span_latency(span, outputs)

        self.assertEqual(span.attributes["rtp_llm.engine.time_to_first_token_ms"], 8.5)
        self.assertEqual(
            span.attributes["rtp_llm.engine.time_per_output_token_ms"], 2.875
        )

    def test_engine_latency_attributes_require_coherent_aux_info(self):
        cases = (
            _FakeOut(True, output_len=0),
            _FakeOut(True, output_len=1),
            _FakeOut(True, output_len=5, first_token_cost_time=0),
            _FakeOut(True, output_len=5, first_token_cost_time=8.5, cost_time=8.0),
        )
        for output in cases:
            with self.subTest(output=output):
                span = _FakeClientSpan()
                outputs = GenerateOutputs(generate_outputs=[output])
                _record_client_span_latency(span, outputs)
                if output.aux_info.output_len == 1:
                    self.assertEqual(
                        span.attributes["rtp_llm.engine.time_to_first_token_ms"],
                        8.5,
                    )
                elif output.aux_info.cost_time < output.aux_info.first_token_cost_time:
                    self.assertEqual(
                        span.attributes["rtp_llm.engine.time_to_first_token_ms"],
                        8.5,
                    )
                else:
                    self.assertEqual(span.attributes, {})
                self.assertNotIn(
                    "rtp_llm.engine.time_per_output_token_ms", span.attributes
                )

    def test_multi_return_shares_ttft_but_omits_ambiguous_tpot(self):
        """n>1 rides one physical stream: TPOT per sequence is not a span value."""
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [
            _FakeOut(True, output_len=5, first_token_cost_time=8.5, cost_time=20.0),
            _FakeOut(True, output_len=7, first_token_cost_time=8.5, cost_time=26.0),
        ]

        _record_client_span_latency(span, outputs)

        self.assertEqual(span.attributes["rtp_llm.engine.time_to_first_token_ms"], 8.5)
        self.assertNotIn("rtp_llm.engine.time_per_output_token_ms", span.attributes)

    def test_multi_return_disagreeing_on_first_token_omits_latency(self):
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [
            _FakeOut(True, output_len=5, first_token_cost_time=8.5, cost_time=20.0),
            _FakeOut(True, output_len=5, first_token_cost_time=9.5, cost_time=20.0),
        ]

        _record_client_span_latency(span, outputs)

        self.assertEqual(span.attributes, {})

    def test_multi_return_with_one_empty_sequence_omits_latency(self):
        span = _FakeClientSpan()
        outputs = GenerateOutputs()
        outputs.generate_outputs = [
            _FakeOut(True, output_len=5, first_token_cost_time=8.5, cost_time=20.0),
            _FakeOut(True, output_len=0, first_token_cost_time=8.5, cost_time=20.0),
        ]

        _record_client_span_latency(span, outputs)

        self.assertEqual(span.attributes, {})

    def test_consumer_break_after_finished_keeps_span_ok(self):
        """The final frame is withheld until the physical RPC reaches OK."""
        span = _FakeClientSpan()
        client = self._build_client(span, total=3)

        async def run():
            gen = client.enqueue(self._make_input())
            async for outputs in gen:
                if outputs.generate_outputs and outputs.generate_outputs[0].finished:
                    self.assertTrue(client._test_stub.iterator.code_resolved)
                    self.assertEqual(client._test_stub.iterator.events, ["code"])
                    break  # mirrors render_response_stream's _check_all_finished
            self.assertFalse(span.finished, "CLIENT must cover RPC termination")
            await gen.aclose()  # injects GeneratorExit at the suspended yield
            self.assertTrue(client._test_stub.iterator.code_waited)
            self.assertEqual(client._test_stub.iterator.events[:2], ["code", "cancel"])

        asyncio.run(run())
        self.assertEqual(span.status, "OK")
        self.assertIsNone(span.error_type)
        self.assertEqual(span.attributes["rpc.response.status_code"], "OK")
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)

    def test_stream_iterated_to_completion_keeps_span_ok(self):
        span = _FakeClientSpan()
        client = self._build_client(span, total=2, finish_last=False)

        async def run():
            async for _ in client.enqueue(self._make_input()):
                pass

        asyncio.run(run())
        self.assertEqual(span.status, "OK")
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)

    def test_break_before_finished_still_reports_cancelled(self):
        """Genuine interruption (client disconnect) must stay Cancelled.

        The root span is unsettled here: the cancellation is still propagating
        outward, so the request has not succeeded.
        """
        span = _FakeClientSpan()
        client = self._build_client(span, total=3, finish_last=False)

        async def run():
            gen = client.enqueue(self._make_input())
            async for _ in gen:
                break
            self.assertFalse(span.finished, "no finished flag seen yet")
            await gen.aclose()
            self.assertTrue(client._test_stub.iterator.cancelled)
            self.assertTrue(client._test_stub.iterator.code_waited)
            self.assertEqual(client._test_stub.iterator.events[:2], ["cancel", "code"])

        asyncio.run(run())
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.error_type, "Cancelled")
        self.assertEqual(span.attributes["rpc.response.status_code"], "CANCELLED")
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)

    def test_stop_word_break_with_renderer_milestone_keeps_span_ok(self):
        """Stop-word truncation is normal before the root span is settled.

        Context-dependent tokenization can make a string stop word miss the
        engine's token-level list, so the renderer can break while the engine is
        still generating. Its explicit milestone keeps that cleanup path OK.
        """
        span = _FakeClientSpan()
        client = self._build_client(
            span,
            total=3,
            finish_last=False,
            trace_state=_FakeTraceState(renderer_completed=True),
        )

        async def run():
            gen = client.enqueue(self._make_input())
            async for _ in gen:
                break  # renderer hit a stop word
            await gen.aclose()

        asyncio.run(run())
        self.assertEqual(span.status, "OK")
        self.assertIsNone(span.error_type)
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)

    def test_break_with_failed_root_reports_cancelled(self):
        """Root span settled with an error keeps the child Cancelled."""
        span = _FakeClientSpan()
        client = self._build_client(
            span, total=3, finish_last=False, trace_state=_FakeTraceState(False)
        )

        async def run():
            gen = client.enqueue(self._make_input())
            async for _ in gen:
                break
            await gen.aclose()

        asyncio.run(run())
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.error_type, "Cancelled")

    def test_response_conversion_error_cancels_before_waiting_for_code(self):
        span = _FakeClientSpan()
        client = self._build_client(span, total=2, finish_last=False)
        call_count = 0

        def fail_second_response(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("conversion failed")
            return trans_output(*args)

        async def run():
            with patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.trans_output",
                side_effect=fail_second_response,
            ):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    async for _ in client.enqueue(self._make_input()):
                        pass

        asyncio.run(run())
        iterator = client._test_stub.iterator
        self.assertEqual(iterator.events[:2], ["cancel", "code"])
        self.assertEqual(span.attributes["rpc.response.status_code"], "CANCELLED")
        self.assertEqual(span.status, "ERROR")
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)

    def test_rpc_error_after_output_keeps_last_confirmed_usage(self):
        span = _FakeClientSpan()
        client = self._build_client(
            span,
            total=1,
            finish_last=False,
            terminal_error=_FakeRpcError(StatusCode.UNAVAILABLE),
        )

        async def run():
            with self.assertRaisesRegex(Exception, "injected terminal RPC error"):
                async for _ in client.enqueue(self._make_input()):
                    pass

        asyncio.run(run())
        self.assertEqual(span.attributes["rpc.response.status_code"], "UNAVAILABLE")
        self.assertEqual(span.status, "ERROR")
        self.assertEqual(span.error_type, "RpcError")
        for key in self.USAGE_KEYS:
            self.assertIn(key, span.attributes)

    def test_request_completed_normally_predicate(self):
        self.assertFalse(_request_completed_normally(None))
        self.assertFalse(_request_completed_normally(_FakeTraceState(None)))
        self.assertFalse(_request_completed_normally(_FakeTraceState(False)))
        self.assertTrue(_request_completed_normally(_FakeTraceState(True)))
        self.assertTrue(
            _request_completed_normally(_FakeTraceState(None, renderer_completed=True))
        )


if __name__ == "__main__":
    setup_logging()
    main()
