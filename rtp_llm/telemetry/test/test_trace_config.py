"""JSON 配置契约、旧配置隔离和本地 OTLP 接收端测试。"""

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from rtp_llm.telemetry import config, tracing

MANUAL = {
    "enabled": True,
    "endpoint": "http://127.0.0.1:4318/custom",
    "headers": {"authorization": "fake-test-only"},
}


class TraceConfigTest(unittest.TestCase):
    def setUp(self):
        config._cache_pid = None
        tracing.reset_telemetry_for_test()

    def tearDown(self):
        tracing.reset_telemetry_for_test()
        config._cache_pid = None

    def test_shared_contract(self):
        cases = json.loads(
            Path(__file__).with_name("trace_config_cases.json").read_text()
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                if "error" in case:
                    with self.assertRaises(config.TraceConfigError) as error:
                        config.parse_trace_config(case["raw"])
                    self.assertEqual(error.exception.code, case["error"])
                else:
                    parsed = config.parse_trace_config(case["raw"])
                    self.assertEqual(parsed.enabled, case["enabled"])
                    self.assertEqual(parsed.sampler_ratio, case.get("ratio", 1.0))

    def test_region_mapping_and_no_mixed_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regions.json"
            path.write_text(
                json.dumps(
                    {
                        "regions": {
                            "cn-test": {
                                "endpoint": "http://localhost:4318/region",
                                "headers": "authorization=Bearer%20fake+x%3D",
                            }
                        },
                        "fallbacks": {"cn-": "cn-test"},
                    }
                )
            )
            values = {
                "enabled": True,
                "region": "cn-other",
                "region_config_file": str(path),
            }
            parsed = config.parse_trace_config(json.dumps(values))
            self.assertEqual(parsed.source, "region")
            self.assertEqual(parsed.headers["authorization"], "Bearer fake+x=")
            with self.assertRaises(TypeError):
                parsed.headers["x"] = "bad"
            for overrides in (
                {"endpoint": MANUAL["endpoint"]},
                {"headers": MANUAL["headers"]},
            ):
                with self.assertRaises(config.TraceConfigError) as error:
                    config.parse_trace_config(json.dumps({**values, **overrides}))
                self.assertEqual(error.exception.code, "incomplete_manual")
            path.write_text("[]")
            self.assertTrue(
                config.parse_trace_config(json.dumps({**values, **MANUAL})).enabled
            )
            with self.assertRaises(config.TraceConfigError):
                config.parse_trace_config(json.dumps(values))
            self.assertFalse(
                config.parse_trace_config(
                    json.dumps({**values, "enabled": False})
                ).enabled
            )
            self.assertFalse(
                config.parse_trace_config(json.dumps(values), tp_rank=1).enabled
            )

    def test_missing_region_file_and_certificate(self):
        for values in (
            {
                "enabled": True,
                "region": "cn-test",
                "region_config_file": "/does-not-exist",
            },
            {**MANUAL, "certificate": "/does-not-exist"},
        ):
            with self.assertRaises(config.TraceConfigError):
                config.parse_trace_config(json.dumps(values))

    def test_once_per_process_warning_and_redaction(self):
        secret = "unique_fake_secret_789"
        with mock.patch.dict(
            os.environ, {config.CONFIG_ENV: '{"' + secret + '":true}'}
        ):
            with self.assertLogs(config._LOGGER, "WARNING") as logs:
                self.assertFalse(config.load_trace_config("frontend").enabled)
                self.assertFalse(config.load_trace_config("frontend").enabled)
            self.assertEqual(len(logs.output), 1)
            self.assertNotIn(secret, " ".join(logs.output))
        parsed = config.parse_trace_config(
            json.dumps({**MANUAL, "headers": {"x": secret}})
        )
        self.assertNotIn(secret, repr(parsed))

    def test_non_owner_rank_has_no_warning(self):
        with mock.patch.dict(
            os.environ, {config.CONFIG_ENV: "invalid"}
        ), mock.patch.object(config._LOGGER, "warning") as warn:
            self.assertFalse(config.load_trace_config("backend", 1).enabled)
            warn.assert_not_called()

    def test_old_switch_cannot_enable(self):
        with mock.patch.dict(
            os.environ, {config.CONFIG_ENV: "", "RTP_LLM_OTEL_TRACE_ENABLE": "1"}
        ):
            with mock.patch.object(tracing, "_init_with_exporter_locked") as init:
                self.assertFalse(tracing.init_telemetry("frontend"))
                init.assert_not_called()

    def test_environment_restored_on_constructor_failure(self):
        values = {config.CONFIG_ENV: json.dumps(MANUAL), "OTEL_SDK_DISABLED": "true"}
        with mock.patch.dict(os.environ, values):
            with mock.patch(
                "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
                side_effect=RuntimeError("unique_fake_secret_789"),
            ):
                with self.assertLogs(tracing._LOGGER, "WARNING") as logs:
                    self.assertFalse(tracing.init_telemetry("frontend"))
            self.assertNotIn("unique_fake_secret_789", " ".join(logs.output))
            self.assertEqual(os.environ["OTEL_SDK_DISABLED"], "true")
            self.assertEqual(
                tracing.telemetry_state(), tracing.TelemetryState.INIT_FAILURE
            )

    def test_wire_export_ignores_legacy_configuration(self):
        received = []
        ready = threading.Event()

        class Receiver(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(
                    (
                        self.path,
                        self.headers,
                        self.rfile.read(int(self.headers["Content-Length"])),
                    )
                )
                self.send_response(200)
                self.end_headers()
                ready.set()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Receiver)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            values = {
                **MANUAL,
                "endpoint": f"http://127.0.0.1:{server.server_port}/custom",
                "schedule_delay_ms": 1,
            }
            legacy = {
                config.CONFIG_ENV: json.dumps(values),
                "OTEL_SDK_DISABLED": "true",
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://invalid.example/ignored",
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "authorization=wrong",
                "OTEL_RESOURCE_ATTRIBUTES": "host.ip=wrong,poison=wrong",
                "OTEL_SERVICE_NAME": "wrong",
                "OTEL_TRACES_SAMPLER": "always_off",
                "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT": "0",
            }
            with mock.patch.dict(os.environ, legacy):
                self.assertTrue(tracing.init_telemetry("frontend"))
                span = tracing.start_server_span("wire", {})
                self.assertIsNotNone(span)
                span.finish()
                self.assertTrue(ready.wait(5))
                self.assertTrue(tracing.shutdown_telemetry(5000))
                self.assertEqual(os.environ["OTEL_SDK_DISABLED"], "true")
            path, headers, body = received[0]
            self.assertEqual(path, "/custom")
            self.assertEqual(headers["authorization"], "fake-test-only")
            self.assertEqual(headers["content-type"], "application/x-protobuf")
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                ExportTraceServiceRequest,
            )

            request = ExportTraceServiceRequest.FromString(body)
            resource = {
                a.key: a.value.string_value
                for a in request.resource_spans[0].resource.attributes
            }
            self.assertEqual(resource["service.name"], "rtp_llm_frontend")
            self.assertNotIn("poison", resource)
            self.assertEqual(
                resource["gen_ai.instrumentation.sdk.name"], "loongsuite-genai-utils"
            )
        finally:
            server.shutdown()
            server.server_close()
            worker.join(5)


if __name__ == "__main__":
    unittest.main()
