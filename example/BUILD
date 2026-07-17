cc_binary(
    name = "otel_http_trace_poc",
    srcs = ["otel_http_trace_poc.cc"],
    deps = [
        "@io_opentelemetry_cpp//api",
        "@io_opentelemetry_cpp//exporters/memory:in_memory_span_exporter",
        "@io_opentelemetry_cpp//exporters/otlp:otlp_http_exporter",
        "@io_opentelemetry_cpp//sdk/src/trace",
    ],
)
