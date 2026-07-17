#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "opentelemetry/exporters/memory/in_memory_span_data.h"
#include "opentelemetry/exporters/memory/in_memory_span_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_options.h"
#include "opentelemetry/sdk/trace/batch_span_processor_factory.h"
#include "opentelemetry/sdk/trace/batch_span_processor_options.h"
#include "opentelemetry/sdk/trace/provider.h"
#include "opentelemetry/sdk/trace/tracer_provider.h"
#include "opentelemetry/sdk/trace/tracer_provider_factory.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/tracer.h"
#include "opentelemetry/trace/tracer_provider.h"

namespace memory_exporter = opentelemetry::exporter::memory;
namespace otlp            = opentelemetry::exporter::otlp;
namespace trace           = opentelemetry::trace;
namespace trace_sdk       = opentelemetry::sdk::trace;

int main(int argc, char** argv) {
    const bool use_memory_exporter = argc > 1 && std::string(argv[1]) == "memory";

    std::shared_ptr<memory_exporter::InMemorySpanData> memory_data;
    std::unique_ptr<trace_sdk::SpanExporter>           exporter;
    std::string                                        exporter_name;
    std::string                                        endpoint;

    if (use_memory_exporter) {
        exporter      = memory_exporter::InMemorySpanExporterFactory::Create(memory_data);
        exporter_name = "in_memory";
    } else {
        otlp::OtlpHttpExporterOptions exporter_options;
        exporter_options.url           = argc > 1 ? argv[1] : "http://127.0.0.1:4318/v1/traces";
        exporter_options.content_type  = otlp::HttpRequestContentType::kBinary;
        exporter_options.console_debug = true;
        endpoint                       = exporter_options.url;
        exporter                       = otlp::OtlpHttpExporterFactory::Create(exporter_options);
        exporter_name                  = "otlp_http";
    }

    trace_sdk::BatchSpanProcessorOptions processor_options;
    processor_options.max_queue_size        = 2048;
    processor_options.schedule_delay_millis = std::chrono::milliseconds(5000);
    processor_options.max_export_batch_size = 512;

    auto processor       = trace_sdk::BatchSpanProcessorFactory::Create(std::move(exporter), processor_options);
    auto provider_unique = trace_sdk::TracerProviderFactory::Create(std::move(processor));
    std::shared_ptr<trace_sdk::TracerProvider> provider(provider_unique.release());

    std::shared_ptr<trace::TracerProvider> api_provider = provider;
    trace_sdk::Provider::SetTracerProvider(api_provider);

    auto tracer = trace::Provider::GetTracerProvider()->GetTracer("rtp_llm_otel_poc");
    {
        auto span = tracer->StartSpan("rtp_llm.otel_poc");
        span->SetAttribute("poc.component", "opentelemetry-cpp");
        span->SetAttribute("poc.exporter", exporter_name.c_str());
        span->SetAttribute("poc.binary", "example/otel_http_trace_poc");
        trace::Scope scope(span);
        span->End();
    }

    const bool flushed        = provider->ForceFlush();
    size_t     exported_spans = 0;
    if (memory_data) {
        exported_spans = memory_data->GetSpans().size();
    }

    provider.reset();
    std::shared_ptr<trace::TracerProvider> none;
    trace_sdk::Provider::SetTracerProvider(none);

    if (use_memory_exporter) {
        std::cout << "otel_http_trace_poc mode=memory flushed=" << flushed << " spans=" << exported_spans << std::endl;
        return flushed && exported_spans == 1 ? 0 : 3;
    }

    std::cout << "otel_http_trace_poc endpoint=" << endpoint << " flushed=" << flushed << std::endl;
    return flushed ? 0 : 2;
}
