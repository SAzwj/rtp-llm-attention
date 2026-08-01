#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "grpc++/grpc++.h"
#include "opentelemetry/exporters/memory/in_memory_span_data.h"
#include "opentelemetry/exporters/memory/in_memory_span_exporter_factory.h"
#include "opentelemetry/trace/context.h"

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/telemetry/GrpcTraceCarrier.h"
#include "rtp_llm/cpp/telemetry/RpcTraceHelper.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"

namespace rtp_llm {
namespace telemetry {

namespace trace_api       = opentelemetry::trace;
namespace memory_exporter = opentelemetry::exporter::memory;

namespace {

std::string traceIdHex(const trace_api::SpanContext& span_context) {
    char buf[32];
    span_context.trace_id().ToLowerBase16(buf);
    return std::string(buf, 32);
}

// Minimal RpcService that extracts the propagated context inside a real gRPC
// handler and reports what it saw through the CheckHealth response payload.
class ExtractingHealthService final: public RpcService::Service {
public:
    grpc::Status
    CheckHealth(grpc::ServerContext* context, const EmptyPB* /*request*/, CheckHealthResponsePB* response) override {
        auto remote_context = extractContextFromServerMetadata(context);
        auto span_context   = trace_api::GetSpan(remote_context)->GetContext();
        if (!span_context.IsValid()) {
            response->set_health("invalid");
        } else {
            response->set_health(traceIdHex(span_context) + ":" + (span_context.IsSampled() ? "1" : "0") + ":"
                                 + (span_context.IsRemote() ? "remote" : "local"));
        }
        return grpc::Status::OK;
    }
};

class GrpcPropagationTest: public ::testing::Test {
protected:
    void SetUp() override {
        TelemetryRuntime::shutdown(5000);
        auto            exporter = memory_exporter::InMemorySpanExporterFactory::Create(span_data_);
        TelemetryConfig config;
        config.enabled = true;
        config.role    = "test";
        config.tp_rank = 0;
        ASSERT_TRUE(TelemetryRuntime::initWithExporter(std::move(exporter), config));

        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
        ASSERT_NE(server_, nullptr);
        ASSERT_GT(port_, 0);

        channel_ = grpc::CreateChannel("127.0.0.1:" + std::to_string(port_), grpc::InsecureChannelCredentials());
        stub_    = RpcService::NewStub(channel_);
    }

    void TearDown() override {
        if (server_) {
            server_->Shutdown();
        }
        TelemetryRuntime::shutdown(5000);
    }

    std::shared_ptr<memory_exporter::InMemorySpanData> span_data_;
    ExtractingHealthService                            service_;
    int                                                port_ = 0;
    std::unique_ptr<grpc::Server>                      server_;
    std::shared_ptr<grpc::Channel>                     channel_;
    std::unique_ptr<RpcService::Stub>                  stub_;
};

TEST_F(GrpcPropagationTest, InjectedContextSurvivesRealGrpcTransport) {
    auto tracer = TelemetryRuntime::tracer();
    auto span   = tracer->StartSpan("client_root");
    ASSERT_TRUE(span->GetContext().IsValid());

    grpc::ClientContext client_context;
    injectSpanToClientContext(&client_context, span);

    EmptyPB               request;
    CheckHealthResponsePB response;
    auto                  status = stub_->CheckHealth(&client_context, request, &response);
    ASSERT_TRUE(status.ok()) << status.error_message();

    const std::string expected = traceIdHex(span->GetContext()) + ":1:remote";
    EXPECT_EQ(response.health(), expected);
    span->End();
}

TEST_F(GrpcPropagationTest, NoMetadataYieldsInvalidContext) {
    grpc::ClientContext   client_context;
    EmptyPB               request;
    CheckHealthResponsePB response;
    auto                  status = stub_->CheckHealth(&client_context, request, &response);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(response.health(), "invalid");
}

}  // namespace

}  // namespace telemetry
}  // namespace rtp_llm
