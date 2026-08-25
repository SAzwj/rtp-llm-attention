#include <atomic>
#include <future>
#include <mutex>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

using namespace ::testing;

namespace rtp_llm {

class MockRpcGenerateStream: public GenerateStream {
public:
    MockRpcGenerateStream(const std::shared_ptr<GenerateInput>& input,
                          const ModelConfig&                    model_config,
                          const RuntimeConfig&                  runtime_config):
        GenerateStream(input, model_config, runtime_config, ResourceContext{}, nullptr) {}

    MOCK_METHOD((ErrorResult<GenerateOutputs>), nextOutput, (int64_t), (override));
    MOCK_METHOD(void, updateOutput, (const StreamUpdateInfo&), (override));
};

class TestLocalRpcServer: public LocalRpcServer {
public:
    grpc::Status poll(std::shared_ptr<GenerateStream>& stream) {
        return pollStreamOutput(nullptr, "request", nullptr, stream);
    }

    std::future<void> cancellationChecked() {
        return cancellation_checked_.get_future();
    }

    std::atomic<bool> cancelled{false};

protected:
    bool isCancelled(grpc::ServerContext*) const override {
        std::call_once(cancellation_check_once_, [this] { cancellation_checked_.set_value(); });
        return cancelled.load();
    }

private:
    mutable std::once_flag     cancellation_check_once_;
    mutable std::promise<void> cancellation_checked_;
};

std::shared_ptr<MockRpcGenerateStream> createMockRpcStream() {
    auto input             = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    return std::make_shared<MockRpcGenerateStream>(input, model_config, RuntimeConfig{});
}

std::shared_ptr<NormalGenerateStream> createNormalRpcStream() {
    auto input             = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->begin_time_us   = autil::TimeUtility::currentTimeInMicroSeconds();
    input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    return std::make_shared<NormalGenerateStream>(input, model_config, RuntimeConfig{}, ResourceContext{}, nullptr);
}

TEST(LocalRpcServerTest, PollContinuesAfterNoUpdateAndStopsOnFinished) {
    TestLocalRpcServer server;
    auto               mock_stream = createMockRpcStream();
    EXPECT_CALL(*mock_stream, nextOutput(_))
        .WillOnce(Return(ErrorResult<GenerateOutputs>(ErrorCode::OUTPUT_QUEUE_NO_UPDATE, "no update")))
        .WillOnce(Return(ErrorResult<GenerateOutputs>(ErrorCode::FINISHED, "finished")));
    std::shared_ptr<GenerateStream> stream = mock_stream;

    EXPECT_TRUE(server.poll(stream).ok());
}

TEST(LocalRpcServerTest, PollInterruptsBlockedOutputWaitAfterCancellation) {
    TestLocalRpcServer              server;
    auto                            cancellation_checked = server.cancellationChecked();
    std::shared_ptr<GenerateStream> stream               = createNormalRpcStream();
    auto poll_result = std::async(std::launch::async, [&server, &stream] { return server.poll(stream); });

    EXPECT_EQ(cancellation_checked.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    server.cancelled = true;

    const auto wait_status = poll_result.wait_for(std::chrono::seconds(5));
    if (wait_status != std::future_status::ready) {
        stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "test poll cancellation timed out");
    }
    EXPECT_EQ(wait_status, std::future_status::ready);
    EXPECT_EQ(poll_result.get().error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
}

}  // namespace rtp_llm
