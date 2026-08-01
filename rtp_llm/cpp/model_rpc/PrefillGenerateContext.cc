#include "rtp_llm/cpp/model_rpc/PrefillGenerateContext.h"

using grpc::Status;
using grpc::ClientContext;

namespace rtp_llm {

PrefillStatInfo::ExecuteStage PrefillStatInfo::saveStage() const {
    return stage;
}

void PrefillStatInfo::restoreStage(PrefillStatInfo::ExecuteStage stage_) {
    stage = stage_;
}

void PrefillStatInfo::nextStage() {
    stage             = static_cast<PrefillStatInfo::ExecuteStage>(static_cast<int>(stage) + 1);
    auto cost_time_us = currentTimeUs() - begin_time;
    begin_time        = currentTimeUs();
    switch (stage) {
        case getRpcConnection: {
            break;
        }
        case multimodalProcess: {
            get_rpc_connection_rt_us += cost_time_us;
            break;
        }
        case remoteAllocateResource: {
            multimodal_process_rt_us += cost_time_us;
            break;
        }
        case enqueueRequest: {
            remote_allocate_resource_rt_us += cost_time_us;
            break;
        }
        case remoteLoadCacheStart: {
            enqueue_request_rt_us += cost_time_us;
            break;
        }
        case pollLocalOutput: {
            remote_load_cache_start_rt_us += cost_time_us;
            break;
        }
        case remoteLoadCacheEnd: {
            poll_local_output_rt_us += cost_time_us;
            break;
        }
        case RemoteGenerate: {
            remote_load_cache_end_rt_us += cost_time_us;
            break;
        }
        case pollRemoteOutput: {
            remote_generate_rt_us += cost_time_us;
            break;
        }
        case finish: {
            poll_remote_output_rt_us += cost_time_us;
            break;
        }
        default: {
            RTP_LLM_CHECK_WITH_INFO(false, "error stage");
        }
    }
}

PrefillGenerateContext::~PrefillGenerateContext() {
    reportTime();
    closeGrpcStream();
    stopStream();
}

void PrefillGenerateContext::stopStream() {
    if (stream_) {
        stream_->releaseKVCacheForPDSep();
        // if is waiting, cancel it
        meta->dequeue(request_id, stream_);
        if (stream_->getStatus() != StreamState::FINISHED) {
            stream_->reportError(ErrorCode::CANCELLED, "cancel stream");
        }
        // if is running, waiting util done
        while (stream_->getStatus() == StreamState::RUNNING) {
            RTP_LLM_LOG_DEBUG("waiting prefill stream [%d] running done to cancel",
                              stream_->generateInput()->request_id);
            usleep(1000);
        }
        // stream status will only be set to finished by scheduler.
        markRequestEnd();
        stream_.reset();
    }
}
grpc::Status PrefillGenerateContext::closeGrpcStream(const std::string& attempt_error_override) {
    if (grpc_stream_closed) {
        return last_grpc_stream_closed_status;
    }
    grpc_stream_closed = true;
    if (cancelled() && client_context) {
        client_context->TryCancel();
    }
    if (client_stream) {
        client_stream->WritesDone();
        last_grpc_stream_closed_status = client_stream->Finish();
    } else {
        last_grpc_stream_closed_status = grpc::Status::OK;
    }
    // P->D CLIENT span reflects the bidi-stream terminal status;
    // idempotent, destructor of the guard is the final fallback.
    if (pd_client_span_guard) {
        // Session stage breakdown (values already accumulated by
        // PrefillStatInfo::nextStage for kmonitor): the span covers the whole
        // ALLOCATE -> load-cache -> GENERATE session. Keys mirror the
        // PrefillStatInfo field names 1:1; these three dominate the span
        // duration (allocate RTT + wait local prefill + wait remote decode
        // token stream), so their sum ~= span length minus us-level stages.
        // Written here so the final retry attempt's guard gets the settled
        // values.
        // pollRemoteOutput() calls closeGrpcStream() as its own last step,
        // i.e. before the stage is settled by the trailing nextStage() in
        // GenerateStreamCall (and this method is idempotent, so the value
        // would stay 0 forever). Settle the in-flight stage locally without
        // touching stat_info so the kmonitor path keeps its own accounting.
        int64_t poll_remote_output_rt_us = stat_info.poll_remote_output_rt_us;
        if (stat_info.stage == PrefillStatInfo::pollRemoteOutput) {
            poll_remote_output_rt_us += currentTimeUs() - stat_info.begin_time;
        }
        pd_client_span_guard->setAttribute(telemetry::kAttrRtpLlmAllocateRtUs,
                                           stat_info.remote_allocate_resource_rt_us);
        pd_client_span_guard->setAttribute(telemetry::kAttrRtpLlmPollLocalOutputRtUs,
                                           stat_info.poll_local_output_rt_us);
        pd_client_span_guard->setAttribute(telemetry::kAttrRtpLlmPollRemoteOutputRtUs, poll_remote_output_rt_us);
        pd_client_span_guard->setAttribute(telemetry::kAttrRpcResponseStatusCode,
                                           telemetry::grpcStatusCodeValue(last_grpc_stream_closed_status.error_code()));
        if (last_grpc_stream_closed_status.ok() && !attempt_error_override.empty()) {
            // Transport Finish()==OK but the caller (CLIENT_GRPC_RET_IF_ERROR)
            // knows this attempt failed (e.g. Read() returned false because the
            // decode side closed the stream cleanly without a response). Without
            // this override the span would be settled as OK and the retry marker
            // of the next attempt dropped by the exactly-once guard — an OK span
            // for a failed attempt.
            pd_client_span_guard->setAttribute(telemetry::kAttrErrorType, attempt_error_override);
            pd_client_span_guard->finish(opentelemetry::trace::StatusCode::kError,
                                         "Prefill-to-decode RPC attempt failed before receiving a response");
        } else if (last_grpc_stream_closed_status.ok()) {
            pd_client_span_guard->finish(opentelemetry::trace::StatusCode::kOk);
        } else {
            const char* error_name = telemetry::grpcStatusCodeName(last_grpc_stream_closed_status.error_code());
            pd_client_span_guard->setAttribute(telemetry::kAttrErrorType, error_name);
            pd_client_span_guard->finish(opentelemetry::trace::StatusCode::kError,
                                         telemetry::grpcStatusDescription(last_grpc_stream_closed_status.error_code()));
        }
    }
    return last_grpc_stream_closed_status;
}

void PrefillGenerateContext::closeGrpcConnection() {
    if (!decode_addr.empty()) {
        resource->rpc_pool.removeConnection(decode_addr);
    }
}

void PrefillGenerateContext::reset() {
    GenerateContext::reset();
    client_stream.reset();
    grpc_stream_closed             = false;
    last_grpc_stream_closed_status = grpc::Status::OK;
}

void PrefillGenerateContext::nextStage() {
    stat_info.nextStage();
}

void PrefillGenerateContext::markRequestEnd() {
    int64_t real_id = request_id;
    if (stream_) {
        real_id = stream_->streamId();
    }
    if (!resource->isTensorParallel()) {
        resource->cache_store->markRequestEnd(std::to_string(real_id));
        return;
    }
    const auto&           prefill_workers = resource->grpc_workers;
    RemoteFinishRequestPB finish_request;
    finish_request.set_request_id(real_id);
    for (int i = 0; i < prefill_workers.size(); i++) {
        auto& prefill_worker = prefill_workers[i];
        auto  connect_status = resource->rpc_pool.getConnection(prefill_worker);
        if (!connect_status.ok()) {
            RTP_LLM_LOG_WARNING("request [%d], get grpc connection for ip %s failed, ignore markRequestEnd for it",
                                real_id,
                                prefill_worker.c_str());
            continue;
        }
        auto          stub = connect_status.value().stub.get();
        ClientContext client_context;
        EmptyPB       response;
        auto          grpc_status = stub->RemoteFinish(&client_context, finish_request, &response);
        if (!grpc_status.ok()) {
            RTP_LLM_LOG_WARNING("request [%d], remote finish for ip %s failed, ignore markRequestEnd for it",
                                real_id,
                                prefill_worker.c_str());
            continue;
        }
    }
}

void PrefillGenerateContext::reportTime() {
    RpcMetricsCollector collector;

    collectBasicMetrics(collector);

    collector.loading_cache_request          = loading_cache_requests;
    collector.get_rpc_connection_rt_us       = stat_info.get_rpc_connection_rt_us;
    collector.remote_allocate_resource_rt_us = stat_info.remote_allocate_resource_rt_us;
    collector.multimodal_process_rt_us       = stat_info.multimodal_process_rt_us;
    collector.enqueue_request_rt_us          = stat_info.enqueue_request_rt_us;
    collector.remote_load_cache_start_rt_us  = stat_info.remote_load_cache_start_rt_us;
    collector.poll_local_output_rt_us        = stat_info.poll_local_output_rt_us;
    collector.remote_load_cache_end_rt_us    = stat_info.remote_load_cache_end_rt_us;
    collector.remote_generate_rt_us          = stat_info.remote_generate_rt_us;
    collector.poll_remote_output_rt_us       = stat_info.poll_remote_output_rt_us;

    reportMetrics(collector);
    metrics_reporter.reset();  // avoid to report metrics in base class
}

}  // namespace rtp_llm
