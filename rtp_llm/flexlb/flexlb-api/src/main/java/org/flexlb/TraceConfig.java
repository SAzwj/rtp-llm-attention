package org.flexlb;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.cert.CertificateFactory;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** 与 Python config.py 一致的 JSON 契约；不依赖 Spring 或 SDK 自动配置。 */
record TraceConfig(boolean enabled, double samplerRatio, String endpoint, Map<String, String> headers,
                   String certificate, String serviceName, String scopeVersion, int maxQueueSize,
                   int maxExportBatchSize, int scheduleDelayMs, int httpTimeoutMs, String source) {
    static final String ENV = "RTP_LLM_TRACE_CONFIG";
    private static final Set<String> FIELDS = Set.of("enabled", "sampler_ratio", "endpoint", "headers",
            "certificate",
            "max_queue_size", "max_export_batch_size", "schedule_delay_ms", "http_timeout_ms");
    private static final Set<String> RESERVED = Set.of("host", "content-length", "content-type", "connection",
            "transfer-encoding", "content-encoding", "trailer", "upgrade", "keep-alive", "te");
    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
            .enable(com.fasterxml.jackson.databind.DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    static final class ConfigException extends IllegalArgumentException {
        final String field;
        final String code;

        ConfigException(String field, String code) {
            super(field + ":" + code);
            this.field = field;
            this.code = code;
        }
    }

    TraceConfig {
        headers = Collections.unmodifiableMap(new LinkedHashMap<>(headers));
    }

    @Override
    public String toString() {
        return "TraceConfig(enabled=" + enabled + ", source=" + source + ")";
    }

    static TraceConfig disabled() {
        return new TraceConfig(false, 1.0, "", Map.of(), "", "", "", 2048, 512, 5000, 3000, "disabled");
    }

    static TraceConfig defaultEnabled() {
        return new TraceConfig(true, 1.0, "http://127.0.0.1:4318/v1/traces",
                Map.of("x-trace-test", "temporary"), "", "", "", 2048, 512, 5000, 3000, "temporary");
    }

    static TraceConfig parse(String raw) {
        if (raw == null || raw.isBlank() || raw.strip().equals("{}")) {
            return defaultEnabled();
        }
        JsonNode root = decode(raw);
        root.fieldNames().forEachRemaining(name -> {
            if (!FIELDS.contains(name)) {
                throw new ConfigException("config", "unknown_field");
            }
        });
        JsonNode enabledNode = root.get("enabled");
        if (enabledNode != null && !enabledNode.isBoolean()) {
            throw new ConfigException("enabled", "expected_boolean");
        }
        boolean enabled = enabledNode == null || enabledNode.booleanValue();
        if (!enabled) {
            return disabled();
        }
        JsonNode ratioNode = root.get("sampler_ratio");
        if (ratioNode != null && !ratioNode.isNumber()) {
            throw new ConfigException("sampler_ratio", "expected_number");
        }
        double ratio = ratioNode == null ? 1.0 : ratioNode.doubleValue();
        if (!Double.isFinite(ratio) || ratio < 0 || ratio > 1) {
            throw new ConfigException("sampler_ratio", "out_of_range");
        }
        int queue = positiveInt(root, "max_queue_size", 2048);
        int batch = positiveInt(root, "max_export_batch_size", 512);
        int delay = positiveInt(root, "schedule_delay_ms", 5000);
        int timeout = positiveInt(root, "http_timeout_ms", 3000);
        if (batch > queue) {
            throw new ConfigException("max_export_batch_size", "batch_exceeds_queue");
        }
        String endpoint = string(root, "endpoint");
        String certificate = string(root, "certificate");
        Map<String, String> headers = headers(root.get("headers"));
        String source = "manual";
        if (!endpoint.isEmpty() || !headers.isEmpty()) {
            if (endpoint.isEmpty() || headers.isEmpty()) {
                throw new ConfigException("config", "incomplete_manual");
            }
        } else {
            throw new ConfigException("config", "incomplete_manual");
        }
        if (endpoint.isEmpty() || headers.isEmpty()) {
            throw new ConfigException("config", "incomplete_destination");
        }
        validateEndpoint(endpoint);
        if (!certificate.isEmpty()) {
            try (var input = Files.newInputStream(Path.of(certificate))) {
                if (CertificateFactory.getInstance("X.509").generateCertificates(input).isEmpty()) {
                    throw new ConfigException("certificate", "invalid_certificate");
                }
            } catch (Exception ignored) {
                throw new ConfigException("certificate", "invalid_certificate");
            }
        }
        return new TraceConfig(true, ratio, endpoint, headers, certificate, "", "",
                queue, batch, delay, timeout, source);
    }

    private static JsonNode decode(String raw) {
        JsonNode node;
        try {
            node = MAPPER.readTree(raw);
        } catch (com.fasterxml.jackson.core.JsonParseException error) {
            String code = error.getOriginalMessage().startsWith("Duplicate field") ? "duplicate_key" : "invalid_json";
            throw new ConfigException("config", code);
        } catch (IOException ignored) {
            // 不附加原异常：Jackson 消息可能包含原始 JSON 和凭证。
            throw new ConfigException("config", "invalid_json");
        }
        if (node == null || !node.isObject()) {
            throw new ConfigException("config", "expected_object");
        }
        return node;
    }

    private static String string(JsonNode node, String name) {
        JsonNode value = node.get(name);
        if (value == null) {
            return "";
        }
        if (!value.isTextual()) {
            throw new ConfigException(name, "expected_string");
        }
        return value.textValue().strip();
    }

    private static int positiveInt(JsonNode node, String name, int fallback) {
        JsonNode value = node.get(name);
        if (value == null) {
            return fallback;
        }
        if (!value.isIntegralNumber()) {
            throw new ConfigException(name, "expected_integer");
        }
        if (!value.canConvertToInt() || value.intValue() <= 0) {
            throw new ConfigException(name, "out_of_range");
        }
        return value.intValue();
    }

    private static Map<String, String> headers(JsonNode node) {
        Map<String, String> result = new LinkedHashMap<>();
        if (node == null) {
            return result;
        }
        if (!node.isObject()) {
            throw new ConfigException("headers", "expected_object");
        }
        node.fields().forEachRemaining(entry -> {
            if (!entry.getValue().isTextual()) {
                throw new ConfigException("headers", "invalid_header");
            }
            addHeader(result, entry.getKey(), entry.getValue().textValue());
        });
        return result;
    }

    private static void addHeader(Map<String, String> result, String name, String value) {
        String normalized = name.toLowerCase(Locale.ROOT);
        if (!name.matches("[!#$%&'*+.^_`|~0-9A-Za-z-]+") || RESERVED.contains(normalized)
                || value.isBlank() || !value.equals(value.strip())
                || value.chars().anyMatch(c -> c < 32 && c != '\t' || c == 127 || c > 255)) {
            throw new ConfigException("headers", "invalid_header");
        }
        if (result.putIfAbsent(normalized, value) != null) {
            throw new ConfigException("headers", "duplicate_header");
        }
    }

    private static void validateEndpoint(String endpoint) {
        try {
            URI uri = new URI(endpoint);
            String scheme = uri.getScheme();
            if (!("http".equalsIgnoreCase(scheme) || "https".equalsIgnoreCase(scheme))
                    || uri.getHost() == null || uri.getRawUserInfo() != null || uri.getRawFragment() != null
                    || uri.getPort() == 0 || uri.getPort() > 65535 || uri.getRawAuthority().endsWith(":")
                    || endpoint.chars().anyMatch(c -> c <= 32 || c >= 127)) {
                throw new IllegalArgumentException();
            }
        } catch (Exception ignored) {
            throw new ConfigException("endpoint", "invalid_endpoint");
        }
    }

}
