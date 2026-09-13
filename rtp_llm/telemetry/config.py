"""Trace 启动配置：唯一入口 RTP_LLM_TRACE_CONFIG，不依赖 OpenTelemetry。"""

import ipaddress
import json
import logging
import math
import os
import re
import ssl
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, NoReturn, Optional, Tuple
from urllib.parse import unquote, urlsplit

CONFIG_ENV = "RTP_LLM_TRACE_CONFIG"
_STRING_FIELDS = {
    "region",
    "endpoint",
    "region_config_file",
    "certificate",
    "service_name",
    "scope_version",
}
_INT_DEFAULTS = {
    "max_queue_size": 2048,
    "max_export_batch_size": 512,
    "schedule_delay_ms": 5000,
    "http_timeout_ms": 3000,
}
_FIELDS = _STRING_FIELDS | set(_INT_DEFAULTS) | {"enabled", "sampler_ratio", "headers"}
_RESERVED_HEADERS = {
    "host",
    "content-length",
    "content-type",
    "connection",
    "transfer-encoding",
    "content-encoding",
    "trailer",
    "upgrade",
    "keep-alive",
    "te",
}
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
_LOGGER = logging.getLogger(__name__)


class TraceConfigError(ValueError):
    """仅携带固定字段名和错误码，不保留输入或第三方异常。"""

    def __init__(self, field_name: str, code: str):
        self.field_name = field_name if field_name in _FIELDS else "config"
        self.code = code
        super().__init__(f"{self.field_name}:{code}")


@dataclass(frozen=True, repr=False)
class TraceConfig:
    enabled: bool = False
    sampler_ratio: float = 1.0
    endpoint: str = ""
    headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    certificate: str = ""
    service_name: str = ""
    scope_version: str = ""
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    schedule_delay_ms: int = 5000
    http_timeout_ms: int = 3000
    source: str = "disabled"

    def __repr__(self) -> str:
        return f"TraceConfig(enabled={self.enabled}, source={self.source})"


def _pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise TraceConfigError("config", "duplicate_key")
        result[key] = value
    return result


def _invalid_constant(_value: str) -> NoReturn:
    raise TraceConfigError("config", "invalid_json")


def _decode(raw: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            raw, object_pairs_hook=_pairs, parse_constant=_invalid_constant
        )
    except TraceConfigError:
        raise
    except (ValueError, RecursionError):
        raise TraceConfigError("config", "invalid_json") from None
    if not isinstance(value, dict):
        raise TraceConfigError("config", "expected_object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TraceConfigError(name, "expected_string")
    return value.strip()


def _headers(value: Any) -> Mapping[str, str]:
    if not isinstance(value, dict):
        raise TraceConfigError("headers", "expected_object")
    result = {}
    for name, item in value.items():
        if not _HEADER_NAME.fullmatch(name) or name.lower() in _RESERVED_HEADERS:
            raise TraceConfigError("headers", "invalid_header")
        name = name.lower()
        if name in result:
            raise TraceConfigError("headers", "duplicate_header")
        if (
            not isinstance(item, str)
            or not item.strip()
            or any(
                ord(c) < 32 and c != "\t" or ord(c) == 127 or ord(c) > 255 for c in item
            )
        ):
            raise TraceConfigError("headers", "invalid_header")
        result[name] = item
    return MappingProxyType(result)


def _endpoint(value: str) -> None:
    try:
        url = urlsplit(value)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or "#" in value
            or any(ord(c) <= 32 or ord(c) >= 127 for c in value)
            or "\\" in value
            or re.search(r"%(?![0-9a-fA-F]{2})", value)
        ):
            raise ValueError()
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError()
        if url.netloc.endswith(":"):
            raise ValueError()
        if ":" in url.hostname:
            ipaddress.IPv6Address(url.hostname)
        else:
            host = url.hostname.removesuffix(".")
            labels = host.split(".")
            if any(
                not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
                for label in labels
            ):
                raise ValueError()
            # 与 Java URI 的 host 判定一致：点分数字必须是完整 IPv4。
            if len(labels) > 1 and labels[-1][0].isdigit():
                ipaddress.IPv4Address(host)
    except ValueError:
        raise TraceConfigError("endpoint", "invalid_endpoint") from None


def _region_path(explicit: str) -> Path:
    if explicit:
        return Path(explicit)
    for base in (Path(__file__).absolute().parent, Path.cwd()):
        for _ in range(8):
            candidate = base / "internal_source/rtp_llm/telemetry/trace_regions.json"
            if candidate.is_file():
                return candidate
            base = base.parent
    raise TraceConfigError("region_config_file", "region_file_unavailable")


def _region_values(region: str, path: str) -> Tuple[str, Mapping[str, str], str]:
    try:
        config = _decode(_region_path(path).read_text(encoding="utf-8"))
        regions, fallbacks = config.get("regions", {}), config.get("fallbacks", {})
        if not isinstance(regions, dict) or not isinstance(fallbacks, dict):
            raise TraceConfigError("region", "invalid_region_config")
        entry = regions.get(region)
        if entry is None:
            for prefix, target in fallbacks.items():
                if not isinstance(target, str):
                    raise TraceConfigError("region", "invalid_region_config")
                if region.startswith(prefix):
                    entry = regions.get(target)
                    break
        if not isinstance(entry, dict):
            raise TraceConfigError("region", "region_not_found")
        endpoint = _string(entry.get("endpoint", ""), "endpoint")
        certificate = _string(entry.get("certificate", ""), "certificate")
        raw_headers = entry.get("headers", "")
        if not isinstance(raw_headers, str):
            raise TraceConfigError("headers", "invalid_region_config")
        pairs = []
        if raw_headers.strip():
            for item in raw_headers.split(","):
                name, sep, value = item.strip().partition("=")
                if not sep or re.search(r"%(?![0-9a-fA-F]{2})", value):
                    raise TraceConfigError("headers", "invalid_header")
                pairs.append((name.strip(), unquote(value.strip(), errors="strict")))
        return endpoint, _headers(_pairs(pairs)), certificate
    except (OSError, UnicodeError):
        raise TraceConfigError(
            "region_config_file", "region_file_unavailable"
        ) from None


def parse_trace_config(raw: Optional[str], tp_rank: int = 0) -> TraceConfig:
    """纯解析入口；异常只含安全错误码。非负责 rank 不读取区域文件。"""
    if not raw or not raw.strip():
        return TraceConfig()
    values = _decode(raw)
    if values.keys() - _FIELDS:
        raise TraceConfigError("config", "unknown_field")
    strings = {name: _string(values.get(name, ""), name) for name in _STRING_FIELDS}
    enabled = values.get("enabled", False)
    if type(enabled) is not bool:
        raise TraceConfigError("enabled", "expected_boolean")
    ratio = values.get("sampler_ratio", 1.0)
    if type(ratio) not in (int, float):
        raise TraceConfigError("sampler_ratio", "expected_number")
    if not 0 <= ratio <= 1 or not math.isfinite(ratio):
        raise TraceConfigError("sampler_ratio", "out_of_range")
    ints = {}
    for name, default in _INT_DEFAULTS.items():
        value = values.get(name, default)
        if type(value) is not int:
            raise TraceConfigError(name, "expected_integer")
        # 三端一致的上界，避免 Java int 和时间单位转换溢出。
        if not 0 < value <= 2147483647:
            raise TraceConfigError(name, "out_of_range")
        ints[name] = value
    if ints["max_export_batch_size"] > ints["max_queue_size"]:
        raise TraceConfigError("max_export_batch_size", "batch_exceeds_queue")
    headers = _headers(values.get("headers", {}))
    if not enabled or tp_rank != 0:
        return TraceConfig()
    endpoint, certificate = strings["endpoint"], strings["certificate"]
    source = "manual"
    if endpoint or headers:
        if not endpoint or not headers:
            raise TraceConfigError("config", "incomplete_manual")
    else:
        if not strings["region"]:
            raise TraceConfigError("region", "missing_destination")
        endpoint, headers, region_certificate = _region_values(
            strings["region"], strings["region_config_file"]
        )
        certificate = certificate or region_certificate
        source = "region"
    if not endpoint or not headers:
        raise TraceConfigError("config", "incomplete_destination")
    _endpoint(endpoint)
    if certificate:
        try:
            ssl.create_default_context(cafile=certificate)
        except (OSError, ValueError):
            raise TraceConfigError("certificate", "invalid_certificate") from None
    return TraceConfig(
        enabled=True,
        sampler_ratio=float(ratio),
        endpoint=endpoint,
        headers=headers,
        certificate=certificate,
        service_name=strings["service_name"],
        scope_version=strings["scope_version"],
        source=source,
        **ints,
    )


def package_scope_version() -> str:
    try:
        from importlib.metadata import version

        return version("rtp_llm")
    except Exception:
        return ""


_cache_lock = threading.Lock()
_cache_pid: Optional[int] = None
_cache = TraceConfig()


def load_trace_config(role: str, tp_rank: int = 0) -> TraceConfig:
    """每进程只读一次；错误仅关闭 Trace，最多告警一次，不打印原输入。"""
    global _cache_pid, _cache
    if tp_rank != 0:
        return TraceConfig()
    with _cache_lock:
        if _cache_pid == os.getpid():
            return _cache
        try:
            config = parse_trace_config(os.environ.get(CONFIG_ENV), tp_rank)
            if config.enabled and not config.scope_version:
                config = replace(config, scope_version=package_scope_version())
            _cache = config
        except TraceConfigError as error:
            _LOGGER.warning(
                "Trace 已关闭 role=%s field=%s reason=%s",
                role,
                error.field_name,
                error.code,
            )
            _cache = TraceConfig()
        except Exception:
            _LOGGER.warning(
                "Trace 已关闭 role=%s field=config reason=config_unavailable", role
            )
            _cache = TraceConfig()
        _cache_pid = os.getpid()
        return _cache
