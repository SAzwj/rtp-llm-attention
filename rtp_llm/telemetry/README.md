# RTP-LLM Trace 部署指南

RTP-LLM 的 HTTP frontend、Dash worker/proxy、C++ P/D/Fusion 后端和 Java FlexLB 使用同一份 JSON 启动契约，导出协议固定为 OTLP/HTTP protobuf。

## 1. 唯一启动入口

环境变量 `RTP_LLM_TRACE_CONFIG` 的值必须是 JSON 对象。每个进程启动时读取一次，不支持热更新。

区域模式：

```bash
export RTP_LLM_TRACE_CONFIG='{"enabled":true,"region":"cn-hangzhou","sampler_ratio":0.1}'
```

手动模式（凭证由发布系统安全注入，以下仅为占位示例）：

```bash
export RTP_LLM_TRACE_CONFIG='{"enabled":true,"endpoint":"https://collector.example/v1/traces","headers":{"authorization":"<secret>"}}'
```

这是不向后兼容的配置变更。旧 `RTP_LLM_OTEL_*`、Trace 相关 `OTEL_*`、FlexLB `-Dotel.*` 不再控制项目的 Trace；只配置旧变量的新进程会保持关闭。必须给 HTTP/Dash、P、D、FlexLB 分别注入 JSON，不能只改其中一个角色。

## 2. 缺省值与校验

| JSON 字段 | 类型 | 缺省值 |
|---|---|---|
| `enabled` | boolean | `false` |
| `sampler_ratio` | number | `1.0`，仅用于没有合法父上下文的新 Trace |
| `region` | string | 空，不使用区域映射 |
| `endpoint` | string | 空；最终必须有完整 HTTP/HTTPS 地址，不自动追加路径 |
| `headers` | object<string,string> | 空对象；启用时最终必须非空 |
| `region_config_file` | string | 自动定位配套内源的区域文件 |
| `certificate` | string | 系统 CA；可显式填写可读、有效的 CA 文件路径 |
| `service_name` | string | 按角色派生；FlexLB 为 `rtp_llm_flexlb` |
| `scope_version` | string | Python/C++ 安装包版本，Java 为空；可显式覆盖 |
| `max_queue_size` | integer | `2048` |
| `max_export_batch_size` | integer | `512` |
| `schedule_delay_ms` | integer | `5000` |
| `http_timeout_ms` | integer | `3000` |

- 未设置、空白、`{}`、缺省 `enabled` 或 `enabled=false`：正常关闭。不读取区域文件、不创建 SDK。
- 拒绝 JSON 语法错误、重复键、未知字段、`null` 和类型强转；`"true"`、`1` 不能代替布尔值。
- 采样率必须为有限数字且在 `[0,1]` 内；`0` 合法，不会替换为默认值。
- 队列、批量、延迟、超时必须是 `[1,2147483647]` 的整数，批量不能超过队列；非法值不做 clamp。
- 字符串配置去除外围空白后为空视为未填写；`headers={}` 视为未填写。
- endpoint 必须有合法主机与端口，不接受 userinfo、fragment、空白、控制字符；`none` 不是地址。
- header 名符合 HTTP token 规则，大小写不敏感且不能重复；值必须非空、可作为 HTTP header 发送，不允许 CR/LF、控制字符或超出 Latin-1 的字符。禁止覆盖 Host、Content-Length、Content-Type 等传输保留项。
- 具体鉴权键由接收端约定，不将某个平台的全部鉴权键强制用于所有接收端。
- 配置校验失败或 SDK 构造失败：告警、关闭该进程 Trace，不阻止推理或调度启动。告警不输出原始 JSON、header 值或带路径/查询参数的 URL。

## 3. 手动配置与区域的优先级

**手动 endpoint+headers 整组优先，禁止混搭凭证。**

1. 提供非空 endpoint 或 headers，即选择手动模式。必须两者配齐，否则告警关闭，即使 region 可以补齐也不回退。
2. 两者完整时忽略区域映射，不读取区域文件；未知 region 或不存在的区域文件不影响手动模式。
3. 未提供手动配置时，必须提供 region。从区域文件取得 endpoint、headers、certificate；JSON 显式 certificate 可覆盖区域 CA。
4. 区域文件不存在、不可读、损坏、区域未命中或最终配置不完整：关闭 Trace，不猜测其他区域。

区域文件沿用内源 `trace_regions.json` 的 `regions`/`fallbacks` 结构和前缀匹配顺序。区域文件的 headers 字符串由适配层解析 percent 编码；JSON 外部接口仅接受 headers 对象。

显式 `region_config_file` 只使用该路径。缺省从模块/程序代码位置、启动工作目录分别向上最多八层寻找 `internal_source/rtp_llm/telemetry/trace_regions.json`，找到即停止。部署建议通过 JSON 指定 Secret 挂载路径；开发机绝对路径不硬编码，区域凭证不打包进镜像。

## 4. 采样与进程边界

三端统一 `ParentBased(TraceIdRatio)`：合法上游上下文携带 sampled 标志时直接跟随；没有合法父上下文才使用本地 `sampler_ratio`。

`sampler_ratio=0` 不是关闭开关：上游已采样的请求仍会被记录。关闭使用 `enabled=false`，上游不能绕过本地关闭或配置校验失败。

发布前提是入口上游可信；本地采样比例不能作为不可信调用方的强制流量上限。

- Python 各 HTTP/Dash 进程独立解析并初始化 SDK。
- 后端 Python 只解析配置，经 pybind 复制为 C++ `TelemetryConfig`，不初始化 Python Provider。C++ 继续仅在实际 `tp_rank==0` 产 span，每个 DP 组独立判断。
- FlexLB 在 Spring 启动前显式初始化自有 SDK，不依赖自动配置或 Java agent。部署不额外挂 agent；JSON 不能关闭独立第三方探针。
- Python 初始化时临时隔离 SDK 隐式环境配置并恢复原值，C++/Java 显式构造 SDK 参数；不再通过旧环境变量传递内部配置。

## 5. 实例身份

Resource 属于本进程 Provider，不沿 traceparent 继承。身份字段不允许由 JSON 伪造。

| 属性 | 来源 |
|---|---|
| `service.name` | JSON 显式值或按角色派生 |
| `service.instance.id` | hostname-pid，hostname 缺失时为 unknown-pid |
| `host.name` | 内核主机名；Python/C++ 用 gethostname，Java 读 procfs |
| `host.ip` | hostname-pid，用于平台实例聚合，**不是 IP** |
| `rtp_llm.pod_ip` | 非空的 `POD_IP` |
| `gen_ai.instrumentation.sdk.name` | 固定 `loongsuite-genai-utils` |
| `rtp_llm.dp_rank` / `rtp_llm.world_rank` | C++ 引擎的真实 rank |

主机名未知时省略 host.name/host.ip，不从 `$HOSTNAME` 或 `/etc/hostname` 回退。启动器仍补全 POD_IP 身份，但不解析/回写 Trace 配置。`gen_ai.engine.index` 保持整数 world_rank。

## 6. 验证与限制

启动成功不等于接收端鉴权成功，不进行启动网络探测。首次导出仍可能因为网络或鉴权失败；导出失败不影响推理，按限频诊断排查，不自动更换地址或凭证。

检查启动日志的角色、manual/region 来源及采样策略；发送受控请求后，用 trace_id 检查父子链、状态、Resource 与阶段时间戳。Fusion、直连 PD、Master 凑批 FetchResponse、Dash proxy、取消和 P 本地完成具有不同拓扑，不以固定 span 数作为通用契约。

配置跨语言测试样例位于 `test/trace_config_cases.json`；Python `test_trace_config.py` 和 Java `OpenTelemetryBootstrapTest` 共同读取。既有生命周期、传播、终态与阶段合成测试保持覆盖。
