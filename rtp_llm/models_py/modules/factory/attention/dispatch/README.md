# 动态 Decode 注意力后端

使用说明。该功能**默认关闭**，不开启时下面这些逻辑都不参与，引擎仍走原来的静态注意力路径。

## 开启方式

```bash
--enable_dynamic_decode_backend true      # 或 ENABLE_DYNAMIC_DECODE_BACKEND=true
```

启动时先做一次精度检查，保留所有 rank 都通过的后端；随后对每个 CUDA Graph capture bucket，在通过的后端里测速并选最快的建图。

只支持 Python BF16 MHA 模型 + 开启 CUDA Graph + BASE/FP8 KV cache + 单个 FULL KV cache group，且不支持 PP/CP、投机、多模态、MLA、headwise 注意力、DeepEP low-latency。遇到这些情况会打印
`dynamic decode disabled, falling back to static path: ...`，然后正常以静态路径启动。

如果检查执行了但没有任何后端通过，启动失败并报
`dynamic decode precision gate failed`。通过时的日志：

```
decode gate thresholds source=builtin id=h20-qwen3-20261030-v1 sha256=... ...
decode gate local rank=0/1 records_ok=True passed=['XQADecodeImpl', ...]
decode gate WORLD passed=['XQADecodeImpl', ...]
decode gate calibration_coverage={'tested_batches': (1, 3, 8), ...}
```

每个场景的判定明细在准入前打印，可以按 rank 区分是哪个场景、哪个后端没过：

```
decode gate rank=3/8 scenario=b1__s0_p0c64d4 backend=XQADecodeImpl layers=2 failures=[...]
```

## 阈值

内置了一套默认阈值。要换成自己的数值，把一整份 JSON 内联写进 `DYN_DECODE_GATE_THRESHOLDS`；它作用于当前进程实际使用的 profile（BASE KV 对应 BF16 那套，FP8 KV 对应 FP8 那套）：

```bash
export DYN_DECODE_GATE_THRESHOLDS='{
  "calibration_id": "h20-qwen3-my-run-v1",
  "target": {"sm_major": 9, "model_family": "qwen3"},
  "signal_split": 0.5,
  "budget_coefficient": 0.09,
  "roundoff_coefficient": 0.001953125,
  "gain_limit": 0.11,
  "cosine_limit": 0.95,
  "nrmse_limit": 0.33,
  "ulp_limit": null
}'
```

| 键 | 取值范围 | 含义 |
| --- | --- | --- |
| `signal_split` | (0, 1) | 参考能量高于该值时启用 cosine/NRMSE/ULP 判定 |
| `budget_coefficient` | >= 0 | 相对参考贡献允许的误差 |
| `roundoff_coefficient` | >= 0 | 低信号场景的绝对容差下限 |
| `gain_limit` | >= 0 | 输出幅度相对增益上限 |
| `cosine_limit` | (0, 1] | 高信号场景的最小余弦相似度 |
| `nrmse_limit` | >= 0 | 高信号场景的最大 NRMSE |
| `ulp_limit` | `null` 或 > 0 | 可选的高信号 ULP 预算 |
| `calibration_id` | 字符串 | 仅用于日志和报告 |
| `target` | `{sm_major, model_family}` | 仅用于日志和报告；与运行时不一致只打 WARNING，继续运行 |

也可使用 `--dyn_decode_gate_thresholds '<完整JSON>'`，CLI 优先于环境变量。原始字符串经标准配置和进程序列化传给模型，仅在精度门执行时解析；未配置使用内置值，显式空字符串属于非法 JSON。覆盖阈值的来源日志记为 `source=config`。

七个数值键必须写全：缺键、多出未知键、数值越界或 JSON 语法错误都会在执行精度门时启动失败，不会把传入值和内置值混成一套没有标定过的参数。只支持内联 JSON，不读取阈值文件。性能选型固定使用 5% 提升门槛和 5% 候选聚类范围。

候选的平台支持条件在每个 rank 的本地 mask 中判断，再按 WORLD 取交集，所有 rank 按同一注册顺序执行。多 rank 校准前向失败时，失败 worker 立即以退出码 70 终止，由现有进程监督机制终止其他 worker，不再尝试可能与模型 TP 通信错位的 WORLD 状态汇总。

请把 `calibration_id` 和 `target` 写清楚，事后只能靠它们确认这次启动用的是哪套阈值。
