# 可复现基准

这里保存当前实现的数值、GPU 链路和显存测量程序。正式结论、输入条件和计量边界见[实现对照报告](../docs/experiments/2026-10-17-幂均值镜像蒸馏实现与估计器对照.md)及[显存报告](../docs/2026-10-17-显存优化与分阶段测量.md)。运行结果写入新的 `runs/` 子目录；GPU 程序通过正常预约协议使用显式设备，不清理其他任务。

| 程序                   | 范围                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------ |
| `hidden_anchor.py`     | 完整中心 sampled BPO 的概率表／hidden 配对；隔离进程测 CPU RSS、耗时及可选 CUDA 峰值 |
| `kernel.py`            | 固定真实条件分布上的三算法 loss／诊断内核                                            |
| `memory.py`            | 同输入 eager／chunked 输出投影的梯度、耗时和存储                                     |
| `memory_native.py`     | 微型随机 Qwen2 的六条件四卡 FSDP 原生链路；`--case` 可选择单项                       |
| `native_validation.py` | 模型／数据可移植性、完整积分和正式长度预算的验证                                     |
| `memory_stages.py`     | 真实 1.5B 模型对、短输入完整 BPO 的两条投影路径及 NVML 采样                          |
| `memory_report.py`     | 对短输入配对运行重新审计并重算分阶段计量                                             |
| `math_memory.py`       | 正常 batch64、训练 7168-token、评测 R16／31744-token 上限的容量测量                  |

`memory_native.py` 的每项采用 2 outer；MDPO/BPO 每轮 2 inner，on-policy 每轮 1 inner，前后各对四题生成两次。微型随机模型用于检查更新、冻结窗口、padding、完整词表与 FSDP 反向，分数不代表任务能力。

`math_memory.py` 从当前数学配方加载 BPO Top-J16＋tail、sampled、full 诊断、默认混合精度和 offload。它保留 batch64／micro1、prompt2048／response7168 和四卡资源设置，缩短为 1 outer、2 inner，以初始化 Adam moments 并执行重复窗口。评测只取前四题，训练前评测关闭，训练后保留每题 16 次和 response31744；`--train` 显式指定训练 parquet，不自动改数据集或筛选输入。预检拒绝超长 prompt。该程序测容量，不比较收敛或准确率。

在已准备 GPU 环境、模型、数据及匹配后端后，从仓库根执行：

```bash
PYTHONPATH=src python benchmarks/memory_native.py \
  --case mdpo-full-full --output runs/mdpo-full-check \
  --backend /path/to/pinned-backend --lease-root /path/to/gpu-leases

PYTHONPATH=src python benchmarks/math_memory.py \
  --output runs/math-memory --model-root /path/to/models \
  --data-root /path/to/data --train /path/to/train.parquet \
  --backend /path/to/pinned-backend --lease-root /path/to/gpu-leases
```

`memory_report.py` 只接受 `memory_stages.py` 的完整配对目录；不能把正常长度单次运行伪装为 eager／chunked 对照。`math_memory.py` 保存解析 Job、完整原生运行、NVML 原始记录与完成审计。测量的瞬时整卡值、worker CUDA peak、host 边界值应按报告定义分别解释。

`hidden_anchor.py` 默认使用 CPU、512 个前缀、词表 8192、hidden128，双方各完成一次小形状预热后以新进程测量，共重复三组。`--precision bf16` 保留低精度投影和 FP32 概率运算；`--gpu 0` 先预约空闲物理 GPU，再在子进程测 CUDA allocated／reserved 峰值。CPU RSS 包含解释器、Torch 和预热的高水位，不能称为显存。此程序不测文件／Ray 搬运或 FSDP。

```bash
python benchmarks/hidden_anchor.py --precision bf16 --output runs/hidden-bf16.json
python benchmarks/memory_native.py --case bpo-sampled-full --center full \
  --anchor-storage hidden --output runs/hidden-native \
  --backend upstream/TA-OPD-v2-20261017 --lease-root ~/.cache/mirroropd/gpu-leases
```

四卡配对时，再用 `--anchor-storage log_probs --center full` 运行同一 canary；不要将默认 Top-J 中心与完整中心视为只改变存储的对照。GPU 命令须等待设备无活动上下文，新增 hidden 原生 canary 尚未执行。
