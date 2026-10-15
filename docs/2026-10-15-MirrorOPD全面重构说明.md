# MirrorOPD 0.2 全面重构说明

本文记录 0.2 接口切换和 0.2.1 启动修复的交付边界及验证证据。工程日常参考见[文档入口](README.md)；历史三算法 50 轮实验与 0.2 正式实验分别记录，不能混为同一结果。

## 版本边界

0.2 删除旧接口、缓存、模式别名和专用 launcher，不提供兼容包装。生产算法为`on_policy`、`mdpo`、`bpo`；模型、数据和配方进入一份严格 Job，运行绑定输入、实际源码与评测身份。理论推导和研究证据继续纳入 Git。

重构前标签`archive/pre-refactor-20261015`对应`74f78a2`；退役源码另有[固定归档](../data/refactor-20261015/legacy-source.tar.gz)。恢复及限制见[维护指南](maintenance.md)。当前模块职责、公式与命令分别由[架构](架构.md)、[算法与实现](算法与实现.md)、[运行指南](运行与复现.md)和[日志参考](日志与评测.md)维护。

## 单进程推理组的启动

训练使用四卡 FSDP 数据并行，每个 vLLM replica 的 executor group 为 world size=1、内部 DP=1。vLLM 0.10 的默认实现先选取空闲 TCP 端口、关闭临时 socket，再等待 worker 启动并监听，存在端口竞争窗口。0.2.1 为这些独立的单进程组配置不同的 FileStore；训练进程组仍使用自己的共享 store。

vLLM 的 EngineCore 通过 spawn 启动，父进程的模块赋值不会自动保留。原生后端选择 `SingletonFileStoreExecutor` 类，在实际 EngineCore 的 executor 初始化入口安装 rendezvous，然后继续使用上游的 worker、消息队列和模型执行实现。较大的 executor group 或内部 DP 会被拒绝，不会误用彼此独立的 store。

新增四卡 canary 实际记录了四个 EngineCore 的四条不同 FileStore URI，完成两次更新及前后评测。270 项测试、干净补丁重放和新 wheel 的隔离检查均通过；所有 `core/` 源文件与 `v0.2.0` 逐字节相同。失败的 TCP 启动和未通过 FileStore 审计的初版尝试保留在[修复验证归档](../data/refactor-20261015/portfix-evidence.tar.gz)中。

更新包后需使用匹配的新后端目录；运行器会拒绝旧补丁，不覆盖旧后端。正式实验重启使用新输出目录，保留失败现场。

## 验证与成本

270 项当前全仓测试已经通过，包含独立目标梯度、归一化端点、缓存 schema 与冻结、padding/EOS/预算末 action、真实模型参数梯度、ragged 运输、不等长与空 rank 全局统计、数据和 scorer、并发预约、事件隔离、完成审计、原生控制流、维护工具契约，以及占用 TCP 端口下的 FileStore 和 executor 初始化位置测试。固定补丁已在干净基线重放后再次运行全部测试。

机器可读结果保存在 [0.2.0 基线摘要](../results/refactor-20261015/summary.json)、[kernel 基准](../results/refactor-20261015/kernel-benchmark.json)、[环境版本](../results/refactor-20261015/environment.json)、[云端日志核验](../results/refactor-20261015/cloud-verification.json) 和 [0.2.1 启动修复验证](../results/refactor-20261015/portfix-verification.json)。[原生验证归档](../data/refactor-20261015/validation-evidence.tar.gz) 包含实际运行配置、输入身份、源码快照、事件、回答、日志和审计结果，由 `data-manifest.json` 记录解压后 SHA-256；归档不包含原模型权重、派生训练 parquet 或 checkpoint；数据准备按输入文件 SHA-256 和记录的配置重建。

wheel 在仓库外目录解包验证：内置 profiles、后端补丁和判分许可证完整；CLI plan 可用；旧 `training` 模块不在 wheel 中。GPU 软件环境仍按已验证环境清单准备，wheel 不代替 CUDA/vLLM/FlashAttention 安装。

GPU kernel 基准使用同一真实模型缓存、V=151665、16 与 56 个前缀，预热 3 次、测量 10 次。54 个新条件覆盖三算法、中心/积分组合和诊断级别；其中 16 个条件与旧实现直接比较。最大逐坐标梯度差 1.94e-7，相对 L2 差 2.77e-6。下表为 56 前缀、full 诊断下的中位时间：

| 算法与估计器                   | 旧实现 ms | 0.2 ms | 旧时间/新时间 |
| ------------------------------ | --------: | -----: | ------------: |
| on-policy，Top-J＋tail sampled |     2.047 |  1.625 |         1.260 |
| MDPO，Top-J＋tail sampled      |     1.469 |  1.646 |         0.893 |
| BPO，Top-J＋tail sampled       |     1.474 |  1.615 |         0.913 |
| MDPO，完整积分                 |     2.004 |  1.394 |         1.438 |
| BPO，完整积分                  |     2.206 |  1.603 |         1.376 |

完整积分复用归一化 target 和概率结果，基准分配峰值下降；部分 Top-J full 诊断分支仍有额外检查成本。这里是 loss 与诊断 kernel 的比较，诊断合同也已清理，不能外推为整模型吞吐加速。不同诊断级别保持梯度一致。

基准输入按块压缩保存，`make data` 恢复后可以重组并校验完整 SHA-256；旧基准程序与精确源码放在验证归档中。新基准与原生矩阵的复现入口为：

```bash
PYTHONPATH=src python experiments/benchmarks/kernel.py \
  --inputs data/refactor-20261015/benchmark-inputs.json \
  --output runs/reproduced-kernel.json --gpu 0 --lease-root /gpu-leases

PYTHONPATH=src python experiments/benchmarks/native_validation.py \
  --output runs/reproduced-native --backend upstream/TA-OPD-v2 \
  --model-root /models --data-root /data --lease-root /gpu-leases
```

七组原生四卡运行通过完整审计，共 22 次真实优化更新：

| 模型与目标                     | 算法      | 更新数 | 评测                                 |
| ------------------------------ | --------- | -----: | ------------------------------------ |
| 真实 1.5B 模型，Top-J sampled  | on-policy |      2 | 训练后 4 题 × 2，短预算              |
| 真实 1.5B 模型，Top-J sampled  | MDPO、BPO |   各 4 | 训练前后各 4 题 × 2，短预算          |
| 不同宽度的微型 Qwen2，完整词表 | MDPO、BPO |   各 4 | 训练前后各 4 题 × 2                  |
| 不同宽度的微型 Qwen2，完整词表 | on-policy |      2 | 训练前后各 4 题 × 2                  |
| 真实 1.5B 模型，原训练长度     | on-policy |      2 | 训练前后各 4 题 × 16，response=31744 |

微型模型使用 JSONL 问答、合法词表 32、学生/教师 hidden size 128/256，以及 padded vocab 64/96。完整词表求和、不同模型尺寸和输出层填充分别进入实际原生链路。初版 head dimension=16 的 fixture 触发了固定软件栈 FlexAttention 的 CUDA 错误，失败日志保留在归档中；当前 fixture 使用学生 head dimension=64，验证范围受该软件栈约束。

长预算运行保持训练 batch=64、prompt=2048、response=7168，测试 prompt=4096、response=31744、context=35840、temperature=0.7、top-p=0.9、每题 16 次采样。初始评测共生成 406285 token，均长约 6348，其中一条实际用满 31744 token；两轮评测耗时约 320 秒和 285 秒。训练更新的梯度范数约 1.18、1.16。所有运行的优化有效 token 计数还与实际 rollout 总 token 数逐 outer 核对。小样本成绩只证明链路，不作任务能力或算法排序结论。

## 适用条件

当前原生后端验证范围为单机 FSDP、SP=1、rollout TP=1；核心要求完整合法词表。支持的模型还受固定版本 HF/verl/vLLM 的能力约束，不能据模型配置可解析就宣称所有架构可运行。checkpoint 保存频率可配置，默认保持原配方关闭；没有增加跨版本或位级一致的恢复承诺。

0.2 正式三组 50 outer 登记使用[已提交的冻结配置](../results/training/math-four-gpu-v02-20261015/plans/requirements.json)：on-policy 共 50 次更新，MDPO/BPO 各 800 次更新；570 题各采样 16 次，在 outer 0、5、10、…、50 评测，共 11 轮。0.2.1 的重启目录为 `runs/math-four-gpu-v02-20261015-attempt-02/{on_policy,mdpo,bpo}`，每组以自己的 `verification.json` 为完成依据。这里保存登记配置，不据此断言正式实验完成；单一 seed、不同 inner 更新量与跨引擎数值误差仍需在研究结论中说明。

## 文档与来源整理

工程文档按当前参考、实验报告和专项分析组织，入口见[docs/README](README.md)。旧`operations/`正文按事实归属提取后删除；`earlier/`和论文源稿移至`theory/`，不对待审推导做去重或合稿。历史数学核验跟随理论来源；有效实验条件、失败与负结果留在所属报告，执行流水留忽略目录。

旧文档中 BPO26 轮快照由已归档 50 轮最终结果补齐，但保留中途图的原始范围；Qwen3 报告撤回没有完整梯度测量支撑的训练效率因果判断；GSM8K 报告保留 BF16 及全局 token 分母错误的独立证据。固定监督/BPO 总报告仍为单文件，实现正确性报告独立维护。

skills 继续复用三个既有入口；maintain-docs 新增实验报告维护引用与骨架，docs/AGENTS 规定报告更新和证据边界。历史文档延期清单清空，现行链接与间距检查不再豁免工程报告。CI 仍验证工程契约，不自动判定数学或科学结论。

完整来源映射、迁移哈希、逐项审阅与本轮验证记录保存在忽略的`runs/docs-rewrite-20261015/`，不新增平行交付报告。本轮没有修改生产算法或重跑历史 GPU 训练；源稿及二进制附件迁移按 SHA-256 核对。

## 已知限制

完整词表缓存仍随有效前缀数与词表大小增长；kernel 收益不等于整模型收益。当前单机 FSDP、SP1/TP1 及单进程推理组验证不能推广为任意架构、多机或词表分片支持。checkpoint 保存开关也不构成完整的中断续训协议。

独立数学参照、CPU 测试、真实 GPU canary 和正式性能结论各有范围。源码或证据缺失的历史运行保留限制，不为获得完整叙述补造结果。
