# 四卡数学训练算法核验

核验时间：2026-10-11。核验范围包括 `docs/book-old/chapters/02_four_inner_algorithms.tex`、`02_opd_instantiation.tex`，仓库 `src/mirroropd/training/`、`src/mirroropd/integrations/verl/fsdp.py`，以及当前部署的 `../backends/mirroropd-taopd/verl/verl/trainer/distillation/losses.py`、`workers/engine_workers.py`、`workers/engine/fsdp/transformer_impl.py`。本次仅增加核验记录，没有改动生产代码。

## 算法定义与实现

冻结 outer anchor 记为 $q=\pi_k$，当前 student 为 $p=\pi_{k,n}$，teacher 为 $t$。在 rollout 固定的 prefix 上，定义

$$
U(a)=\log p(a)-\log q(a),\qquad r_k(a)=\log t(a)-\log q(a),\qquad r_\theta(a)=\log t(a)-\log p(a).
$$

所有生产方法都通过停止梯度的系数 $c(a)$ 乘当前 action log-prob 传递梯度，即 $\operatorname{sg}(c(a))\log p(a)$。这个 scalar surrogate 的值不等于真实 KL、镜像目标或平方 residual，跨 inner 的 scalar loss 下降不能证明子问题收敛。

| 生产 loss mode              | 生产系数 $c$                                                                   | 语义                                                            |
| --------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| `mirror_opd_onpolicy`       | $-(r_\theta+\widehat{\mathrm{KL}}(p\Vert t))$                                  | 每轮重新 rollout 后一次完整 optimizer 更新                      |
| `mirror_opd_is`             | $-e^U(r_\theta+\widehat{\mathrm{KL}}(p\Vert t))$                               | 固定 prefix、固定 anchor action，token importance sampling 校正 |
| `mirror_opd_is_no_baseline` | $-e^U r_\theta$                                                                | 去掉条件 baseline 的 Token-IS OPD                               |
| `mirror_mdpo`               | $e^U(U/\eta-r_k)$                                                              | 有限镜像系数 MDPO 的解析 policy-gradient 更新                   |
| `mirror_bpo`                | $(U+\widehat{\mathrm{KL}}(q\Vert p))/\eta-r_k-\widehat{\mathrm{KL}}(q\Vert t)$ | Tail-aware 中心的 BPO 解析更新                                  |
| `mirror_pmd_mean`           | $U-\eta(r_k+\widehat{\mathrm{KL}}(q\Vert t))$                                  | PMD-Mean，区别于 BPO                                            |

MDPO 生产实现符合旧书正式 PG 公式。其 $1/\eta$ 常数 score 项被解析消去，不能再通过对 coefficient 求导补回。实现使用普通 Adam 优化神经网络；旧书独立 categorical logits 下 Fisher 伪逆的自然梯度一步收敛定理不适用于该生产训练。

BPO 生产实现符合旧书 Algorithm 5 的 Tail-aware 解析更新。它没有对近似平方目标的 student center 继续求导。对 Tail-aware center 继续求导会引入额外项，改变既定算法；这一差异在旧书已经明确推导。生产代码当前语义正确。

### 一次 outer 与十六次 inner

部署的 `TrainingWorker.train_mini_batch` 把配置的全局 mini-batch 除以 data parallel 数，然后迭代 epoch 与 mini-batch。单个 outer 中的实际 optimizer 次数为

$$
J_{\mathrm{actual}}=\frac{B\,n}{B_{\mathrm{mini}}}\,E,
$$

其中 $B$ 是 prompt batch，$n$ 是每题 rollout 数，$E$ 是 `ppo_epochs`，要求整除。micro-batch 只负责一个 optimizer 更新内的梯度累积，不增加 $J$。

四卡、$n=1$、$B=32$、`ppo_mini_batch_size=32` 时，每个 DP rank 持有 8 条 rollout。micro=1 表示一次 full-batch optimizer 更新包含每 rank 的 8 次 micro 前后向。full on-policy 设置 `ppo_epochs=1`；MDPO 与 BPO 设置 `ppo_epochs=16`。这样对应每轮一次与十六次完整更新，outer 数相同，但总 optimizer 更新数相差十六倍。

`mirror_opd_onpolicy` 在代码中没有阻止多 epoch 或多 mini-batch。配置若令一个 outer 做多次 optimizer 更新，后续动作仍来自冻结旧 anchor，代码却不乘 importance ratio，此时不能称为 full on-policy。该项属于配置必须强制的 invariant。基础 launcher 默认 `ppo_epochs=2`，必须覆盖。

MDPO、BPO 都是在每个 outer 开始构造一次 anchor 与 teacher compact cache，所有 inner 共用该 cache；每次 inner 重算当前 student 的 log-prob、ratio、student center。outer 结束后同步权重，再重新生成 rollout。旧书的 token-IS 只校正 action 分布，不把固定 prefix 访问测度 $d_k$ 校正为当前 student 的整条轨迹分布。

### Tail-aware 中心与 KL 解释

anchor 的 Top-K 集合 $S$ 在整个 outer 内固定。对任何分布 $x$，cache 使用单个聚合 tail bucket，定义

$$
\widehat{\mathrm{KL}}(q\Vert x)=\sum_{a\in S}q(a)\log\frac{q(a)}{x(a)}+q(S^c)\log\frac{q(S^c)}{x(S^c)}.
$$

这与精确 KL 的差恰为

$$
\mathrm{KL}(q\Vert x)-\widehat{\mathrm{KL}}(q\Vert x)=q(S^c)\mathrm{KL}\bigl(q(\cdot\mid S^c)\Vert x(\cdot\mid S^c)\bigr)\ge0.
$$

因此 K=16 的日志 `mirror/teacher_center` 和 `mirror/student_center` 是聚合 tail 的 KL 下界；除完整支持或尾部条件分布相同外，它们不等于完整词表 KL。MDPO coefficient 不使用这两个中心，所以 K 对 MDPO 主要影响诊断；BPO coefficient 使用两个中心，所以 K 改变实际近似更新。

记两个中心的欠估量为 $b_T$、$b_S$，BPO 的近似梯度相对精确梯度偏差是

$$
\widehat g_{\mathrm{BPO}}-g_{\mathrm{BPO}}=(b_T-b_S/\eta)\,\mathbb E_q[\nabla\log p].
$$

小 $\eta$ 会放大学生中心误差。K=16 可作为首轮流程参数，但在判断 BPO 质量时必须记录 tail mass 与 support overlap，并在独立固定 prefix 上使用完整 KL 或支持规模对照。

full on-policy 的 prefix 条件 baseline 即使使用聚合 tail 近似，在 $p=q$ 下也只是 action-independent 控制变量，不改变总体 score 梯度。真实运行中 vLLM 与 HF 的数值偏差仍需检查：第一 inner 的 sampled log-ratio 应接近零、student center 应接近零、ESS fraction 应接近 1、support overlap 应接近 1。

## 支持条件与配置覆盖

当前部署要求完整本地词表 logits，sequence parallel size=1，关闭 fused kernels。普通 FSDP 的 data parallel 四卡可分片参数；actor tensor parallel/vocabulary shard 不能直接接入当前 primitive。rollout 的 vLLM tensor parallel 属于推理部署，与 loss primitive 的词表分片约束是两件事。

训练 sampling 必须为有限正 temperature、top_p=1、top_k=-1 或 0、repetition_penalty=1、do_sample=true。不允许训练 rollout quantization、skip rollout、多轮 agent、REMAX sampling overrides、额外 rollout correction。验证可以单独使用 greedy、temperature=0。

teacher 与 student 必须共享 tokenizer 的完整 token 到 ID 语义、特殊 token 和所用 chat template，输出 vocab size 也必须相同。部署 worker 在加载 teacher 时调用共享 tokenization 核验；这是数据和模型接入前必须通过的检查，不能只比较文件名。

`use_task_rewards=false`、`use_policy_gradient=false`、`loss_max_clamp=null`、`log_prob_min_clamp=null`、`actor.use_kl_loss=false`、`algorithm.use_kl_in_reward=false`、`entropy_coeff=0` 保持既定镜像 distillation 更新。数学 reward 只用于监控与测试；追加 PPO 或 verifier reward 会改变算法。

基础 launcher 默认 LoRA rank=8、Qwen3 小模型、单卡、极短 response、关闭 W&B。正式 math 配置必须覆盖 student/teacher 路径、LoRA rank=0、四卡、全部 train/test 路径、响应长度、logger 与 `WANDB_MODE=online`。推荐采用显式 `token-mean` aggregation；当前部署在计算 Mirror loss 前刷新 global batch metadata，已经修复历史上首 micro 使用局部分母的缺陷。

## 显存与主存风险

DeepSeek-R1-Distill-Qwen-1.5B 的配置词表大小为 151936。compact cache 减少传输与持久缓存，生产前向仍物化完整 logits；cache producer 还有 FP32 log-softmax、布尔 tail mask 与 masked tensor。student loss 还有 response-valid 选择副本、FP32 log-softmax 及其梯度。K 从 16 降到更小不能消除这些完整词表 buffer。

单个 $L\times V$ FP32 tensor 的大小为 $4LV$ 字节。V=151936 时，2048、4096、8192 个 token 分别约为 1.16、2.32、4.64 GiB。BF16 logits 本身再占一半，cache masking/softmax 临时 buffer 和模型 activation 另计。多卡 FSDP 分片参数不会按四分之一缩小每 rank 的词表 activation。

因此 actor training micro=1、anchor log-prob micro=1、teacher/ref log-prob micro=1 应分别显式配置。关闭 dynamic batch，或者设置动态 token budget 不超过单条最大序列长度；只设置 micro=1 而保留动态 batching 不能保证一次前向只含一条长序列。开启 remove-padding 与 gradient checkpointing；首轮完整最长序列 canary 的峰值显存才是四卡预算证据。

生产 loss 的 entropy/tail diagnostic 已按 64 个 prefix 分块临时空间，但 `log_p` 主 tensor 与 cache producer 完整 mask 没有分块。长响应仍是主要风险。降低 response length 可能让数学思考全部截断，必须同时记录训练与测试的 reached-budget 比例，不能把短预算的失败误判为算法失败。

默认 actor 参数 dtype=BF16 与 fp32 参数存储、BF16 前向的训练含义不同：很小学习率下 BF16 更新量可被舍入吞掉。历史运维记录已经揭示这一问题。正式低学习率训练优先 FP32 参数存储，optimizer CPU offload、teacher param offload；显存与主存预算都要实测。四个 DP rank 的 CPU offload 会增加主存使用，且 rollout 权重同步也有 CPU 全权重/allocator 峰值，不能只监控 GPU。

部署 FSDP1 teacher 支持 `MIRROR_INFERENCE_EXPLICIT_OFFLOAD=1`，避免静态 CPUOffload inference hooks 把整模型滞留 GPU；应核实 Ray runtime env 传入该变量。FSDP2 的 offload policy 是另一条实现，不能混用 FSDP1 的判断。

## 监控与实验判断

1. 首个 inner 检查 student center、sampled log-ratio、ESS、overlap。每个 outer 都应重新回到 anchor 一致状态。
2. MDPO J=16 关注 ratio 最大值、ESS fraction 与 coefficient RMS；实现故意没有 ratio clipping，非有限 coefficient 会显式报错。ratio 大并不自动说明 bug，但低 ESS 表明少量 action 主导更新，应调整学习率或 eta，再复测。
3. 全部 inner 日志在 optimizer 更新前测量。最后一个 inner 指标描述第十六次更新前的 student，不能当作更新后或最终 checkpoint 指标。
4. grad_norm 是 clipping 前的全局 norm，需要与 grad_clip_scale 同时报出。当前 loss 已直接回传 distillation surrogate，额外 PPO loss 没有参与训练。
5. teacher center 随每个 outer 的 prefix 测度变化。不能仅凭其时间趋势断言固定评价分布上的完整 KL 改善。数学测试完整题集的 step 0/定期/step 50 accuracy、响应长度与预算截断比例提供独立质量证据。
6. baseline 50outer 与 MDPO/BPO 50outer 共享 prompt 数和评测频率；MDPO/BPO 多了十六倍 inner optimizer 更新。正式报告需列出 outer、inner 和耗时，不能称其为同等计算预算比较。

## 实际数值验证

在 `envs/mirroropd/bin/python` 的 Torch 2.7.1+cu126 下，CPU 完整枚举四 action：anchor=(0.4,0.3,0.2,0.1)，teacher=(0.05,0.15,0.3,0.5)，student logits=(-0.3,0.2,0.7,-0.1)，eta=0.3，K=4。对每个 action 的生产 surrogate 以 anchor 概率求期望，autograd 梯度与 `src/mirroropd/token_inner.py` 的原始目标比较。

BPO 最大绝对梯度差为 1.7881393432617188e-7；MDPO 为 5.960464477539063e-8。两者通过 atol=5e-7、rtol=5e-6 比较。这个验证证明完整支持下的条件总体梯度语义，不证明 GPU worker 部署、实际四卡归一化、数值稳定或数学效果；这些需要完整训练 canary 与后续 run 日志验证。
