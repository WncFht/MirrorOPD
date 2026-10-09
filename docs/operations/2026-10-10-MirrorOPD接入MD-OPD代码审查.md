# MirrorOPD 接入 MD-OPD 代码审查

## 结论与审查范围

MD-OPD 的 Ray 调度、FSDP 引擎、模型加载、数据路由、优化器和 vLLM 权重传输可以复用。MirrorOPD 接入需要独立配置、冻结缓存、教师评分、actor loss、张量布局、同步恢复和验证七部分。现有 `mdpo_kl_loss`、`mopd_resid`、`mopd_grpo` 分别有自己的计算语义，直接更名会改变 MirrorOPD 的数学目标。

本次审查的 MirrorOPD 基线为 `f8f7f7c25d99158534f05f840736e4945d9e3d6e`，MD-OPD 基线为 `f043a754c3d0992b4cab7dff93c80d362bc1d65a`，分支为 `mopd-m1`。审查覆盖生产缓存、损失、FSDP 适配、启动配置、实际 TA-OPD checkout 的训练接线，以及 MD-OPD 的 trainer 主循环、OPD mixin、教师 worker、FSDP logits 路径、actor loss、anchor、replay、EOS 和 checkpoint。下文是接入设计，本次没有实现 MD-OPD 的 Mirror 模式。

## 当前 MirrorOPD 的生产链路

### outer 与 inner 的计算顺序

一次 outer 从同步完成的 student 采样 response。此时的条件 token 分布记为 $q$，它也是该轮冻结的 anchor。actor 在 response 对应的 prediction positions 上构建紧凑缓存，然后 teacher 对相同 prefix 和 anchor 指定的支持集评分。全部缓存完成后才能更新 actor。

inner 始终使用同一批 prefix、采样动作、anchor 支持集及教师量。每次 forward 计算当前 student 分布 $p$，重算 student center，然后反传 sampled action 的 log probability。多个 inner 完成后，训练器把最新 actor 权重同步给 rollout，再开始下一 outer。`ppo_epochs` 是调度参数；每个 epoch 若包含多个 mini-batch，就包含多个 optimizer update。严格的 full-batch $J$ 次 inner 要求每个 epoch 只有一个完整 mini-batch。

现有生产链路的职责如下。

| 部分                 | 第一方维护位置                                          | 职责                                                                |
| -------------------- | ------------------------------------------------------- | ------------------------------------------------------------------- |
| anchor/teacher cache | `src/mirroropd/training/cache.py`                       | 冻结支持集、head/tail 概率、动作概率和 teacher center               |
| score surrogate      | `src/mirroropd/training/losses.py`                      | Mean、BPO、MDPO 和 OPD 对照的逐 token 损失                          |
| FSDP tensor adapter  | `src/mirroropd/integrations/verl/fsdp.py`               | prediction-position cache 与 dense/jagged logits 对接               |
| 指标聚合             | `src/mirroropd/training/metrics.py`                     | 有效 token 的充分统计量、DP 聚合、inner 记录                        |
| tokenizer 契约       | `src/mirroropd/training/tokenization.py`                | token-ID、special tokens 和模板一致性                               |
| 实验配置与留痕       | `training/run_configured_training.sh`                   | 配置解析、模型/数据审计、版本、outer/inner/GPU 记录                 |
| 分布式训练           | TA-OPD/verl 及 `patches/ta-opd-mirror-token-loss.patch` | teacher/ref 选择、缓存路由、mask、actor update 和 rollout hard sync |

`token_inner.py`、`opd_pg.py` 和 `opd_lm.py` 是数学参考与研究实现。接入 MD-OPD 应调用生产内核，保持损失只有一份维护位置。

### 缓存与概率分区

`build_mirror_anchor_cache()` 从温度缩放后的完整 anchor logits 选 Top-K 支持集 $S$，保存 `mirror_anchor_ids`、`mirror_anchor_head_log_probs`、`mirror_anchor_tail_log_prob` 和 `mirror_anchor_action_log_prob`。同时保存 `mirror_actions`、`mirror_valid` 和用于 teacher 校验的词表大小。完整 logits 保留在 worker 本地，通信量为 $O(NK)$，其中 $N$ 是有效 prefix 数。

`complete_mirror_teacher_cache()` 接收相同 prefix 的 teacher logits，在 $S$ 上 gather 概率，并计算补集的总概率和采样动作概率。它保存 teacher head/tail、动作 log probability 和 teacher center。teacher 原生 Top-K 无法替代 $S$：两者选出的 token 不同，求和对应的分区也不同。

令 $q_R=\sum_{a\notin S}q(a)$，其他分布的 tail 同理。当前采用的 student center 为

$$
C_s=\sum_{a\in S}q(a)\log\frac{q(a)}{p(a)}+q_R\log\frac{q_R}{p_R}.
$$

teacher center 为

$$
C_t=\sum_{a\in S}q(a)\log\frac{q(a)}{t(a)}+q_R\log\frac{q_R}{t_R}.
$$

这是 Top-K 加尾部合并的分区 KL。当 $K<V$ 时，它与完整词表 KL 存在差异；支持集、tail 与估计误差必须在报告中写清。teacher center 在 outer 内冻结，student center 随 inner 的当前 $p$ 重算。anchor 与 teacher 缓存全部停止梯度。

### 损失与停止梯度

令采样动作 $a\sim q$，定义

$$
u=\log p(a)-\log q(a),\qquad r=\log t(a)-\log q(a).
$$

每个 token 的损失统一写成

$$
\ell(a)=\operatorname{stopgrad}(c(a))\log p(a).
$$

生产内核中的三个系数分别是

$$
\begin{aligned}
c_{\mathrm{Mean}}&=u-\eta(r+C_t),\\
c_{\mathrm{BPO}}&=(u+C_s)/\eta-r-C_t,\\
c_{\mathrm{MDPO}}&=e^u(u/\eta-r).
\end{aligned}
$$

Mean/BPO 在冻结 $q$ 的 rollout 测度下更新。MDPO 通过 $e^u=p(a)/q(a)$ 引入重要性比率。实际梯度为 $\operatorname{stopgrad}(c)\nabla\log p(a)$，中心、奖励和 ratio 均不独立反传。生产实现没有 PPO ratio clipping，非有限 coefficient 直接报错；参数梯度的 global-norm clipping 则由优化器链路单独处理。

内核另有 `opd_onpolicy`、`opd_is` 和 `opd_is_no_baseline` 对照。on-policy 对照要求使用更新前当前策略的样本；多次复用一个 outer batch 后，样本来自冻结 $q$，不能继续把无重要性修正的更新称为 on-policy。

### 张量与监控

cache 必须放在预测 action 的前驱 logit 位置。driver 的 padded response 字段与 actor 的 packed full-sequence 字段之间，需要显式 pad/unpad；Top-K 字段还必须保留最后的 $K$ 维。prompt、padding 和无效终端位置在 gather 与概率运算之前被排除。adapter 当前限定 sequence parallel 为 1，并要求本地完整词表，不能直接用于 TP 分片 logits。

inner 指标先按有效 token 累积 count、sum、sum-square、min/max，再跨 micro-batch 和 DP rank 聚合。平均多个 micro-batch 的均值会让不同长度的 batch 获得错误权重。每个 optimizer update 单独记录 center、ratio、coefficient、entropy、head/tail mass、支持集重叠和梯度范数；outer 指标与 inner 指标有独立横轴。

## MD-OPD 的接入位置与现有差异

MD-OPD 的实际 actor 链路为 `RayPPOTrainer → ActorRolloutRefWorker → FSDPEngine → loss_function`。完整 logits 的接入点在 FSDP engine，当前目录结构没有旧版 `dp_actor.py` 接入路线。

| 接口                  | MD-OPD 源码与审查行号                                                               | 可复用能力与所需变化                                                                 |
| --------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| driver rollout/update | `verl/verl/trainer/ppo/ray_trainer.py:1752`、`:1998`、`:2048`、`:2108`              | 保留 rollout、old-logprob、actor update 与同步调度，在 actor 更新前完成 Mirror cache |
| mini-batch 调度       | `verl/verl/trainer/ppo/ray_trainer.py:1631`                                         | 复用 epochs、mini-batch、shuffle，明确 optimizer update 与 inner 索引                |
| OPD 数据准备          | `verl/verl/trainer/ppo/opd_trainer_mixin.py:458`、`:613`                            | 新增独立 Mirror 分支，避免沿用混合 generator/anchor 定义                             |
| worker loss 选择      | `verl/verl/workers/engine_workers.py:634`                                           | 显式选择 Mirror loss，区分 FSDP teacher 与 distillation teacher loop                 |
| full-logits callback  | `verl/verl/workers/engine/fsdp/transformer_impl.py:1412`、`:1452`、`:1669`          | 温度缩放后调用第一方 adapter，输出逐位置 loss 与 diagnostics                         |
| teacher/anchor        | `verl/verl/workers/fsdp_teacher_worker.py:362`、`:416`、`:529`                      | 复用模型与路由，扩展 anchor cache 和指定 IDs teacher scoring                         |
| anchor registry       | `verl/verl/utils/opd_anchor.py:51`                                                  | 复用 FSDP1/FSDP2 snapshot 基础设施，按 outer 边界同步                                |
| PPO 与现有 MDPO       | `verl/verl/workers/utils/losses.py:221`、`verl/verl/trainer/ppo/core_algos.py:2187` | Mirror 分支直接聚合 surrogate，不叠加既有 PPO/MDPO KL                                |
| M2 residual/GRPO      | `verl/verl/trainer/ppo/opd_core_algos.py:324`、`:413`                               | 保留现有算法，Mirror 使用独立 modes                                                  |
| replay                | `verl/verl/trainer/ppo/mopd_replay_buffer.py:55`                                    | 第一版关闭跨 outer replay，保证冻结 outer 测度                                       |
| checkpoint            | `verl/verl/trainer/ppo/ray_trainer.py:1283`、`:1352`                                | 第一版限定 outer 边界恢复，并重同步 anchor 与 rollout                                |

现有 MDPO 在 PPO objective 上加 $\mathrm{KL}(p\Vert q)/\eta$ 数值估计，可选 k3 或带 straight-through backward 的 k3+。Mirror MDPO 使用前述 ratio-weighted score surrogate，两者的 clipping、baseline 和梯度构造不同。

`mopd_resid` 对平方残差直接自动微分，并计算 frozen anchor 对 current student log probability 的完整词表期望。`mopd_grpo` 使用 complement ratio、cap 和 clip mask。它们与 Mirror Mean/BPO 的停止梯度系数具有不同计算图，不能拿现有 advantage 字段替换后直接宣称等价。

一个需要单独核对的采样测度问题位于 `opd_trainer_mixin.py:613–634`：`anchor_log_probs` 来自实际 generator 的记录概率，teacher-generated rows 使用 teacher probability；state terms 来自独立 frozen anchor。anchor 多 outer 才刷新、永久冻结或混入 teacher rows 时，这些量对应不同分布。Mirror 的 action/head/tail/center 必须全部来自同一个 outer $q$。

现有 M3 replay 跨 generation 采样旧 trajectory，带 priority 与 coverage，但没有相应重要性修正。现有 M4 将多个 EOS token 合并为一个 action class。第一版应关闭这两项；若以后支持，需要分别推导跨 outer 采样权重，以及在同一个 action space 上重做 anchor、teacher、student 的 head/tail 和采样概率。

## 七部分实施设计

1. **配置与核心依赖。** 增加独立 `mirror_pmd_mean`、`mirror_bpo`、`mirror_mdpo` modes，显式配置 eta、K、inner epochs、teacher 与来源版本。MD-OPD 直接加载 `~/src/mirroropd/src` 的生产包。先支持 FSDP、完整词表、SP=1、共享 tokenizer，关闭 fused logits loss、PPO clipping 分支、额外 KL/entropy、rollout correction、跨 outer replay 和 EOS class 合并。
2. **outer 缓存生命周期。** fresh student rollout 后冻结本次 $q$，构建 anchor cache，再完成 teacher cache。同一 outer 的所有 inner 都使用这份不可变缓存；支持集也不能在 inner 中重新选。记录 outer ID 与 rollout/anchor 权重版本。
3. **TeacherWorker 指定支持集评分。** 在现有教师路由下增加 compact-cache 输出。先运行 anchor，释放完整 logits 后再运行 teacher；teacher gather anchor IDs，并保留 tail mass。多教师 collective 顺序沿用现有路由契约，同一个 prefix 的 teacher 行不能错位。
4. **actor full-logits loss。** 在 FSDP engine 温度缩放后的完整 logits 回调中调用 Mirror adapter，独立聚合逐位置 score loss。更新期间不用 M2 的额外 anchor forward，不经 PPO advantage/ratio clipping，也不叠加现有 `mdpo_kl_loss`。
5. **prediction-position 与全局归一化。** 增加 cache 字段的 DataProto/TensorDict 传输、pad/unpad 和复原，明确只 shift 一次。沿用 `dp_size`、`batch_num_tokens`、`global_batch_size`、`loss_scale_factor`，使改变 micro-batch 划分不改变总梯度。
6. **inner、同步与恢复。** 先让 `ppo_epochs` 承担 inner 调度；完整 inner 结束后同步 vLLM，检查下一 outer 版本。第一版只在 outer 边界保存/恢复；加载 actor checkpoint 后重建 anchor 并同步 rollout。支持 inner 中途恢复时，还需持久化 outer batch/cache、inner 索引、优化器/调度器和 RNG 状态。
7. **诊断与验证。** 保留逐 inner token moments 与独立 outer/inner 横轴，加入 cache checksum、版本检查和 provenance。正式效果测试使用完整数据与统一 token/optimizer budget，分别报告 CPU parity、GPU canary 和 benchmark。

第一版要验证两个问题：MD-OPD engine 是否给出与独立 Mirror 内核一致的 loss/gradient；真实多个 inner 与下一次 rollout 是否保持冻结/同步顺序。若 micro-batch 划分改变总梯度、inner 改变 cache、teacher 支持集错位或 rollout 版本未更新，就说明接线不符合该算法，应先修正接口。

验收顺序如下。

1. 固定小 logits，比较三种方法的 loss 和每个参数梯度，覆盖 K=0、部分支持集和全词表。
2. 覆盖 dense/jagged、不同 response 长度、无效 IDs、temperature、teacher 指定 IDs、空 tail 和 DP/global-token 聚合。
3. 用小 causal LM 连续执行多个 outer/inner，逐步检查 cache 冻结、student center 重算与下一轮 anchor 更新。
4. 在 archbox 做单卡 canary，验证真实 Ray worker、vLLM、checkpoint boundary 和 optimizer update 次数。
5. 固定完整 GSM8K 配置与预算后比较 OPD、Mean、BPO 和 MDPO，并保存模型、数据、采样、种子、代码版本及复现命令。

## 本次验证范围

本机实际通过 `tests/test_upstream_opd_semantics.py`、`tests/test_upstream_mirror_wiring.py` 和 `tests/test_upstream_mirror_lm.py`。覆盖损失/梯度、compact cache、tail 偏差恒等式、padding、teacher/ref 路由、PPO 绕过、rollout 版本检查，以及三种方法各 3 outer × 4 inner 的小模型验证。隔离发布工作区通过 `make check`。

这些验证由 CPU tensor、AST 接线检查和小模型 oracle 组成。本次没有运行新的 GPU 训练，也没有验证 MD-OPD 的 Mirror 集成；本报告里的 MD-OPD 修改点仍是实施设计。

发布审计还复现了 TA-OPD 补丁基线不一致：原补丁可以在 `migrated_head=5bed008...` 重放，在 manifest 的原始 `base_commit=62669af...` 无法重放。此次整理将现有 checkout 的完整差异重新导出为相对原始基线的补丁，并在干净基线上检查、重放及逐文件比较。这个修订补全已有接线的恢复材料，训练内核保持原样。
