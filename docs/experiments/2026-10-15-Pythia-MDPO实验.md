# Pythia MDPO 实验

本报告记录 2026-10-08 至 2026-10-09 的旧版 MDPO 实验。模型、数据和更新语义绑定各 run 的配置与源码指纹；历史 η=+∞ 对照不属于当前 API 2。实验没有得到“教师对齐改善且真实文本 NLL 不退化”的结果。

## 固定问题与判定标准

本轮首先验证两个问题：训练是否按照“rollout、冻结锚点、复用同一批数据进行 inner 更新、hard sync、重新 rollout”的顺序执行；在真实预训练语言模型中，这组有限 inner 更新是否改善独立文本前缀上的教师对齐。

链路通过的条件包括以下几项。

1. 每组恰有 32 条 outer 记录和 128 条 inner 记录，inner 索引完整覆盖每个 outer 的 4 次更新。
2. 每轮第一条 inner 记录中的学生/锚点 log-ratio 为零，重要性权重 ESS 占比为一，学生/锚点 KL 中心为零。
3. 同一轮 inner 使用冻结的锚点、教师信号和 rollout；下一轮 rollout 使用更新后的模型版本。
4. 损失与梯度有限，训练退出码为零，四个保存时点的 checkpoint 均能严格加载。
5. 每组 512 条生成记录都能对应到冻结训练集中的不同 source row。

优化效果用独立固定前缀的双向全词表 KL、学生熵和真实续写 NLL 判定。训练 surrogate 的数值和 rollout 平均 reward 用于诊断；两者不能替代固定测度上的评价。

## 模型、数据与长度

### 模型版本

Pythia 使用相同的 tokenizer 词表映射，适合先排除跨 tokenizer 对齐问题。选用相邻的小规模学生与教师，使全参数训练、精确全词表评估和多个 checkpoint 都能在现有显存内运行。模型来源是 EleutherAI 的原始模型发布[^pythia]。

| 角色 | 模型                    | 固定 revision                              |
| ---- | ----------------------- | ------------------------------------------ |
| 学生 | `EleutherAI/pythia-14m` | `cf967c0a9a04383db6f7b1108d86b2962634b4ac` |
| 教师 | `EleutherAI/pythia-70m` | `a39f36b100fe8a5377810d56c3f4789b9c53ac42` |

两个模型的输出维度均为 50,304，tokenizer 含 50,277 个 token。训练概率及全词表 oracle 均使用模型的完整输出维度。准备数据时核对两个 tokenizer 的 token→ID 映射、EOS ID 和测试文本编码。实际 rollout、学生评分和教师评分统一使用学生 tokenizer。教师参数冻结。

Pythia 是基础语言模型，本轮使用自然文本续写。没有添加 chat role、system prompt 或 assistant 结束标记。模板严格拼接消息正文，准备阶段检查模板处理后的 prompt IDs 与原始 64 个 IDs 一致。

### 数据生成

数据为 `Salesforce/wikitext` 的 `wikitext-2-raw-v1`，固定 revision 为 `b08601e04326c79dfdd32d625aee71d232d685c3`[^wikitext]。

准备程序按官方 train、validation 两个 split 分别处理文本行。过滤空行、标题行和不足 128 个 token 的行。对每个候选行取前 64 个 token 作为 prompt；仅保留这些 token 经 decode、再 encode 后完全一致的行。随后用固定数据种子 20261008 打乱 train 候选，用种子 20261009 打乱 validation 候选，分别取 512 行和 64 行。

每个保存记录包含 source row、split、prompt 文本、64 个 prompt IDs、后续 64 个真实 token IDs 和源文本 SHA-256。训练模型只看到前 64 个 token，随后自己生成。保存的原文后 64 个 token 用于 teacher-forced 评价，不进入本轮 OPD 损失。

官方 train 与 validation split 分开取样，保证验证文本不参与本次参数更新。这不能证明验证文本没有出现在 Pythia 的预训练语料中。

### Rollout 与截断

每个训练 prompt 生成一条响应，温度为 1，top-p 为 1，top-k 为 -1；遇到 EOS 或生成 64 个 token 时停止。总上下文上限为 128。这里的长度上限用于分析短上下文的 token 动力学。

主实验实际生成 32,570 个有效 response token，507/512 条轨迹触及 64-token 上限，占 99.02%。低学习率对照生成 32,292 个 token，497/512 条触及上限，占 97.07%。本轮结果针对有长度上限的续写，不应解释为自然结束的完整长轨迹结果。

上游生成日志默认 decode 会清理英文标点附近的空格，显示文本与原始 prompt 字符串有格式差异。用同一 tokenizer 的默认 decode 对照后，两组均验证 512/512 条 prompt 对应原始 IDs，且每条训练 prompt 使用一次。对应关系单独保存在 `rollout-source-index.json`。原始生成日志保存文本，没有保存每个 sampled action 的原始 ID 和完整缓存张量；因此它适合审查生成内容，不能独立重放逐 token 的全部梯度计算。

## 优化设置与执行

### 冻结设置

正式配置位于 [Pythia MDPO 配置](../../data/refactor-20261015/legacy-source.tar.gz#training/configs/pythia-mdpo.json)。训练启动入口复用现有 [TA-OPD/verl 启动程序](../../data/refactor-20261015/legacy-source.tar.gz#training/run_upstream_mirror_training.sh)，不另写 trainer。

| 参数                         | 主实验                                    | 学习率对照           |
| ---------------------------- | ----------------------------------------- | -------------------- |
| outer / prompt 数            | 32 / 512                                  | 相同                 |
| prompt batch                 | 16                                        | 相同                 |
| 每条 prompt 的 rollout 数    | 1                                         | 相同                 |
| inner 更新数 $J$             | 4                                         | 相同                 |
| mini-batch / GPU micro-batch | 16 / 1                                    | 相同                 |
| 梯度聚合                     | 完整 16 条响应的 token mean               | 相同                 |
| 优化器                       | AdamW，$\beta=(0.9,0.999)$                | 相同                 |
| 学习率 / 调度                | $10^{-5}$ / constant                      | $10^{-6}$ / constant |
| weight decay / gradient clip | 0 / 1                                     | 相同                 |
| $\eta$ / 缓存 top-$K$        | 0.1 / 32                                  | 相同                 |
| 参数更新                     | 全参数，LoRA rank 0                       | 相同                 |
| 同步                         | 每轮 inner 结束后 hard sync               | 相同                 |
| 参数与前向精度               | FP32 参数；FSDP BF16 前向，FP32 reduction | 相同                 |
| rollout / 教师评分精度       | vLLM BF16 / HF BF16                       | 相同                 |
| 训练与数据顺序种子           | 42                                        | 相同                 |
| vLLM 种子                    | 0                                         | 相同                 |
| checkpoint                   | outer 8、16、24、32                       | 相同                 |

mini-batch 等于完整 prompt batch，所以每次 inner 更新都遍历本轮所有响应。micro-batch 为一条响应，框架进行累积后执行一次优化器更新；它不会对每条响应单独执行一次参数更新。所有有效 response token 等权，长度更长的响应贡献更多 token。

固定种子不等于 GPU 运算和异步采样逐 bit 可复现。两组采用相同数据与初始化配置；由于参数更新不同，后续 on-policy 轨迹也随之不同。学习率对照控制配置差异，不是强制复用两组完全相同的训练轨迹。

### 历史目标与更新语义

冻结锚点 $q$ 和教师 $T$，本组使用对数 reward $r=\log(T/q)$，没有当前默认 κ=0.5 的幂均值变换。条件目标为 $F_\eta(p)=\operatorname{KL}(p\Vert q)/\eta-\mathbb E_p r$。采样动作来自 q，系数为 $(p/q)[\log(p/q)/\eta-r]$，整个停止梯度后乘以 $\log p$。

完整推导及 η 的端点解释见[历史 MDPO 数学来源](../../theory/implementation-sources/2026-10-15-历史MDPO目标与有限内层.md)。surrogate 标量不等于原目标；有限四次 AdamW 更新没有证明达到 Gibbs 最优解。top-32 用于中心及支持诊断，不截断 sampled action 的完整词表归一化。

### 每轮执行顺序

1. 将最新学生参数同步到 vLLM，记录 rollout weight version。
2. 从冻结训练集取 16 个尚未使用的 prompt，按当前学生生成每个 prompt 的一条响应。
3. 对这些 response token 的 predecessor prefix，用学生锚点评分并冻结 action log-prob、top-32 支持及 head/tail 概率；教师完成 action 评分与中心诊断缓存。
4. 复用这 16 条响应，执行四次完整 batch 的 score-surrogate 更新。每次更新前记录 inner 指标，梯度累积完后记录 clipping 前范数并执行 AdamW。
5. 将更新后的学生 hard sync 到 rollout 服务，再进入下一个 outer。每八轮保存 checkpoint。

第一个 outer 未收敛也会在四次更新后同步。当前实现采用固定 $J=4$，没有基于内层目标或梯度的收敛停止器。四次更新应解释为本轮计算预算，不能解释为已求出内层最优解。

## 固定前缀评价

每个验证样本由固定的 64-token prompt 和原文 64-token continuation 构成。评价输入始终为这 128 个真实 token，与训练 rollout 无关。分别评价 prompt 的 64 个非空 prefix，以及预测 continuation 的 64 个 prefix；二者共享 prompt 的最后一个 prefix，合计每条文本 127 个位置。64 条文本总计 8,128 个 KL/entropy 位置，续写 NLL 总计 4,096 个目标 token。

学生和教师以 FP32、eval 模式进行 HF 前向。log-softmax 和概率矩计算提升至 FP64。每个位置对完整 50,304 维输出计算双向 KL，得到数值 oracle，不使用 top-$K$、尾部修正或独立动作采样来估计评价 KL。每次只评分一条 128-token 文本，词表计算按八个 prefix 分块，避免保留整个验证集的 logits。

checkpoint 使用单 rank FSDP model state，初始化单 rank Gloo process group 后 materialize ShardedTensor/DTensor，并严格加载所有 model keys。四个 checkpoint 都通过实际加载和评分。CPU 测试额外验证普通 state dict、单 rank ShardedTensor、精确 KL 和 continuation token 的预测位置对齐。

统计先在同一条文本中平均，再对 64 条文本进行配对 bootstrap。每次重采样 64 条文本，重复 5,000 次，报告相对初始模型变化的 95% percentile 区间。由于每条文本具有相同位置数，点估计也等于全 prefix 平均。这个区间刻画固定训练所得模型在验证文本上的差异，不刻画多个训练种子的随机性。不能把同一文本的 127 个相关 prefix 当成 127 个独立样本。

## 学习率对照结果

### 固定验证前缀

| 状态                | $\operatorname{KL}(\text{Student}\Vert\text{Teacher})$ | $\operatorname{KL}(\text{Teacher}\Vert\text{Student})$ | 学生续写 NLL |
| ------------------- | -----------------------------------------------------: | -----------------------------------------------------: | -----------: |
| 初始模型            |                                               0.987310 |                                               0.890709 |     4.980831 |
| $10^{-5}$，outer 8  |                                               0.977756 |                                               1.003436 |     5.281590 |
| $10^{-5}$，outer 16 |                                               1.044912 |                                               1.070674 |     5.420331 |
| $10^{-5}$，outer 24 |                                               1.046621 |                                               1.082083 |     5.414517 |
| $10^{-5}$，outer 32 |                                               1.085772 |                                               1.108130 |     5.408573 |
| $10^{-6}$，outer 8  |                                               0.950803 |                                               0.891189 |     5.033202 |
| $10^{-6}$，outer 16 |                                               0.928000 |                                               0.907274 |     5.097592 |
| $10^{-6}$，outer 24 |                                               0.913243 |                                               0.924768 |     5.144357 |
| $10^{-6}$，outer 32 |                                               0.912054 |                                               0.952285 |     5.202112 |

KL 单位为 nats，NLL 单位为 nats/token。教师真实续写 NLL 为 4.193058，显著低于初始学生；教师在这个评价集合上确实具有更好的语言建模能力。

![固定前缀 KL 与真实续写 NLL](../../results/training/pythia-mdpo/figures/validation-dynamics.png)

最终主实验相对初始模型的学生到教师 KL 增量为 0.09846，配对区间为 [0.07909, 0.11719]；NLL 增量为 0.42774，区间为 [0.39477, 0.46107]。退化不能用这 64 条文本的平均抽样误差解释。

低学习率对照的学生到教师 KL 变化为 -0.07526，配对区间为 [-0.08345, -0.06725]；NLL 变化为 +0.22128，区间为 [0.19595, 0.24704]。教师到学生 KL 增量为 +0.06158，区间为 [0.05371, 0.06960]。这三个方向一致说明：reverse KL 对齐改善与覆盖和真实文本似然退化同时发生。

学生熵从 5.03523 降到主实验的 4.77761、低学习率对照的 4.45569 nats。低学习率对照的熵接近教师的 4.43612，但熵接近仍不能推出 token 概率正确。

### Hard sync 与 inner 动力学

所有 32 轮的第一条 inner 记录均满足 log-ratio RMS 为零、ESS 占比为一、锚点到学生中心为零。这验证了每轮起点与冻结锚点一致；各轮完成后 rollout version 递增。

![主实验的 inner 动力学](../../results/training/pythia-mdpo/figures/inner-dynamics.png)

下面按各轮有效 token 数加权，汇总每次更新前的指标。$n=0$ 表示第一步更新前，$n=3$ 表示第四步更新前；当前日志没有额外记录第四步执行后的完整 batch 前向。

| 学习率    | $n$ | sampled log-ratio RMS | 锚点到学生 Tail-aware KL | ESS/token 数 |
| --------- | --: | --------------------: | -----------------------: | -----------: |
| $10^{-5}$ |   0 |                     0 |                        0 |            1 |
| $10^{-5}$ |   1 |               0.20545 |                  0.01169 |      0.95295 |
| $10^{-5}$ |   2 |               0.22555 |                  0.01317 |      0.95050 |
| $10^{-5}$ |   3 |               0.26108 |                  0.01572 |      0.94406 |
| $10^{-6}$ |   0 |                     0 |                        0 |            1 |
| $10^{-6}$ |   1 |               0.33604 |                  0.02890 |      0.81785 |
| $10^{-6}$ |   2 |               0.34621 |                  0.02918 |      0.83686 |
| $10^{-6}$ |   3 |               0.34859 |                  0.03013 |      0.84736 |

ESS 对本轮有效 token 上的重要性比率计算：

$$
\widehat{\operatorname{ESS}}=
\frac{(\sum_i\rho_i)^2}{\sum_i\rho_i^2},\qquad
\rho_i=\frac{\pi_\theta(a_i\mid s_i)}{\pi_k(a_i\mid s_i)}.
$$

它衡量这些经验重要性权重的集中程度。轨迹内 token 相关，所以这个数不等于独立 prefix 的统计有效样本数。它也不是 reward 加权 ESS；reward 的幅度和符号会另外影响梯度方差。

低学习率对照没有表现出更高的 ESS。其实际 on-policy 轨迹和后续梯度不同，表中还有少数大系数事件的影响，不能把学习率减十倍直接推成观测 log-ratio 减十倍。本轮没有把这些异常裁掉。最大正 score 系数为主实验的 331.75、对照的 2934.04。clipping 前梯度范数分别落在 [889.27, 8596.76] 和 [1912.41, 520285.75]，每次均超过阈值一。

全局 gradient clipping 限制梯度范数，却不直接限制参数变化对应的 KL；AdamW 还使用累积二阶矩进行各参数缩放。因而当前日志支持继续研究数值精度与参数更新的局部敏感性，不能仅凭 clipping 和有限 loss 判断已进入稳定的 Natural Gradient 收敛区间。

主实验 score 系数 RMS 在四个时点分别约为 3.913、4.970、4.520、4.128，并未趋近零。这个量含有限动作样本噪声，不是总体梯度范数；结合有限 $J$ 和高 clipping 前范数，本轮没有证明 inner 收敛。hard sync 本身通过了链路验证，outer 的教师对齐改善仍需要独立评价。

### KL 改善与 NLL 退化的反例

在一个固定 prefix 上，设真实 token 分布为 $Q=(0.8,0.2)$，初始学生 $P=(0.9,0.1)$，教师 $T=(0.7,0.3)$，更新后的学生 $P'=(0.6,0.4)$。全部分布支持一致。直接计算可得

$$
\begin{aligned}
\operatorname{KL}(P\Vert T)
&=0.9\log\frac{0.9}{0.7}+0.1\log\frac{0.1}{0.3}
=0.116322,\\
\operatorname{KL}(P'\Vert T)
&=0.6\log\frac{0.6}{0.7}+0.4\log\frac{0.4}{0.3}
=0.022582.
\end{aligned}
$$

与此同时，真实分布下的交叉熵满足

$$
\begin{aligned}
H(Q,T)&=-0.8\log0.7-0.2\log0.3=0.526135,\\
H(Q,P)&=-0.8\log0.9-0.2\log0.1=0.544805,\\
H(Q,P')&=-0.8\log0.6-0.2\log0.4=0.591919.
\end{aligned}
$$

教师优于初始学生，学生到教师的 KL 也下降，但新学生的真实 NLL 更高。这个例子证明了“教师 NLL 更好且 reverse KL 下降”不足以推出“学生 NLL 下降”。实际训练还受到共享参数与不同 prefix 测度的影响，因此评价必须同时保留双向 KL 和真实 token NLL。

## η 扫描

此前固定配置为 $\eta=0.1$。在 MDPO 损失中，锚点 KL 的系数为 $1/\eta=10$。本轮扫描 $\eta\in\{0.03,0.1,0.3,1,+\infty\}$，保持学习率 $10^{-6}$、Pythia-14M 学生、Pythia-70M 教师、冻结 WikiText-2 数据与短上下文设置一致。数据、模型 revision、混合精度和评价位置的完整定义沿用 [Pythia 训练报告](2026-10-15-Pythia-MDPO实验.md)。

每组 512 条不同训练 prompt，prompt batch 与 mini-batch 均为 16，micro-batch 为一条响应，每轮累积完整 batch 后执行四次 inner 更新。共 32 outer、128 次 AdamW 更新，$K=32$，每条 prompt 生成一条响应，温度一、无 top-p/top-k 截断，prompt/response 上限均为 64。每轮 hard sync。所有组使用相同初始化和配置种子；策略更新改变后，后续 on-policy 轨迹自然不同。

其中 $\eta=0.1$ 使用此前已完成的低学习率实验，另外四组新运行。损失核心只增加了正无穷端点支持，有限 $\eta$ 的算式没有改变。这是单种子有限 inner 预算的扫描，不是精确内层优化的比较。

所有五组均完成 32 outer 和 128 inner 更新，正常退出。四组新运行各保存四个 checkpoint，并在相同验证前缀上完成五个时点的全词表评价。下面为 outer 32 的结果；KL 为 nats，NLL 为 nats/token。

| $\eta$    | 原始目标的 $1/\eta$ | 学生到教师 KL | 教师到学生 KL | 真实续写 NLL |   学生熵 |
| --------- | ------------------: | ------------: | ------------: | -----------: | -------: |
| 初始模型  |                   — |      0.987310 |      0.890709 |     4.980831 | 5.035230 |
| 0.03      |             33.3333 |      0.926909 |      0.944758 |     5.179526 | 4.528496 |
| 0.1       |                  10 |      0.912054 |      0.952285 |     5.202112 | 4.455692 |
| 0.3       |              3.3333 |      0.909943 |      0.946174 |     5.168605 | 4.431645 |
| 1         |                   1 |      0.909935 |      0.938502 |     5.150048 | 4.474173 |
| $+\infty$ |                   0 |      0.914251 |      0.916784 |     5.088139 | 4.541630 |

![五组 eta 的固定前缀 KL 与真实续写 NLL](../../results/training/pythia-eta-scan/figures/validation-dynamics.png)

学生到教师 KL 的点估计以 $\eta=1$ 最低，与 0.3 的差异只有 $7.5\times10^{-6}$ nats，这个量级不足以支持两者优劣排序。$\eta=\infty$ 在本轮的 NLL 和教师到学生 KL 两项中最低；它的学生到教师 KL 略高于 0.3 和 1。五组均改善了学生到教师 KL，同时真实续写 NLL 与教师到学生 KL 均比初始化更高。

本轮没有得到“更强锚点正则带来更好训练效果”的证据。$\eta=0.03$ 的学生到教师 KL 改善少于其他组，NLL 也没有优于 0.3、1 和无正则组。单种子、相同更新次数的结果用于选择后续检查方向，不能据此断言无正则方法在充分训练或大模型上更好。

### 相对 eta=0.1 的配对差异

保持同一组验证文本，对最终模型逐文本计算差异，再进行配对 bootstrap。表中负值表示相对 $\eta=0.1$ 指标降低。区间只反映这批文本的抽样变化，不包括重新训练的随机性，也没有对多个比较作校正。

| $\eta$    | 学生到教师 KL 差异及 95% 区间 | NLL 差异及 95% 区间           |
| --------- | ----------------------------- | ----------------------------- |
| 0.03      | +0.01485 [0.01217, 0.01778]   | -0.02259 [-0.02838, -0.01664] |
| 0.3       | -0.00211 [-0.00424, -0.00004] | -0.03351 [-0.03960, -0.02764] |
| 1         | -0.00212 [-0.00387, -0.00040] | -0.05206 [-0.05936, -0.04482] |
| $+\infty$ | +0.00220 [-0.00231, 0.00768]  | -0.11397 [-0.12873, -0.09992] |

无正则组相对 0.1 的学生到教师 KL 区间包含零，其 NLL 明确降低。但无正则组 NLL 比初始化高 0.10731，说明它只是减轻退化，并未改善真实文本似然。

### Inner 动力学与正则项幅度

![eta 对 inner 偏离与 ESS 的影响](../../results/training/pythia-eta-scan/figures/eta-inner-comparison.png)

五组所有 outer 的第一个 inner 状态均满足 sampled log-ratio RMS 为零、ESS/token 数为一、锚点到学生中心为零。下面按有效 token 数加权，汇总第四次更新前的指标。中心为 anchor top-32 加聚合 tail 的 KL 数值，不是完整词表 KL。

| $\eta$    | log-ratio RMS | 锚点到学生 Tail-aware KL | ESS/token 数 | score 系数 RMS |
| --------- | ------------: | -----------------------: | -----------: | -------------: |
| 0.03      |       0.34304 |                  0.02904 |      0.84574 |       29.19213 |
| 0.1       |       0.34859 |                  0.03013 |      0.84736 |       10.10845 |
| 0.3       |       0.34836 |                  0.03086 |      0.85656 |        5.25652 |
| 1         |       0.35720 |                  0.03294 |      0.86090 |        4.21879 |
| $+\infty$ |       0.35848 |                  0.03284 |      0.83844 |        4.18627 |

更小 $\eta$ 的 proximal 项 $u_\theta/\eta$ 会放大非零 log-ratio，在本轮反映为更大的 score 系数 RMS。0.03 组第四次更新前约为 29.19，明显高于 1 和无正则组的约 4.2；这说明强正则也能产生大的恢复梯度，不能把小 $\eta$ 直接解释为小的随机梯度幅度。

模型偏离锚点的 RMS 在五组之间接近，没有按 $\eta$ 成比例缩小。这里使用 AdamW、全局 gradient clipping 和共享 transformer 参数；学习率、归一化优化器状态、动作样本及混合精度共同影响实际参数更新。观察到的 ESS 占比在 0.84–0.86 附近，也不具有随 $\eta$ 单调改善的关系。这些是经验重要性权重的集中程度，不能当作相关轨迹中独立样本数。

所有组第四次更新前的 score 系数 RMS 仍明显非零。它包含 Monte Carlo 动作噪声，不能单独作为内层停止条件。本轮固定四次更新，没有证明 inner 收敛。因此本轮可以比较同等预算下的行为，但不能验证 Gibbs 精确最优解或无正则精确求解的集中化极限。

### 扫描的解释限制

后续对照可保留 $\eta=1$ 与无正则组作为目标语义不同的两个对照，$\eta=0.3$ 作为有限 proximal 设置。该扫描不支持优先选择 0.03。对于纯 reverse-KL OPD，应该用 $\eta=1$ 解释总体目标；对于删除 proximal 项的冻结 reward 对照，用 $\eta=\infty$ 标注。

在增加 inner 次数前，需要记录最后一次更新后的固定 batch 目标与梯度，并核对 FP32 前向下的参数变化。现有 BF16 前向、AdamW clipping 和有限采样的结果尚未解释 NLL 退化的全部原因；本轮没有通过修改精度、样本数或估计量来同时引入新的变量。

四组新实验实际生成约 3.24–3.25 万 response token/组，97.27%–98.83% 的响应触及 64-token 上限。验证 oracle 精确覆盖指定短 prefix 测度，结论仍局限于有长度上限的文本续写。每组累计 outer 时间约 125–127 秒，GPU 按秒记录的峰值约 4,265–4,267 MiB。此 η 扫描不包含 SmolLM2。

## 后续预算校准与长程失稳

校准矩阵已在 archbox 顺序完成 20 个作业，全部退出码为 0。每个作业有 `total_training_steps=1`，因此外层指标各只有一个点；这不是 inner 未执行。`J=1` 的 inner 文件有 1 行，`J=4` 有 4 行，`J=16` 有 16 行。资源组使用 B=16/32/64、micro-batch=1/2/4（B=64 的 micro=4 未运行），inner 组使用 J=1/4/16 与 lr=1e-7/3e-7/1e-6，eta 组使用 0.1/0.3/1.0。

矩阵首先排除了明显不稳定的组合。`J=16, lr=3e-7` 的最后一个 inner ESS fraction 为 0.062、最大 importance ratio 为 203.98；`B=16, micro=4` 为 0.088 和 121.96；这些配置不能作为默认。`J=4, lr=1e-7` 在 B=32、micro=2 下最后一个 inner ESS fraction 为 0.871，最大 ratio 为 9.93，作为长程候选。资源矩阵说明 B=32、micro=2 在 12 GiB RTX 4070 SUPER 上留有余量；B=64 没有带来足够稳定的权重集中度改善，先不作为默认。

随后运行了两个 16-outer、64-inner 的确认作业，均无 checkpoint、退出码为 0，并实时写入 W&B：`eta=0.3`[^source-1] 与 `eta=1`[^source-2]。`eta=0.3` 的前两个 outer 出现瞬态尖峰（ESS fraction 0.180、0.079；最大 ratio 103.8、170.9），从第 3 个 outer 起 ESS 大多在 0.85--0.96，后续最大 ratio 多数小于 12。`eta=1` 的平均梯度范数更低，但在第 3、5、7 个 outer 仍反复出现 ESS 下探，最低 0.163，最大 ratio 108.98。基于这两个长程样本，这轮校准选择的候选为 `B=32, micro=2, J=4, lr=1e-7, eta=0.3, K=16`；该选择是稳定性与 inner 计算量的工程折中，尚不是收敛定理。

这两次确认也改变了实验解释：单 outer 矩阵只能筛掉明显失稳点，不能证明 outer 动力学稳定。正式消融必须保留逐 inner ESS、importance ratio、梯度范数和 reward coefficient RMS，并把前两个 outer 的尖峰单独报告。这些单种子确认没有补齐独立种子及固定前缀 full-vocab 评价，不能据此判定算法优劣。

随后按要求把 inner 预算改为 `J=16`，并运行了 32 outer、512 inner 的长程确认。`lr=1e-7` 的 16-outer 试验已经出现 ESS fraction 最低 0.400、最大 ratio 58.25；将学习率降到 `3e-8` 后，前 20 个 outer 的 ESS 多数在 0.86--0.96，但第 21--25 个 outer 仍出现一次长程失稳（最低 ESS 0.106、最大 ratio 143.23），随后恢复。该结果否定了“只要降低学习率并固定 J=16 就已经稳定”的判断；J=16 的最终 setting 仍需更长 horizon、seed 重复和 full-vocab 评价。

J=16、`lr=3e-8` 的 run 为 `uuk4hr4d`[^source-3]。该 run 还记录了裁剪诊断：512 个 inner update 的 `grad_was_clipped` 全为 1，`grad_clip_scale` 范围为 `4.31e-4--9.95e-3`，均值约 `6.18e-3`。因此实时日志中的 `grad_norm` 是裁剪前范数，不能当作实际更新范数；训练确实长期由 global-norm clip=1 主导。后续实验必须同时报告 pre-clip norm、clip scale 和参数更新 norm，并增加 clip threshold 的控制实验。

## 来源、资源与证据

主实验累计 outer step 时间为 127.75 秒，对照为 128.45 秒；这个和包括训练轮内的 rollout、评分、更新及 checkpoint 开销，不包括框架启动与训练结束后的独立评价。GPU 按秒采样得到的峰值分别为 4265、4267 MiB。两组各保存四个完整训练 checkpoint，约 873 MiB/组，留在 archbox 的忽略运行目录，不提交权重。

运行时 Python 3.12.13、PyTorch 2.8.0、Transformers 4.57.6、vLLM 0.11.0、Ray 2.59.0、datasets 4.8.5。官方 TA-OPD 基线固定为 `62669afb5101bfdf57dc5ddf81e78f000c5ce2fc`，主仓库训练入口提交为 `15223d5`。

archbox 的实际框架源码位于 `~/src/mirroropd-taopd/verl`，第一方核心位于 `~/src/mirroropd/src`。远端不是 Git checkout，日志以文件 SHA-256 为实际部署证据。两组各核对 24 个第一方核心及框架修改的 Python 文件，均与本机源码一致。主实验 provenance 在训练结束后采集，对照在启动前采集；该时点差异写入 `source-verification.json`。没有用空的远端 `source-commit.txt` 冒充实际版本。

GPT-NeoX 接入修复了两个架构假设：普通多头注意力没有 `num_key_value_heads` 时，取 `num_attention_heads`；vLLM 的 MoE 专用 patch 对无需该 patch 的 dense 模型直接返回。先前失败停在模型初始化阶段，修复后 2 outer × 2 inner 的 smoke 和两组完整实验均通过。集成补丁在干净官方基线上成功 apply，并通过 Python 编译验证。

| 内容                           | 本机位置                                                                                                                         |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| 固定输入、source IDs、校验信息 | [data/pythia-wikitext](../../data/pythia-wikitext)                                                                               |
| 主实验生成文本                 | [rollouts](../../data/pythia-wikitext/rollouts)                                                                                  |
| 低学习率对照生成文本           | [rollouts-low-lr](../../data/pythia-wikitext/rollouts-low-lr)                                                                    |
| 主实验配置、全部指标与评分     | [results/training/pythia-mdpo](../../results/training/pythia-mdpo)                                                               |
| 对照配置、全部指标与评分       | [results/training/pythia-mdpo-low-lr](../../results/training/pythia-mdpo-low-lr)                                                 |
| 训练算法核心                   | [losses.py](../../data/refactor-20261015/legacy-source.tar.gz#src/mirroropd/training/losses.py)                                  |
| 框架缓存与位置适配             | [fsdp.py](../../data/refactor-20261015/legacy-source.tar.gz#src/mirroropd/integrations/verl/fsdp.py)                             |
| 精确前缀评价程序               | [checkpoint diagnostics](../../data/refactor-20261015/legacy-source.tar.gz#experiments/scoring/pythia_checkpoint_diagnostics.py) |
| 聚合、配对区间                 | [report program](../../data/refactor-20261015/legacy-source.tar.gz#experiments/analysis/pythia_mdpo_report.py)                   |
| 可复现绘图                     | [plot program](../../data/refactor-20261015/legacy-source.tar.gz#experiments/plotting/plot_pythia_mdpo.py)                       |

每组结果目录保存原始 `console.log`、`inner.jsonl`、`outer.jsonl`、GPU CSV、Hydra 完整解析配置、参数覆盖列表、数据 manifest、源码 provenance、退出码和各 checkpoint 的逐 prefix 评价。解析配置作为运行证据保持原始格式。图经过 figure-viz 审计和目检，PNG、PDF、审计文件与聚合 CSV 共同保存。

## 历史复现条件

原始输入、source row、rollout 文本和各 run 配置分别位于 [Pythia 数据](../../data/pythia-wikitext)、[训练结果](../../results/training/pythia-mdpo)、[低学习率结果](../../results/training/pythia-mdpo-low-lr)与[η 汇总](../../results/training/pythia-eta-scan)。归档只保存了文本，缺少全部 sampled action ID 与缓存，不能据此独立重放每个梯度。

训练、评分、η 扫描和校准矩阵的旧入口位于[固定源码归档](../../data/refactor-20261015/legacy-source.tar.gz)内的 `training/`，分析/绘图位于其 `experiments/`。先按[维护指南的历史恢复说明](../maintenance.md)恢复到独立目录，并使用该 run 对应的模型、数据和旧环境，再执行归档入口。当前 `mirroropd run` 不承诺复现这些旧配置。η=∞ 在旧 Hydra 中以 `1e309` 解析；当前配置显式拒绝非有限 η。

本轮文档整理只核对仓库中的记录及引用，没有重跑训练、checkpoint 评价或 bootstrap。多种子重复、更新后的固定 batch 目标与梯度、精度对 NLL 退化的因果分离仍缺证据。

### 参考文献

[^pythia]: EleutherAI. Pythia model releases. Hugging Face. [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m), [Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m).

[^wikitext]: Salesforce. WikiText dataset release. Hugging Face. [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext).

[^source-1]: `eta=0.3`. [来源](https://wandb.ai/wncfht/mirror-opd-pythia/runs/abdxqqj2).

[^source-2]: `eta=1`. [来源](https://wandb.ai/wncfht/mirror-opd-pythia/runs/bpa3haul).

[^source-3]: `uuk4hr4d`. [来源](https://wandb.ai/wncfht/mirror-opd-pythia/runs/uuk4hr4d).
