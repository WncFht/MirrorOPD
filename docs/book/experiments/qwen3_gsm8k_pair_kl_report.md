# Qwen3 GSM8K 真实 rollout 与 Top-K 标量 KL

本实验固定温度 $T=1.0$，在 GSM8K test 的全部 1319 个 prompt 上各采样一条 Student 轨迹。Student 使用 Qwen3-1.7B，Teacher 使用 Qwen3-4B；生成阶段使用 vLLM，打分阶段对同一条轨迹的每个 prefix 计算 Student 与 Teacher 的完整词表分布。

## 统计对象

对 Student 分布 $p$ 和 Teacher 分布 $q$，真值是

$$
K(p,q)=\operatorname{KL}(p\Vert q)=\sum_v p(v)\log\frac{p(v)}{q(v)}.
$$

令 $S_K$ 为 Student 概率最大的 $K$ 个 token。每个 prefix 的截断估计量、一个 sampled-token 修正估计量分别为

$$
\widehat K_{\mathrm{trunc}}=\sum_{v\in S_K}p(v)\log\frac{p(v)}{q(v)},
$$

$$
\widehat K_{\mathrm{corr}}=\widehat K_{\mathrm{trunc}}+\mathbf 1\{Y\notin S_K\}\log\frac{p(Y)}{q(Y)},
\qquad Y\sim p.
$$

真实 rollout token 就是 $Y$。因此截断的 prefix bias 是 $\widehat K_{\mathrm{trunc}}-K$，其方差为零；修正估计量的条件期望等于 $K$，条件方差为

$$
\operatorname{Var}(\widehat K_{\mathrm{corr}}\mid p,q)
=\sum_{v\notin S_K}p(v)\ell(v)^2
-\left(\sum_{v\notin S_K}p(v)\ell(v)\right)^2,
\quad \ell(v)=\log\frac{p(v)}{q(v)}.
$$

汇总表中 `rmse_trunc` 是 prefix 平均的确定性平方误差，`rmse_corrected_theory` 是 prefix 条件方差的平均开方；同时保留真实 rollout token 得到的 `rmse_corrected_observed`，用来显示单次 sampled correction 的实际波动。

## 工程设定

生成使用 archbox 上的 vLLM 0.29.0。该机没有 `nvcc`，因此关闭 vLLM 的 FlashInfer sampler，仍使用 vLLM 的批量调度和 KV cache；这只改变采样内核，不改变 $T=1.0$、$p$ 或 $q$ 的数学定义。完整 rollout 原始文件保存在 archbox 的 `~/src/md-opd-work/runs/qwen3_gsm8k_student_t1/rollouts.jsonl`，共 350,166 个生成 token。

打分脚本使用 Transformers 的 `logits_to_keep`，只保留生成 token 对应的 logits。由于单张 RTX 4070 SUPER 只有 12 GiB，同时加载两个 bf16 模型会挤压激活和词表 logits，当前完整统计采用 bitsandbytes NF4 4-bit 权重、bf16 计算；因此这里的“真值”是量化后 Student/Teacher 分布的完整词表 KL，`metadata.json` 明确记录这一点。两条 pilot 轨迹已经与完整词表公式逐项核对。正式结果在全量任务结束后写入 `kl_summary_by_k.csv`，图由 `plot_pair_kl.py` 生成。

## 全量结果

全量 prefix 数为 350,166，平均 reference reverse KL 为 0.2693 nats。Student 的 Top-1、Top-4、Top-8、Top-32 平均质量分别为 0.9421、0.9975、0.9995、0.99996。对应截断 RMSE 从 $K=0$ 的 1.0397 降到 $K=32$ 的 0.00217；一个 sampled-token 修正的理论 RMSE 从 0.8095 降到 0.02249。因而在真实 Qwen 分布上，$K\geq 4$ 后截断的确定性误差已经小于单 token 修正的条件标准差，体现了无偏高方差与有偏低方差之间的交换关系。

真实 rollout token 的修正误差还受生成阶段 bf16 vLLM 与打分阶段 NF4 量化 Student 的轻微分布不一致影响；例如 $K=32$ 的观测平均误差为 $-2.76\times10^{-4}$，远小于其 0.0519 的观测 RMSE。理论 bias 仍按打分分布严格为零，观测 bias 不应被解读为修正估计量的条件 bias。完整数值见 [kl_summary_by_k.csv](qwen3_gsm8k_pair_kl/kl_summary_by_k.csv)，图见 [pair_kl_topk_metrics.png](qwen3_gsm8k_pair_kl/pair_kl_topk_metrics.png)。

## 同一支持集上的归一化 Top-$K$ KL

截图中的 Top-$K$ OPD 使用的不是未归一化截断项，而是先把两个分布限制在 Student 的 Top-$K$ 支持集 $S_K$，再分别归一化。记

$$
P_K=\sum_{v\in S_K}p(v),\qquad Q_K=\sum_{v\in S_K}q(v),
$$

则

$$
\bar p_K(v)=\frac{p(v)\mathbf 1\{v\in S_K\}}{P_K},\qquad
\bar q_K(v)=\frac{q(v)\mathbf 1\{v\in S_K\}}{Q_K}.
$$

归一化 Top-$K$ KL 的精确计算式为

$$
K_K^{\mathrm{norm}}
=\operatorname{KL}(\bar p_K\Vert\bar q_K)
=\frac{1}{P_K}\sum_{v\in S_K}p(v)\log\frac{p(v)}{q(v)}
 +\log\frac{Q_K}{P_K}.
$$

因此它不是完整词表 KL 的另一个无偏估计量，而是一个改变了目标分布的确定性近似。对固定 prefix，$K_K^{\mathrm{norm}}$ 没有 sampled-token 随机性，所以本实验中 `variance_normalized=0`；这里的 variance 指估计量条件于该 prefix 的采样方差，不是不同 prefix 之间归一化 KL 数值的方差。后者若用于 reward 归一化，应另行报告。

全量结果如下。bias 和 RMSE 都相对于完整词表 $K(p,q)$ 计算，$K=0$ 的归一化分布未定义，因此留空。

| $K$ | mean normalized KL | bias | variance | RMSE |
|---:|---:|---:|---:|---:|
| 1 | $5.1\times10^{-11}$ | $-0.26932$ | $0$ | $1.03968$ |
| 2 | $0.18540$ | $-0.08392$ | $0$ | $0.54660$ |
| 4 | $0.24304$ | $-0.02629$ | $0$ | $0.28804$ |
| 8 | $0.26058$ | $-0.00874$ | $0$ | $0.15007$ |
| 16 | $0.26622$ | $-0.00310$ | $0$ | $0.08367$ |
| 32 | $0.26805$ | $-0.00127$ | $0$ | $0.05438$ |

在这个实验中，归一化 Top-$K$ 的 bias 与 RMSE 随 $K$ 单调下降，但在 $K=32$ 仍大于未归一化截断的 RMSE $0.00217$。原因是归一化会把头部质量重新放大到总质量一，$log(Q_K/P_K)$ 只校正头部总质量差异，无法恢复尾部的 token-wise KL；它消除了“质量不足”的尺度误差，却没有消除“尾部结构”误差。$K=1$ 时两个归一化分布都是 point mass，KL 恒为零，这正是该定义相对于完整 KL 的极端偏差。

原图 [pair_kl_topk_metrics.png](qwen3_gsm8k_pair_kl/pair_kl_topk_metrics.png) 已加入归一化 Top-$K$ 的绝对 bias、RMSE 和条件 variance 曲线；逐 $K$ 的全部列（`mean_normalized_kl`、`bias_normalized`、`variance_normalized`、`rmse_normalized`）见 [kl_summary_by_k.csv](qwen3_gsm8k_pair_kl/kl_summary_by_k.csv)。

## Top-$K$ KL reward 的有效样本数

若把每个 prefix 视为一个 sample，并把一个非负的 top-$K$ KL 标量 $R_i$ 当作样本权重，可以使用标准的权重有效样本数

$$
\operatorname{ESS}_R
=\frac{\left(\sum_{i=1}^{N}R_i\right)^2}{\sum_{i=1}^{N}R_i^2}.
$$

当 prefix 独立同分布时，归一化后的大样本形式为

$$
\frac{\operatorname{ESS}_R}{N}
\longrightarrow
\frac{\mathbb E[R]^2}{\mathbb E[R^2]}
=\frac{1}{1+\operatorname{CV}(R)^2}.
$$

这衡量的是 reward 加权平均实际使用了多少个等权样本。它不是普通 on-policy rollout 的 proposal ESS：如果只做未加权的 on-policy 平均，proposal 本身就是 $\pi_k$，形式上的 proposal ESS 是 $N$；ESS 的下降来自 reward 重尾，而不是采样分布错配。

在当前 1.7B rollout、4B teacher 的完整 $K_k^T$ 数据上，若把 $R_i=K_k^T(s_i)$ 作为非负标量权重，则

$$
\mathbb E[R]=0.269321,
\qquad
\mathbb E[R^2]=1.08094,
$$

给出

$$
\frac{\operatorname{ESS}_R}{N}=0.06710,
\qquad
\operatorname{ESS}_R\approx 23497
$$

（$N=350166$ 个 prefix）。这说明重尾已经把标量 reward 加权的有效样本量压到约 6.7%。这只是 $K_k^T$ 的示例；对每个 top-$K$ Trunc reward，应使用对应的逐 prefix $R_i^{(K)}$ 重新计算。

Trunc 和 Corrected 的差异可以写成

$$
R_i^{\mathrm{trunc}}=H_K(s_i),
\qquad
R_i^{\mathrm{corr}}=H_K(s_i)+\mathbf 1\{Y_i\notin S_K\}\ell_i(Y_i).
$$

Trunc 删除了尾部随机项，因此通常降低 $\mathbb E[R^2]$，提高 reward-ESS，但引入尾部 bias；Corrected 恢复无偏性，却把尾部二阶矩重新带回去，可能显著降低 ESS。因此应同时报告

$$
\mathbb E[(R^{(K)})^2],
\qquad
\operatorname{ESS}_{R^{(K)}},
\qquad
\text{Trunc bias}.
$$

### Vanilla OPD 中的 signed reward

Vanilla OPD 常用的是 token-level signed reward

$$
r(s,a)=\log\frac{\pi_T(a\mid s)}{\pi_k(a\mid s)},
$$

而不是非负的 $K_k^T(s)$。它满足

$$
\mathbb E_{a\sim\pi_k}[r(s,a)]=-K_k^T(s),
$$

并且可以正可以负。因此普通的

$$
\frac{(\sum_i r_i)^2}{\sum_i r_i^2}
$$

会因为正负抵消而接近零，不能作为稳定的 ESS 定义。对 signed reward，至少要同时报告二阶矩和信噪比

$$
\mathrm{SNR}_r
=\frac{|\mathbb E[r]|}{\sqrt{\operatorname{Var}(r)/N}},
$$

或者使用绝对权重版本

$$
\operatorname{ESS}_{|r|}
=\frac{(\sum_i|r_i|)^2}{\sum_i r_i^2}.
$$

真正用于 policy gradient 的样本是

$$
G_i=r_i\,\nabla_\theta\log\pi_\theta(a_i\mid s_i).
$$

因此最直接的梯度有效样本指标应基于 $G_i$ 的范数，例如

$$
\operatorname{ESS}_{G}
=\frac{\left(\sum_i\lVert G_i\rVert\right)^2}{\sum_i\lVert G_i\rVert^2},
$$

同时配合梯度均值与二阶矩。vanilla OPD 的样本效率低，严格来说不是因为 proposal ESS 小，而是因为 signed log-ratio 的重尾使 $\mathbb E[\lVert G\rVert^2]$ 很大、梯度 SNR 很低。Top-$K$ Trunc、reward clipping、baseline/centering 和多 token tail correction 都是在降低这个二阶矩；其中 Trunc 用 bias 换 variance，Corrected 保持无偏但可能保留较低 ESS。

## 解释边界

这里的 `bias` 和 `variance` 是对固定 prefix、固定 Student/Teacher 分布的 scalar KL 估计量分析。它们不是 policy-gradient estimator 的方差，也没有把 KL 数值反向传播；这与当前只研究 KL 本身估计质量的实验目标一致。Student rollout 的答案正确率只作为轨迹数据的附带信息，不把 GSM8K reward 混入 KL 统计。
