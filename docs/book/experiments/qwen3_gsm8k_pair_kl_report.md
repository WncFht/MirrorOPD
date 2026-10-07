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

真实 rollout token 就是 $Y$。因此截断的 prefix bias 是 $widehat K_{\mathrm{trunc}}-K$，其方差为零；修正估计量的条件期望等于 $K$，条件方差为

$$
\operatorname{Var}(\widehat K_{\mathrm{corr}}\mid p,q)
=\sum_{v\notin S_K}p(v)\ell(v)^2
-\left(\sum_{v\notin S_K}p(v)\ell(v)\right)^2,
\quad \ell(v)=\log\frac{p(v)}{q(v)}.
$$

汇总表中 `rmse_trunc` 是 prefix 平均的确定性平方误差，`rmse_corrected_theory` 是 prefix 条件方差的平均开方；同时保留真实 rollout token 得到的 `rmse_corrected_observed`，用来显示单次 sampled correction 的实际波动。

## 工程设定

生成使用 archbox 上的 vLLM 0.29.0。该机没有 `nvcc`，因此关闭 vLLM 的 FlashInfer sampler，仍使用 vLLM 的批量调度和 KV cache；这只改变采样内核，不改变 $T=1.0$、$p$ 或 $q$ 的数学定义。完整 rollout 原始文件保存在 archbox 的 `~/src/md-opd-work/runs/qwen3_gsm8k_student_t1/rollouts.jsonl`，共 350,166 个生成 token。

打分脚本使用 Transformers 的 `logits_to_keep`，只保留生成 token 对应的 logits。由于单张 RTX 4070 SUPER 只有 12 GiB，同时加载两个 bf16 模型会挤压激活和词表 logits，当前完整统计采用 bitsandbytes NF4 4-bit 权重、bf16 计算；`metadata.json` 明确记录这一点。两条 pilot 轨迹已经与完整词表公式逐项核对。正式结果在全量任务结束后写入 `kl_summary_by_k.csv`，图由 `plot_pair_kl.py` 生成。

## 全量结果

全量 prefix 数为 350,166，平均 reference reverse KL 为 0.2693 nats。Student 的 Top-1、Top-4、Top-8、Top-32 平均质量分别为 0.9421、0.9975、0.9995、0.99996。对应截断 RMSE 从 $K=0$ 的 1.0397 降到 $K=32$ 的 0.00217；一个 sampled-token 修正的理论 RMSE 从 0.8095 降到 0.02249。因而在真实 Qwen 分布上，$K\geq 4$ 后截断的确定性误差已经小于单 token 修正的条件标准差，体现了无偏高方差与有偏低方差之间的交换关系。

真实 rollout token 的修正误差还受生成阶段 bf16 vLLM 与打分阶段 NF4 量化 Student 的轻微分布不一致影响；例如 $K=32$ 的观测平均误差为 $-2.76\times10^{-4}$，远小于其 0.0519 的观测 RMSE。理论 bias 仍按打分分布严格为零，观测 bias 不应被解读为修正估计量的条件 bias。完整数值见 [kl_summary_by_k.csv](qwen3_gsm8k_pair_kl/kl_summary_by_k.csv)，图见 [pair_kl_topk_metrics.png](qwen3_gsm8k_pair_kl/pair_kl_topk_metrics.png)。

## 解释边界

这里的 `bias` 和 `variance` 是对固定 prefix、固定 Student/Teacher 分布的 scalar KL 估计量分析。它们不是 policy-gradient estimator 的方差，也没有把 KL 数值反向传播；这与当前只研究 KL 本身估计质量的实验目标一致。Student rollout 的答案正确率只作为轨迹数据的附带信息，不把 GSM8K reward 混入 KL 统计。
