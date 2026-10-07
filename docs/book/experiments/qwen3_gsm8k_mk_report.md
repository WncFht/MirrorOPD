# Qwen3 在 GSM8K 上的完整轨迹与 Top-$K$ 质量

## 实验问题

本实验回答两个问题。第一，Qwen3-1.7B 与 Qwen3-4B 在 GSM8K 的真实生成轨迹上，top-$K$ 集合承载了多少下一 token 概率质量。第二，当 $K=32$ 时，尾部质量 $1-M(32)$ 是否普遍足够小，从而解释 Trunc 与 sampled-token Corrected 估计量的实际差异。

这里的 $M(K)$ 是每个 prefix 的模型分布 $pi(\cdot\mid s)$ 的 top-$K$ 概率质量：

$$
M_s(K)=\sum_{v\in\operatorname{TopK}(\pi(\cdot\mid s),K)}\pi(v\mid s).
$$

它只描述概率分布的质量集中程度，不包含 teacher/student 的 log-ratio。因此 $M(32)$ 很大可以直接说明截断 KL 的概率尾部很小，但不能单独决定 KL 数值误差或 Corrected 估计量的方差。

## 数据与计算设置

使用 GSM8K test split 的全部 1319 个 prompt。每个 prompt 只生成一条完整轨迹，temperature 为 1.0，最大新 token 数为 512；轨迹在模型自然结束时提前停止。Qwen3-1.7B 的轨迹来自已有全量 rollout，Qwen3-4B 在 archbox 上重新完成了 1319 条 rollout。

对每条保存的轨迹，使用同一个模型再次进行 teacher-forcing forward。对于 prefix $s_t$，输入是 prompt 加上此前已经生成的 token，模型输出完整 vocabulary logits；在完整 softmax 上取 $K\in\{1,2,4,8,16,32\}$ 的 top-$K$，并记录每一个 prefix 的 $M_s(K)$。模型加载采用 NF4 4-bit 权重和 bf16 计算，以适配 archbox 的 12GB GPU；softmax 仍在完整词表上计算，top-$K$ 不是 vLLM 的截断 logprob 近似。

1.7B 共得到 350166 个 prefix，4B 共得到 335570 个 prefix。每个模型均覆盖全部 1319 个 prompt。

## 总体结果

![mean top-K mass](qwen3_gsm8k_mk/mean_mk_vs_k.png)

表中 prefix 级统计先把所有生成位置放在一起；prompt 级统计先在每个 prompt 内求平均，再对 1319 个 prompt 求统计量。

| 模型 | 层级 | mean $M(1)$ | mean $M(2)$ | mean $M(4)$ | mean $M(8)$ | mean $M(16)$ | mean $M(32)$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | prefix | 0.942075 | 0.986763 | 0.997471 | 0.999493 | 0.999878 | 0.999961 |
| Qwen3-1.7B | prompt mean | 0.945931 | 0.988731 | 0.998114 | 0.999662 | 0.999923 | 0.999976 |
| Qwen3-4B | prefix | 0.945300 | 0.988205 | 0.997941 | 0.999612 | 0.999899 | 0.999952 |
| Qwen3-4B | prompt mean | 0.949406 | 0.990109 | 0.998515 | 0.999752 | 0.999938 | 0.999970 |

两个模型都表现出非常尖锐的 next-token 分布。对 prefix 直接平均时，$K=32$ 的尾部质量均值分别为 $3.90\times10^{-5}$ 和 $4.79\times10^{-5}$。换句话说，从概率质量角度看，大多数 prefix 的 top-32 已经接近完整词表。

## $M(32)$ 的分布，而不是只有均值

![prompt mean M32](qwen3_gsm8k_mk/prompt_mean_m32_histogram.png)

prompt 级的 $M(32)$ 几乎全部集中在 1 附近，但仍有少量困难 prompt 的平均尾部质量明显更大。具体地说，prompt 级尾部质量 $1-\overline{M(32)}$ 的统计为：

| 模型 | mean | median | p90 | p99 | 尾部 $<10^{-3}$ | 尾部 $<10^{-4}$ | 尾部 $<10^{-5}$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | $2.39\times10^{-5}$ | $1.68\times10^{-6}$ | $4.40\times10^{-5}$ | $3.77\times10^{-4}$ | 99.92% | 94.47% | 77.48% |
| Qwen3-4B | $2.99\times10^{-5}$ | $4.27\times10^{-7}$ | $1.78\times10^{-5}$ | $8.93\times10^{-4}$ | 99.17% | 95.22% | 87.11% |

prefix 级分布更重尾，因为一个 prompt 只要出现少数几个高熵位置，就会贡献少量大尾部 prefix。![tail ECDF](qwen3_gsm8k_mk/tail32_ecdf.png)

均值和中位数之间的巨大差距正是这个现象：Qwen3-4B 的 prefix 级 median tail32 只有 $7.83\times10^{-12}$，但 mean tail32 是 $4.79\times10^{-5}$。因此“$M(32)$ 很大”对典型位置成立，对每一个位置的统一上界不成立。

## 为什么 $M(32)$ 大时 Trunc 经常比 Corrected 好

固定 prefix，令 $p=\pi_k(\cdot\mid s)$、$q=\pi_\theta(\cdot\mid s)$，并记

$$
\ell(v)=\log\frac{p(v)}{q(v)},\qquad H=\sum_{v\in S_{32}}p(v)\ell(v),\qquad T=\sum_{v\notin S_{32}}p(v)\ell(v),
$$

其中 $S_{32}$ 是 p 的 top-32 集合。真实单步 KL 为 $D=H+T$。

Trunc 估计量为 $\widehat D_{\mathrm{trunc}}=H$，所以它的条件偏差和方差是

$$
\operatorname{Bias}(\widehat D_{\mathrm{trunc}})=-T,
\qquad
\operatorname{Var}(\widehat D_{\mathrm{trunc}})=0,
\qquad
\operatorname{MSE}(\widehat D_{\mathrm{trunc}})=T^2.
$$

sampled-token Corrected 估计量先从 $p$ 采样 $Y$，若 $Y\in S_{32}$ 则回退到 $H$，若 $Y$ 在尾部则补上 $\ell(Y)$：

$$
\widehat D_{\mathrm{corr}}=H+\mathbf 1\{Y\notin S_{32}\}\ell(Y).
$$

它在条件于 prefix 时无偏，但方差为

$$
\operatorname{Var}(\widehat D_{\mathrm{corr}})
 =\sum_{v\notin S_{32}}p(v)\ell(v)^2-T^2.
$$

令尾部概率为 $\delta=1-M(32)$。当尾部 log-ratio 有界 $|\ell(v)|\le L$ 时，

$$
|T|\le \delta L,
\qquad
\operatorname{Var}(\widehat D_{\mathrm{corr}})\le \delta L^2.
$$

这给出关键的尺度差异：Trunc 的 RMSE 上界按 $O(\delta)$ 缩小，Corrected 的标准差按 $O(\sqrt\delta)$ 缩小。于是当 $\delta$ 很小时，Corrected 虽然无偏，单样本 RMSE 仍然可能大于 Trunc；它的优势是消除系统性尾部偏差，而不是保证每个 prefix 的单次 RMSE 更小。

因此，对本实验的 $M(32)$ 结果，正确结论是：$M(32)$ 大时 Trunc 通常更低方差、也常有更小 RMSE，但“只要 $M(32)$ 大就必然更好”不成立。还必须看尾部的 $\ell(v)^2$ 加权二阶矩，以及任务是否要求无偏 KL。尾部质量很小但 teacher 在尾部给出极端 log-ratio 时，Corrected 的方差仍会被少数事件主导。

![tail ECDF](qwen3_gsm8k_mk/tail32_histogram.png)

## 与已有 pair-KL 结果的关系

前一组 Qwen3-1.7B rollout 上的 student--teacher pair-KL 实验显示，$K=32$ 时 Trunc RMSE 为约 $2.17\times10^{-3}$，sampled-token Corrected 的理论 RMSE 为约 $2.25\times10^{-2}$。这并不矛盾：本报告测量的是 student 分布的尾部概率质量，而 KL 估计量的 Corrected 方差还乘上尾部 log-ratio 的平方。当前 pair-KL 的 scoring 模型使用 NF4 量化，而 rollout 由 bf16 vLLM 产生；量化差异还会使经验 Corrected bias 偏离理想条件无偏结论。

## 可复现实验文件

- `1p7b/prefix_mk.csv` 与 `4b/prefix_mk.csv`：逐 prefix 的 $M(K)$ 和直接计算的 `tail32`。
- `1p7b/prompt_mk.csv` 与 `4b/prompt_mk.csv`：逐 prompt 的均值和中位数。
- `mk_distribution_stats.csv`：全部分位数统计。
- `tail32_summary.csv`：尾部质量阈值比例。
- `mean_mk_vs_k.{png,pdf}`、`prompt_mean_m32_histogram.{png,pdf}`、`tail32_ecdf.{png,pdf}`：图像及可打印版本。
- `vllm_gsm8k_rollout.py` 和 `score_single_model_mk.py`：rollout 与完整词表评分脚本。
