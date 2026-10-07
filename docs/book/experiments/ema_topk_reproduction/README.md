# EMA 论文 Top-K 标量 KL 合成实验复现

本目录只分析标量 reverse KL 的两个估计量，不分析 KL 梯度，也不加入 K3++：确定性头部截断 `Trunc-Top-K`，以及带一个 sampled token 尾部修正的 `Top-K corrected`。实验在 archbox 的 NVIDIA GeForce RTX 4070 SUPER 上运行，结果保存在现有的 `k_analysis/` 目录中。

## 输入数据与生成方式

实验为每个情景固定生成一个 token-level categorical distribution pair：

$$
\pi_{\rm ref}=\operatorname{softmax}(z_{\rm ref}),
\qquad
\pi=\operatorname{softmax}(s z),
$$

其中 $z,z_{\rm ref}\sim\mathcal N(0,I_{32000})$ 独立抽取。对每个目标质量 $M_{32}\in\{0.2,0.5,0.8,0.9\}$，用二分法选择 $s$，使

$$
\sum_{j\in\operatorname{TopK}(\pi,32)}\pi(j)\approx M_{32}.
$$

不同 panel 只改变策略的集中度，使用同一个固定的分布对计算真值、偏差和方差。

这里的 $B$ 是对同一个 prefix 的单-token 标量估计量做平均时的有效 Monte Carlo 样本数。实际 OPD 中每个 prefix 只需要一个 sampled token；minibatch 中不同 prefix 和 token position 的平均才产生 $B$ 所代表的方差降低。$B$ 不是算法必须对同一 prefix 重复 rollout 的次数。

## 两个标量 KL 估计量

固定一个 prefix，记

$$
\ell(y)=\log\frac{\pi(y)}{\pi_{\rm ref}(y)},
\qquad
K=\operatorname{KL}(\pi\Vert\pi_{\rm ref})
 =\mathbb E_{Y\sim\pi}[\ell(Y)].
$$

令 $S_K=\operatorname{TopK}(\pi,K)$。

### Trunc-Top-K

确定性截断只计算头部：

$$
\widehat K_{\rm trunc}(K)
 =\sum_{j\in S_K}\pi(j)\ell(j).
$$

因此

$$
\operatorname{Bias}[\widehat K_{\rm trunc}(K)]
 =-\sum_{j\notin S_K}\pi(j)\ell(j),
\qquad
\operatorname{Var}[\widehat K_{\rm trunc}(K)]=0.
$$

$K$ 增大通常会缩小被忽略的 tail；但在 $B\to\infty$ 时，Trunc-Top-K 的 bias 仍然存在。

### Top-K corrected

先精确计算头部，再使用一个 sampled token $Y\sim\pi$ 修正尾部：

$$
\widehat K_{\rm corr}(K)
 =\sum_{j\in S_K}\pi(j)\ell(j)
  +\mathbf 1\{Y\notin S_K\}\ell(Y).
$$

它满足

$$
\mathbb E[\widehat K_{\rm corr}(K)]=K,
$$

因此标量 KL 的 bias 为零。它的方差只来自尾部：

$$
\operatorname{Var}[\widehat K_{\rm corr}(K)]
 =\sum_{j\notin S_K}\pi(j)\ell(j)^2
 -\left(\sum_{j\notin S_K}\pi(j)\ell(j)\right)^2.
$$

当 outer rollout 来自 $\pi_k$、当前策略为 $\pi_\theta$ 时，尾部 sampled-token 项使用停止梯度的重要性权重

$$
\mathbf 1\{Y\notin S_K\}
\operatorname{sg}\!\left(\frac{\pi_\theta(Y)}{\pi_k(Y)}\right)
\log\frac{\pi_\theta(Y)}{\pi_{\rm ref}(Y)},
\qquad Y\sim\pi_k.
$$

前向标量期望仍然是当前策略的 tail KL。本文的合成实验先使用 on-policy 情况 $\pi_k=\pi$，隔离估计量本身的 bias--variance 性质。

## Bias、variance 与 RMSE

所有图都报告相对量。对任意标量估计量，

$$
\operatorname{MSE}=\operatorname{Bias}^2+\operatorname{Var},
\qquad
\operatorname{RMSE}=\sqrt{\operatorname{MSE}}.
$$

对 $B$ 个独立 sampled-token 估计量取平均时，Top-K corrected 的方差变为 $\operatorname{Var}/B$，而 Trunc-Top-K 的 bias 和零方差保持不变。因此

$$
\operatorname{RMSE}_{\rm trunc}
 =|\operatorname{Bias}_{\rm trunc}|,
\qquad
\operatorname{RMSE}_{\rm corr}(B)
 =\sqrt{\operatorname{Var}_{\rm corr}/B}.
$$

对应输出如下：

- [标量 KL bias](k_analysis/topk_value_bias.png)：Trunc-Top-K 的 tail bias 与 Top-K corrected 的零 bias。
- [标量 KL variance](k_analysis/topk_value_variance.png)：Trunc-Top-K 的零 Monte Carlo 方差与 Top-K corrected 的 tail variance。
- [标量 KL RMSE](k_analysis/topk_value_rmse_vs_k.png)：$B\in\{1,32,512,32768\}$ 时两种估计量的误差交叉关系。
- [逐点数据](k_analysis/value_metrics.csv)：包含 KL 真值、head/tail 值、两种估计量的 bias、variance 和标准差。

图中的结论是：Trunc-Top-K 在小 $K$ 时可能依靠零方差取得较小的单次 RMSE，但其误差在大 $B$ 时停留在 tail bias floor；Top-K corrected 在标量上无偏，增大 $K$ 通常会降低 tail variance，但由于方差还包含尾部均值平方项，个别很小的 $K$ 区间可以出现非单调变化。最终 RMSE 由 tail variance 与 minibatch 的有效样本数 $B$ 共同决定。
