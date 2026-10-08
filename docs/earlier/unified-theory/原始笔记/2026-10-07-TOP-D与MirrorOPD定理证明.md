# TOP-D 与 Mirror OPD 的加权、定理和证明

本文对比 TOP-D[^topd]（原中文源文件为 2607.04751_zh_CN/neurips_2026.tex） 与本项目的 OPD mirror 更新。外部来源的定理按原文假设重述并给出证明；本项目的结论单独标明额外假设。

## 加权结构

TOP-D 有三种不同的权重。外部近端 reward 是

$$
\widetilde r_\alpha(a|s)
=\log\left(1-\alpha+\alpha\frac{T(a|s)}{\pi_{\mathrm{old}}(a|s)}\right).
$$

在同一批 rollout 的内部 epoch 中，$\pi_{\mathrm{old}}$、teacher、$\alpha$、reward、return 和 advantage 都固定。PPO 的 token importance ratio

$$
p_t(\theta)=\frac{\pi_\theta(a_t|s_t)}
{\pi_{\mathrm{old}}(a_t|s_t)}
$$

随 student 参数变化，并在

$$
\operatorname{clip}(p_t,1-\epsilon,1+\epsilon)
$$

中截断。组内 token advantage

$$
\widehat A_t=\frac{\widetilde R_t-\mu_{\mathrm{group}}}
{\sigma_{\mathrm{group}}}
$$

由 rollout 后计算并在内部 epoch 中复用。因而 TOP-D 的 weighted batch 是固定的，PPO ratio 不是固定的。

小 $\alpha$ 时

$$
\widetilde r_\alpha
=\alpha(\rho-1)+O(\alpha^2),
\qquad \rho=T/\pi_{\mathrm{old}}.
$$

不做 advantage normalization 时，raw gradient 量级为 $O(\alpha)$，学习会变慢；做标准化后，整体尺度被部分抵消，但 $\sigma_{\mathrm{group}}$ 同时变小，有限 batch 的归一化噪声会增大。$\alpha$ 越小意味着更低的方差、更小的单步信号和通常更多的有效更新。

我们的 mirror OPD 使用

$$
G_\eta(\pi_k)(a|s)
=\frac{\pi_k(a|s)^{1-\eta}T(a|s)^\eta}
{\sum_b\pi_k(b|s)^{1-\eta}T(b|s)^\eta}.
$$

它在 projective log-ratio 度量下有精确收缩和 Gibbs 闭式解，这是 TOP-D 没有的；TOP-D 则有 arithmetic interpolation 带来的全局 token reward 二阶矩界。

## TOP-D 定理一：有界方差

假设词表有限，且

$$
\|\nabla_\theta\log\pi_\theta(a|s)\|\leq M.
$$

令

$$
\widetilde g
=\nabla_\theta\log\pi_\theta(a|s)
\log\left(1-\alpha+\alpha\frac{T(a|s)}{\pi_\theta(a|s)}\right).
$$

则论文声称存在常数

$$
C^\star=\frac{(\log t^\star)^2}{t^\star-1},
\qquad
\log t^\star=2(1-1/t^\star),\quad t^\star>2,
$$

使

$$
\operatorname{Var}(\widetilde g)
\leq M^2|\mathcal V|
\max\{\log^2(1-\alpha),C^\star\alpha\}.
$$

证明先使用 $\operatorname{Var}(Z)\leq\mathbb E\|Z\|^2$。对单个 action 写 $p=\pi_\theta(a|s)$、$q=T(a|s)$，定义

$$
h(p,q)=p\left[\log(1-\alpha+\alpha q/p)\right]^2.
$$

若 $q<p$，则 $1-\alpha<1-\alpha+\alpha q/p<1$，所以

$$
h(p,q)\leq\log^2(1-\alpha).
$$

若 $q\geq p$，令 $u=1/p>1$。由 $q<1$，

$$
h(p,q)\leq\frac{\log^2(1+\alpha u)}{u}=f(u).
$$

驻点满足

$$
\frac{2\alpha u}{1+\alpha u}=\log(1+\alpha u).
$$

令 $t=1+\alpha u$，得到 $\log t=2(1-1/t)$。函数 $\log t-2(1-1/t)$ 在 $(1,2)$ 递减、在 $(2,\infty)$ 递增，且在 $2$ 为负、在 $e^2$ 为正，因此有唯一根 $t^\star>2$。代回得到

$$
f(u)\leq\frac{(\log t^\star)^2}{t^\star-1}\alpha=C^\star\alpha.
$$

两种区域合并后，每个 action 的 $h(p,q)$ 都不超过 $B_\alpha=\max\{\log^2(1-\alpha),C^\star\alpha\}$。有限词表求和给出 $\mathbb E[\widetilde r^2]\leq|\mathcal V|B_\alpha$。再由 $\|\widetilde g\|^2\leq M^2\widetilde r^2$，对 prefix 和 prompt 取期望，即得定理。

## TOP-D 定理二：算术近端算子收敛

定义

$$
\mathcal T_\alpha(\pi)=(1-\alpha)\pi+\alpha T,
\qquad
d(\pi,\rho)=\mathbb E_x\|\pi(\cdot|x)-\rho(\cdot|x)\|_1.
$$

若 $\pi_{k+1}=\mathcal T_\alpha(\pi_k)+\epsilon_k$，则

$$
d(\pi_{k+1},T)\leq(1-\alpha)^{k+1}d(\pi_0,T)
+\sum_{i=0}^{k}(1-\alpha)^{k-i}\|\epsilon_i\|_1.
$$

证明使用三角不等式：

$$
d(\pi_{k+1},T)\leq d(\mathcal T_\alpha(\pi_k),T)
+d(\pi_{k+1},\mathcal T_\alpha(\pi_k)).
$$

第二项是 $\|\epsilon_k\|_1$，第一项为

$$
d(\mathcal T_\alpha(\pi_k),T)
=\mathbb E_x\|(1-\alpha)(\pi_k-T)\|_1
=(1-\alpha)d(\pi_k,T).
$$

于是得到单步递推。反复代入即得有限步界。若 $\limsup_k\|\epsilon_k\|_1\leq\epsilon_\infty$，则

$$
\limsup_k d(\pi_{k+1},T)
\leq\epsilon_\infty\sum_{j=0}^{\infty}(1-\alpha)^j
=\frac{\epsilon_\infty}{\alpha}.
$$

## TOP-D 定理三：性能下界

假设响应长度不超过 $T_{\max}$，令 $\ell_\pi=\mathbb E_\pi|y|$，并定义 $d^\pi_{\mathrm{norm}}(s)=d^\pi(s)/\ell_\pi$。令

$$
\zeta_\pi(\widetilde\pi)
=\eta(\pi)+\ell_\pi
\mathbb E_{d^\pi_{\mathrm{norm}},\widetilde\pi}[A^\pi(s,a)],
\qquad
\xi=\max_s|\mathbb E_{\widetilde\pi}[A^\pi(s,a)]|.
$$

论文给出

$$
\eta(\widetilde\pi)
\geq\zeta_\pi(\widetilde\pi)
-2\xi T_{\max}\ell_\pi
\mathbb E_{d^\pi_{\mathrm{norm}}}
\left[\operatorname{TV}(\widetilde\pi,\pi)\right].
$$

证明从 performance difference identity 开始：

$$
\eta(\widetilde\pi)
=\eta(\pi)+\sum_s d^{\widetilde\pi}(s)
\mathbb E_{\widetilde\pi}[A^\pi(s,a)].
$$

而 $\zeta_\pi$ 使用 $d^\pi$。相减并使用 $\xi$：

$$
|\eta(\widetilde\pi)-\zeta_\pi(\widetilde\pi)|
\leq\xi\sum_s|d^{\widetilde\pi}(s)-d^\pi(s)|.
$$

对每个时间 $t$，构造只在第 $i$ 个 action 分布不同的混合轨迹。望远镜求和和确定性前缀转移给出

$$
\|\Pr_{\widetilde\pi}(s_t\in\cdot)-\Pr_\pi(s_t\in\cdot)\|_1
\leq2\sum_{i=1}^{t-1}\mathbb E_{s_i\sim\pi}
\left[\operatorname{TV}(\widetilde\pi,\pi)\right].
$$

再对 $t\leq T_{\max}$ 求和，交换时间顺序，并用 $T_{\max}-i\leq T_{\max}$：

$$
\sum_s|d^{\widetilde\pi}(s)-d^\pi(s)|
\leq2T_{\max}\sum_s d^\pi(s)
\operatorname{TV}(\widetilde\pi(\cdot|s),\pi(\cdot|s)).
$$

代入 $d^\pi=\ell_\pi d^\pi_{\mathrm{norm}}$ 即得定理。

## 我们的 mirror OPD 定理

### 定理四：projective 收缩

对正 support 的策略定义

$$
G_\eta(\pi)(a|s)
=\frac{\pi(a|s)^{1-\eta}T(a|s)^\eta}{Z_\eta(s)}.
$$

则

$$
\log\frac{G_\eta(\pi)(a|s)}{T(a|s)}
=(1-\eta)\log\frac{\pi(a|s)}{T(a|s)}-\log Z_\eta(s).
$$

对 action 取振幅，归一化常数消失，因此

$$
\operatorname{osc}_a\log\frac{G_\eta(\pi)}T
=(1-\eta)\operatorname{osc}_a\log\frac{\pi}T.
$$

若实际同步策略相对精确 target 的振幅误差为 $\varepsilon_k(s)$，则

$$
\Delta_{k+1}(s)\leq(1-\eta)\Delta_k(s)+\varepsilon_k(s),
$$

展开得到

$$
\Delta_K(s)\leq(1-\eta)^K\Delta_0(s)
+\sum_{j=0}^{K-1}(1-\eta)^{K-1-j}\varepsilon_j(s).
$$

### 定理五：mirror inner 的唯一解

固定 anchor $\pi_k$，令

$$
\mathcal J_k(p)
=\mathbb E_p\left[\log\frac{T}{\pi_k}\right]
-\frac1\eta\operatorname{KL}(p\|\pi_k).
$$

则

$$
\mathcal J_k(p)
=\frac1\eta\log Z_\eta
-\frac1\eta\operatorname{KL}(p\|G_\eta(\pi_k)).
$$

证明是展开 KL：

$$
\begin{aligned}
-\frac1\eta\operatorname{KL}(p\|G_\eta)
&=-\frac1\eta\mathbb E_p[
\log p-(1-\eta)\log\pi_k-\eta\log T+\log Z_\eta]\\
&=\mathbb E_p[\log(T/\pi_k)]
-\frac1\eta\operatorname{KL}(p\|\pi_k)
-\frac1\eta\log Z_\eta.
\end{aligned}
$$

移项即得。由于 KL 非负，唯一最优分布为 $p=G_\eta(\pi_k)$。当 $\eta=1$ 时，$G_1(\pi_k)=T$，所以 exact tabular vanilla mirror OPD 一轮到达 teacher。

### 定理六：mirror OPD 的条件方差界

若存在 support floor $\pi_k(a|s)\geq p_{\min}$、$T(a|s)\geq p_{\min}$，且 score gradient 有界，则

$$
g=\nabla_\theta\log\pi_\theta(a|s)\log\frac{T(a|s)}{\pi_k(a|s)}
$$

满足

$$
\operatorname{Var}(g)\leq M^2\log^2(1/p_{\min}).
$$

因为 $|\log(T/\pi_k)|\leq\log(1/p_{\min})$，所以 $\|g\|^2\leq M^2\log^2(1/p_{\min})$，最后使用 $\operatorname{Var}(g)\leq\mathbb E\|g\|^2$ 即得。没有 support floor、clipping 或其他尾部条件时，vanilla OPD 不存在 TOP-D 那样的统一全局方差界。

### 定理七：固定 anchor 下的性能下界

若在一个 outer 内冻结 $\pi_k$ 和 $r_k=\log T-\log\pi_k$，并满足有限 horizon 与有界 advantage，则可以把 TOP-D 的 performance-difference proof 应用于这个固定 reward MDP：

$$
\eta_{r_k}(p)
\geq\zeta_k(p)
-2\xi_kT_{\max}\ell_k
\mathbb E_{d^{\pi_k}_{\mathrm{norm}}}
\left[\operatorname{TV}(p,\pi_k)\right].
$$

证明需要把“固定 reward”写进每一步。记固定 reward MDP 下的 advantage 为 $A_k^{\pi_k}$，则 performance difference identity 给出

$$
\eta_{r_k}(p)-\eta_{r_k}(\pi_k)
=\sum_s d^p(s)\,\mathbb E_{a\sim p(\cdot|s)}[A_k^{\pi_k}(s,a)].
$$

定义

$$
\zeta_k(p)=\eta_{r_k}(\pi_k)+\sum_s d^{\pi_k}(s)\,\mathbb E_{a\sim p(\cdot|s)}[A_k^{\pi_k}(s,a)].
$$

两式相减，并令

$$
\xi_k=\max_s\left|\mathbb E_{a\sim p(\cdot|s)}[A_k^{\pi_k}(s,a)]\right|,
$$

得到

$$
\left|\eta_{r_k}(p)-\zeta_k(p)\right|
\leq \xi_k\sum_s|d^p(s)-d^{\pi_k}(s)|.
$$

有限 horizon 下，对每个时间 $t$ 构造只在第 $i$ 个 action 分布不同的混合轨迹。相邻混合轨迹的 future state map 相同，因而数据处理不等式给出

$$
\left\|\Pr_p(s_t\in\cdot)-\Pr_{\pi_k}(s_t\in\cdot)\right\|_1
\leq 2\sum_{i=1}^{t-1}\mathbb E_{s_i\sim\pi_k}
\left[\operatorname{TV}(p(\cdot|s_i),\pi_k(\cdot|s_i))\right].
$$

对 $t\leq T_{\max}$ 求和并交换 $t,i$ 顺序，每个第 $i$ 项最多出现 $T_{\max}$ 次，因此

$$
\sum_s|d^p(s)-d^{\pi_k}(s)|
\leq 2T_{\max}\ell_k\,
\mathbb E_{s\sim d^{\pi_k}_{\mathrm{norm}}}
\left[\operatorname{TV}(p,\pi_k)\right].
$$

代回绝对误差界并取下界即得定理。这个结论不适用于在同一个 inner block 中不断刷新 $\pi_k$ 的 vanilla OPD，因为此时 $r_k=\log T-\log\pi_k$ 也随 anchor 改变，前后两步不再是同一个 reward MDP，performance difference identity 不能直接串接。

## 结论

TOP-D 的主要优势是 arithmetic proximal reward 的统一二阶矩界、固定 old-policy batch 的多 epoch 数据复用，以及 PPO clip 对 inner drift 的控制。我们的 mirror OPD 的主要优势是精确 Gibbs target、projective contraction 和明确的 $\eta$-dependent 几何动力学。两者可以组合，但组合后需要重新证明：算术插值的方差定理属于 TOP-D reward，几何 mirror 的收缩定理属于我们的 operator，不能把二者的定理直接互换。

### 参考文献

[^topd]: Zhengpeng Xie, Li Lyna Zhang, Zeke Xie, and Mao Yang. _Trust Region Policy Distillation_. arXiv:2607.04751, 2026. [本地中文 TeX](../2607.04751_zh_CN/neurips_2026.tex)
