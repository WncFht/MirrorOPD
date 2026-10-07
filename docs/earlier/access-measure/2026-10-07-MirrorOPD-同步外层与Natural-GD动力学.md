# MirrorOPD 的同步外层与 Natural-GD 动力学

## 结论先行

把第 $k$ 次 outer step 的 student 权重同步为 $pi_k$，然后在整个 inner 阶段固定这个 reference，最后得到 $pi_{k+1}$，其数学对象是一个带误差的 Policy Mirror Descent（PMD）算子。MDPO 的“多个普通梯度步”只是对这个固定 reference 的 KL 镜像子问题做数值求解；它本身不是参数空间中的 Natural Gradient。Natural Gradient 只在局部二次模型、Fisher 矩阵取 exact 或 Gauss--Newton 近似、且步长足够小时与镜像步一阶等价。

对你文档中的 OPD 信号，若 teacher 条件分布为 $T(cdotmid s)$，anchor 为 $pi_k(cdotmid s)$，并令

$$
r_k(s,a)=\log T(a\mid s)-\log\pi_k(a\mid s),
$$

则精确 outer 解是

$$
\pi_{k+1}^{\star}(a\mid s)=
\frac{\pi_k(a\mid s)^{1-\eta}T(a\mid s)^\eta}
{Z_k(s)},
\qquad
Z_k(s)=\sum_a\pi_k(a\mid s)^{1-\eta}T(a\mid s)^\eta.
$$

因此 OPD 的同步动力学不是“每次把参数向 teacher 做一次普通梯度下降”，而是策略分布的几何插值。定义 $\rho_k(a\mid s)=\log[\pi_k(a\mid s)/T(a\mid s)]$，则

$$
\rho_{k+1}(a\mid s)=(1-\eta)\rho_k(a\mid s)-\log Z_k(s).
$$

对任意两个 action 作差后归一化常数消失：

$$
\rho_{k+1}(a\mid s)-\rho_{k+1}(a'\mid s)
=(1-\eta)\bigl[\rho_k(a\mid s)-\rho_k(a'\mid s)\bigr].
$$

当 $0<\eta<1$ 时，这是 projective log-ratio 的精确收缩；$\eta=1$ 一步得到 teacher；$1<\eta<2$ 交替收敛；$\eta=2$ 保持振幅；$\eta>2$ 发散。这个结论只需要固定 support 和精确词表解，不需要局部二次近似。

## 文档中的算法对象

你给出的 TeX 文件把 token-level 镜像子问题写成

$$
\max_{p\in\Delta(\mathcal A)}
\left\{\mathbb E_{a\sim p}[f(s,a)]-
\frac1\eta\operatorname{KL}(p\Vert\pi_k)\right\}.
$$

其唯一解由 KKT 条件给出 $p(a)\propto\pi_k(a)\exp(\eta f(a))$。PMD-Part 回归完整的 action-dependent log-ratio，PMD-Mean 先中心化 target，Bregman Policy Optimization（BPO）对 reverse-KL residual 求梯度，MDPO 直接最大化上述 KL 正则化目标。tabular、全 support、精确 action expectation 且 target 可表示时，四者的零点相同；有限参数、有限样本和有限 inner steps 会产生不同的投影轨迹。

同步 outer/inner 的顺序必须写清楚：

1. 在 outer $k$ 开始时保存 $pi_k$，并冻结 teacher $T$、rollout state 测度和 target reward $r_k$。
2. 固定 reference $pi_k$，在 student 参数 $\theta$ 上运行 $M_k$ 个 inner optimizer updates。
3. 将 inner 结束的策略记为 $pi_{k+1}$，只在此时刷新 reference，开始下一个 outer step。

如果在每个 inner update 后都令 reference 跟随 student，KL 项在新 anchor 处重新变成零，算法就不再是同一个 PMD/MDPO 子问题，也不能使用固定 reference 的 proximal 或 mirror descent 收敛证明。

经典 Mirror Descent 的几何基础是：在一个凸函数生成的 Bregman divergence 下，把线性化目标和 proximity penalty 放在同一个子问题中；KL mirror descent 只是负熵生成函数的特殊情形。[^mirror]

## 与原始论文的关系

### MDPO

MDPO 将标准 RL 目标在旧策略处线性化，再加上 $\operatorname{KL}(\pi\Vert\pi_k)$ proximity term，并用多个普通 SGD steps 近似求解 trust-region 子问题。论文明确区分了 on-policy 与 off-policy 版本：旧策略、状态访问测度和 replay correction 的选择属于估计问题，不能从镜像更新公式中自动消失。MDPO 的 inner optimizer 是 Euclidean parameter SGD；只有把策略分布当作坐标，或在 anchor 附近用 Fisher 二次模型，才可把它解释为 Natural Gradient 的局部近似。[^mdpo]

一个直接的实现后果是：在 $\theta_0=\theta_k$ 初始化时，$\operatorname{KL}(\pi_{\theta}\Vert\pi_k)$ 对 $\theta$ 的一阶梯度为零。因此第一次普通梯度步只看到 reward/advantage 的 policy-gradient 项；KL 的 trust-region 曲率要在后续 inner steps 中才会影响轨迹。只做一次 inner step 不能声称已经执行了完整 MDPO。

### PMD 与正则化 PMD

PMD 的 MDP 分析把每轮策略更新写成对旧策略的 Bregman proximal 子问题，并利用 exact policy improvement、regularizer 的凸性和 occupancy 分布得到全局收敛。Lan 的 PMD 工作给出强凸和一般凸 regularizer 下的线性收敛与随机采样复杂度；Zhan 等人的 GPMD 把 regularizer 和 Bregman divergence 统一起来，并明确分析 imperfect policy evaluation 与 imperfect policy update 的稳定性。[^pmd][^gpmd]

这些定理不能直接套在 OPD 上。它们的核心信号是某个 MDP value/advantage 的线性化，而 OPD 使用 $\log T-\log\pi_k$。OPD 因而首先拥有的是“朝 teacher 的镜像几何收缩”，而不是原始环境 return 的单调改进。若要得到 return 定理，还必须证明 teacher log-ratio 与某个 advantage 的关系，并控制 state occupancy shift、teacher bias 和 inner optimization error。

### Bregman Gradient Policy Optimization 与 Natural Policy Gradient

Bregman Gradient Policy Optimization 将 Bregman 几何和 policy-gradient/momentum 结合，在非凸参数化下给出 stationarity 样本复杂度；它支持选择不同 Bregman divergence，但这仍是优化框架，不等于任意固定 KL inner loop 都有 PMD 的 tabular 全局收敛。[^bgpo]

Natural Policy Gradient（NPG）在 Fisher--Rao 度量下使用

$$
\Delta\theta_k=\alpha F_k^{\dagger}g_k,
\qquad
F_k=\mathbb E_{s,a\sim\pi_k}
[\nabla_\theta\log\pi_k(a\mid s)\nabla_\theta\log\pi_k(a\mid s)^{\mathsf T}].
$$

它是策略分布上的 mirror/proximal 更新在局部坐标中的一阶表示。现有 NPG 收敛结果在 tabular 或具有明确 MDP regularity、advantage 估计和 step-size 条件下成立；不能把“使用 Fisher 预条件”本身当作 OPD 的全局定理。[^npg]

Fisher 度量作为自然梯度的来源可追溯到 Amari 的信息几何工作：参数更新应按分布之间的局部 KL 距离测量，而不是按任意 Euclidean 参数坐标测量。[^amari]

## 固定 reference 下的 Natural-GD 动力学

固定一个历史 $s$，令 student logits 为 $z_\theta(s)$，$\pi_\theta=\operatorname{softmax}(z_\theta)$。在同步后的 anchor $\theta_k$ 附近写 $\Delta=\theta-\theta_k$。KL 的二阶项为 categorical Fisher，参数空间曲率为

$$
A_k=\mathbb E_s[J_k(s)^{\mathsf T}
\bigl(\operatorname{Diag}(\pi_k)-\pi_k\pi_k^{\mathsf T}\bigr)J_k(s)],
$$

其中 $J_k$ 是 logits 对参数的 Jacobian。把固定 outer target 的一阶 policy-gradient 写成 $b_k$，则局部 inner 目标是

$$
Q_k(\Delta)=\frac1{2\eta}\Delta^{\mathsf T}A_k\Delta-b_k^{\mathsf T}\Delta.
$$

普通 Euclidean GD 的递推为

$$
\Delta_{m+1}=(I-\gamma A_k/\eta)\Delta_m+\gamma b_k.
$$

若在商空间上 $A_k$ 的特征值属于 $[m_k,M_k]$，则 $0<\gamma<2\eta/M_k$ 时收敛到 $\Delta_k^\star=\eta A_k^\dagger b_k$，收缩因子为

$$
q_k=\max_{\lambda\in[m_k,M_k]}\left|1-\frac{\gamma\lambda}{\eta}\right|.
$$

最优常数步长 $\gamma=2\eta/(m_k+M_k)$ 给出 $q_k=(\kappa_k-1)/(\kappa_k+1)$。因此固定 anchor 的 inner budget $M_k$ 只带来几何逼近误差 $q_k^{M_k}$；病态 Fisher 会直接放大所需 inner steps。

Natural-GD 取 $A_k^\dagger$ 作预条件时，局部一步是

$$
\Delta_{\mathrm{NG}}=\eta A_k^\dagger b_k.
$$

这与镜像子问题的局部最优位移一致到一阶。有限步仍有三个差异：参数化策略族通常不对指数族更新封闭；真实 objective 含有三阶及更高阶项；Fisher 可能使用 damping、采样估计或低秩近似。因此“MDPO = Natural GD”只在局部一阶意义成立。

## OPD 的 outer 定理

设 $\pi_{k+1}$ 是精确 OPD mirror target $\pi_{k+1}^\star$ 的近似，定义每个历史的 projective inner error

$$
\varepsilon_k(s)=\operatorname{osc}_a\left(
\log\frac{\pi_{k+1}(a\mid s)}{\pi_{k+1}^\star(a\mid s)}
\right).
$$

则 OPD 同步迭代满足

$$
\Delta_{k+1}(s)\le (1-\eta)\Delta_k(s)+\varepsilon_k(s),
$$

其中 $\Delta_k(s)=\operatorname{osc}_a\log[\pi_k(a\mid s)/T(a\mid s)]$。于是对 $0<\eta\le1$，

$$
\Delta_K(s)\le(1-\eta)^K\Delta_0(s)+
\sum_{j=0}^{K-1}(1-\eta)^{K-1-j}\varepsilon_j(s).
$$

若 $\varepsilon_j(s)\le\bar\varepsilon(s)$，则 $\limsup_K\Delta_K(s)\le\bar\varepsilon(s)/\eta$。这给出“inner 不必精确收敛，但误差必须小于 outer 收缩能吸收的量”的严格版本。若 inner 误差来自局部 Natural-GD，取 $\varepsilon_k(s)=O(q_k^{M_k})+O(\|\Delta\theta\|^2)$，即可得到可操作的 $M_k$ 与步长条件。

在 $\eta=1$ 时，精确 target 每个历史都是 teacher，但有限参数和有限 inner 只保证接近 teacher；在这种情况下增大 inner budget 比继续增加 outer steps 更直接。对 $\eta<1$，outer 本身提供稳定的 teacher-student interpolation，也允许用较少 inner steps 逐步逼近。

### 第一轮未收敛后重新 rollout 的动力学

记第 $k$ 轮的精确 mirror 算子为

$$
G_\eta(\pi_k)(a|s)=
\frac{\pi_k(a|s)^{1-\eta}T(a|s)^\eta}
     {\sum_b \pi_k(b|s)^{1-\eta}T(b|s)^\eta}.
$$

inner 只运行 $M_k$ 步时，实际同步策略可写成 $\pi_{k+1}=G_\eta(\pi_k)+e_k$；这里 $e_k$ 用 projective log-ratio 振幅定义。下一轮保存当前的 $\pi_{k+1}$，重新 rollout 并用 $r_{k+1}=\log T-\log\pi_{k+1}$ 定义新的 target，然后求解 $G_\eta(\pi_{k+1})$。它不会继续优化旧的 $G_\eta(\pi_k)$。

精确 tabular 情形给出

$$
\Delta_{k+1}(s)\le(1-\eta)\Delta_k(s)+\varepsilon_k(s),
$$

从而

$$
\Delta_K(s)\le(1-\eta)^K\Delta_0(s)+
\sum_{j=0}^{K-1}(1-\eta)^{K-1-j}\varepsilon_j(s).
$$

因此 $0<\eta\le1$ 且 $\varepsilon_k(s)\le\bar\varepsilon(s)$ 时，重新 rollout 后仍收敛到半径 $\bar\varepsilon(s)/\eta$ 的 teacher 邻域；若 $\varepsilon_k\to0$，则 $\Delta_k\to0$。第一轮提前同步的影响正是递推中的 $\varepsilon_0$，而不是一个永久固定的旧 target。

局部 Natural-GD 下，令第 $k$ 轮局部最优位移为 $\Delta\theta_k^\star=\eta A_k^\dagger b_k$。从同步点开始经过 $M_k$ 步 GD，有

$$
\|\Delta\theta_{k,M_k}-\Delta\theta_k^\star\|_{A_k}
\le q_k^{M_k}\|\Delta\theta_k^\star\|_{A_k}.
$$

故下一轮的残差可分解为

$$
\varepsilon_{k+1}\le C_{\mathrm{proj}}q_{k+1}^{M_{k+1}}
\|\Delta\theta_{k+1}^\star\|_{A_{k+1}}
+C_{\mathrm{nl}}\|\Delta\theta_{k+1}^\star\|^2
+C_{\mathrm{rollout}}\xi_{k+1}.
$$

三项分别来自 inner 未充分收敛、局部二次近似误差和新 rollout 的估计误差。增大 $M_k$ 只能降低第一项。实际 stopping rule 应检查当前策略到本轮固定 target 的 residual，达到阈值后同步并重新 rollout。$\eta=1$ 时每轮精确 target 都是同一个 teacher，外层本身没有额外的 mirror contraction；但若 inner 求解器对 teacher 的误差满足 $\varepsilon_{k+1}^{\mathrm{inner}}\le q^{M}\Delta_k$，则分散到多轮的 inner 仍给出 $\Delta_{k+1}\le q^M\Delta_k$。这只是把连续的 inner steps 分批执行，不能消除 rollout 噪声、参数投影误差或 optimizer 状态变化。

### 与 TOP-D 的关系：小 $\eta$ 不等于小 $\alpha$

neurips_2026.tex 中的 TOP-D 使用概率空间的算术插值

$$
H_\alpha(\pi_k)=(1-\alpha)\pi_k+\alpha T,
\qquad
\widetilde r_\alpha=\log\frac{H_\alpha(\pi_k)}{\pi_k}
=\log\left(1-\alpha+\alpha\frac{T}{\pi_k}\right).
$$

我们的 OPD mirror step 则是

$$
G_\eta(\pi_k)=
\frac{\pi_k^{1-\eta}T^\eta}{Z_k},
\qquad
r_\eta=\log\frac{G_\eta(\pi_k)}{\pi_k}
=\eta\log\frac{T}{\pi_k}-\log Z_k.
$$

二者的差别不是记号差别。$\alpha$ 控制算术混合的概率位移，$\eta$ 控制 KL mirror 子问题的指数倾斜强度。对小步长展开，令 $r=\log(T/\pi_k)$，则

$$
H_\alpha(\pi_k)=\pi_k+\alpha(T-\pi_k),
$$

而

$$
G_\eta(\pi_k)(a)
=\pi_k(a)\left[1+\eta\bigl(r(a)-\mathbb E_{\pi_k}r\bigr)\right]+O(\eta^2).
$$

因此，只有在 $T$ 已经接近 $\pi_k$ 时，使用 $r=\log(T/\pi_k)\approx(T-\pi_k)/\pi_k$，两者才满足 $\alpha\approx\eta$ 的一阶近似。远离 teacher 时，TOP-D 的算术插值仍是线性概率更新，而 mirror OPD 是 Fisher 几何下的指数更新。

### 可以迁移的方差定理与其边界

TOP-D 的有界方差证明依赖关键恒等式

$$
\widetilde r_\alpha=\log(1-\alpha+\alpha\rho),
\qquad \rho=T/\pi_k,
$$

它给出下界 $\widetilde r_\alpha\geq\log(1-\alpha)$。在有限词表和 score gradient 满足 $\|\nabla_\theta\log\pi_\theta\|\leq M$ 时，论文得到 token gradient 二阶矩的统一上界，形式为

$$
\mathbb E\|\widetilde g\|^2
\leq M^2|\mathcal V|
\max\{\log^2(1-\alpha),C^\star\alpha\}.
$$

这个结论不能直接替换成我们的小 $\eta$ 结论。mirror reward $\eta\log(T/\pi_k)$ 仍然向下无界；$\eta$ 只做缩放，不建立统一的 support-independent 方差上界。二元反例是

$$
\pi_k=(1/2,1/2),\qquad T=(\epsilon,1-\epsilon),
$$

此时

$$
\mathbb E_{\pi_k}\left[\log^2(T/\pi_k)\right]
\geq\frac12\log^2(2\epsilon)\longrightarrow\infty
\quad(\epsilon\to0).
$$

所以 vanilla OPD 或 mirror OPD 要得到类似 TOP-D 的 uniform variance theorem，必须额外假设 teacher/student 概率下界，或者显式做 reward clipping、概率平滑或 support restriction。例如若 $T(a|s),\pi_k(a|s)\geq p_{\min}>0$，则 $|\log(T/\pi_k)|\leq\log(1/p_{\min})$，从而

$$
\operatorname{Var}(g_k)
\leq M^2\log^2(1/p_{\min}).
$$

这是真正可迁移到 OPD 的方差定理，但它的常数依赖 support floor，不能宣称 vanilla OPD 在整个概率空间上天然有界。

还要区分 token-level estimator 和完整 policy-gradient estimator。论文的定理只直接控制 $\nabla\log\pi(y_k|s_k)\,\widetilde r_k$ 的二阶矩；若使用完整 return $\sum_{j\geq k}r_j$，还需要有限 horizon、奖励二阶矩和长度尾部条件，不能由 token 定理自动推出整条 response 梯度的方差界。

### 收敛定理的对应关系

TOP-D 的收敛定理来自算术算子

$$
H_\alpha(\pi)=(1-\alpha)\pi+\alpha T,
$$

因此在 $L_1$ 距离中有精确恒等式

$$
d(H_\alpha(\pi),T)=(1-\alpha)d(\pi,T).
$$

加上每轮优化误差 $\epsilon_k$ 就得到论文的

$$
d(\pi_{k+1},T)\leq(1-\alpha)^{k+1}d(\pi_0,T)
+\sum_{j=0}^{k}(1-\alpha)^{k-j}\|\epsilon_j\|_1.
$$

我们的几何 mirror 算子在 projective log-ratio 度量中有对应的精确恒等式

$$
\operatorname{osc}_a\log\frac{G_\eta(\pi_k)}{T}
=(1-\eta)\operatorname{osc}_a\log\frac{\pi_k}{T}.
$$

因此两套理论的结构完全平行，但度量不同：TOP-D 是 $L_1$/TV 的算术收缩，我们是 Hilbert/projective log-ratio 的几何收缩。若假设所有有效 action 的概率都落在 $[p_{\min},1]$，则这些度量在该紧集上等价，可以把我们的 projective 收敛转换成 TV 或 $L_1$ 收敛；没有这个 bounded-ratio 假设，不能直接把 projective contraction 改写成全局 $L_1$ contraction。

### 性能下界能否迁移

论文的性能下界本质上是有限时域 MDP 的 performance-difference bound。对固定 outer anchor $\pi_k$，冻结 $r_k=\log T-\log\pi_k$，可以定义

$$
\zeta_k(p)=\eta_r(\pi_k)+
\ell_k\,\mathbb E_{s\sim d^{\pi_k}_{\mathrm{norm}},a\sim p}[A^{\pi_k}(s,a)].
$$

在有限 horizon、$|A^{\pi_k}|\leq A_{\max}$ 和新旧策略 TV 距离可控时，可得到同型下界

$$
\eta_r(p)
\geq
\zeta_k(p)-
2A_{\max}T_{\max}\ell_k
\mathbb E_{d^{\pi_k}_{\mathrm{norm}}}
\left[\operatorname{TV}(p(\cdot|s),\pi_k(\cdot|s))\right].
$$

但这只是固定 reward 的 surrogate bound。vanilla OPD 的 reward $\log T-\log\pi$ 随当前策略变化；若没有在一个 outer 内冻结 anchor，就不能把每一步都当成同一个固定-reward MDP 来套用该定理。即使冻结了 anchor，还必须证明实际 PPO/clipping 或 Natural-GD 更新确实提高了 $\zeta_k$，而论文给出的 PPO 目标本身没有自动保证这一点。

对 mirror OPD，若额外假设 $|r_k|\leq B$，则

$$
\operatorname{TV}(G_\eta(\pi_k),\pi_k)
\leq\sqrt{\tfrac12\operatorname{KL}(G_\eta(\pi_k)\|\pi_k)}
=O(\eta B).
$$

代入上面的性能下界，性能误差项是 $O(T_{\max}\xi_k\eta B)$。这说明小 $\eta$ 可以降低单步 occupancy-shift penalty，但并不自动保证性能提升；仍需 surrogate advantage gain 大于该 penalty。

### vanilla OPD 的理论位置

vanilla OPD 对应 TOP-D 的 $\alpha=1$，即时信号为

$$
r_k=\log\frac{T(a_k|s_k)}{\pi_k(a_k|s_k)}.
$$

它有三个层次的结论：

1. 在 exact tabular、固定 prefix、完整 action expectation 且 inner 全局收敛时，$\eta=1$ 的 mirror OPD 一步得到 $T$；这是分布空间的精确结论。
2. 在有限参数和有限 inner steps 下，只能得到 teacher projection 加优化误差；若梯度噪声没有 support floor 或 clipping，则不存在 TOP-D 那种全局统一方差界。
3. 对实际 vanilla policy-gradient，若假设目标 $J(\theta)$ 是 $L$-smooth、梯度估计无偏且方差 $\sigma^2$ 有界，则标准非凸 SGD 只能推出

$$
\frac1K\sum_{k=0}^{K-1}\mathbb E\|\nabla J(\theta_k)\|^2
=O\left(\frac{J(\theta_0)-J_*}{\gamma K}+L\gamma\sigma^2\right).
$$

若再有 PL/强凹条件，则可得到线性收敛到 $O(\gamma\sigma^2)$ 噪声地板。vanilla OPD 的困难在于这些条件恰好由无界 log-ratio、teacher support mismatch、长 response return 和 on-policy occupancy drift 破坏，因此不能无条件声称全局收敛或单调性能提升。

## 访问测度与 OPD 的边界

如果 loss integrand 只依赖当前 prefix $s_t=(x,y_{<t})$ 和当前 action $a_t$，目标是 $d_{\pi_k}(s_t)\pi_k(a_t\mid s_t)$，而数据来自行为测度 $\nu$，只需使用历史权重

$$
\Omega_k(s_t)=\frac{d_{\pi_k}(s_t)}{d_\nu(s_t)}.
$$

未来 suffix 不出现在 integrand 中，额外乘 suffix ratio 会改变目标并增大方差。若回放数据同时改变了当前 action 测度，则还要单独处理 action importance ratio；这和 history ratio 是两个不同的校正问题。

对 OPD，最干净的理论对象是：固定 prompt batch、固定 anchor rollout、固定 $T$ 和 $r_k$，在 anchor prefix 测度下优化一个局部 token objective。若想把结论推广到完整 sequence-level objective，必须额外控制 prefix occupancy drift。现有 TeX 中的 sequence experiment 已经显示，未访问 prefix 的误差不会因其他 prefix 的 inner 收敛而自动消失；这也是 LLM OPD 与独立 token toy 的主要差别。

## 可直接采用的分析路线

建议把 OPD 的证明分成三层。第一层在每个固定 prefix 上证明 Gibbs target 和 projective contraction；第二层在固定 reference 的参数空间证明 Natural-GD/普通 GD 的 inner 逼近误差；第三层再用 occupancy mismatch、teacher approximation 和采样误差把局部误差汇总到 sequence-level。第一层已经是精确结论，第二层需要 Fisher 非退化或阻尼后的谱界，第三层才需要 RL 特有的 concentrability 或 prefix-occupancy 假设。

实现上，outer loop 应显式保存 `anchor_state_dict`，所有 inner steps 读取同一份 $\pi_k$ 的 log-prob；inner 结束后再同步。记录以下三个量足以诊断动力学：当前策略到精确 mirror target 的 KL、projective log-ratio 振幅 $\Delta_k$、以及 inner 梯度预条件后的残差。只看 teacher KL 无法区分“target 本身在移动”和“inner 没有求解好”这两种情况。

## Outer 同步策略

### Hard sync：OPD 的默认策略

Hard sync 在 outer 边界执行

$$
\pi_k\leftarrow\pi_{\mathrm{student}},
\qquad
\text{freeze }\pi_k\text{ for }M_k\text{ inner updates}.
$$

它正好对应前面的固定-reference 镜像子问题，因而保留 Gibbs 闭式解、projective contraction 和 inexact recursion。MDPO 的 on-policy 算法也是一轮采样后固定旧策略并重复多个 SGD steps；原文实验发现 $m=1$ 不能让 KL proximity 生效，$m=10$ 在其 MuJoCo on-policy 设置中表现较好。这个数值不能直接搬到 LLM，但“同一 outer batch 内不刷新 reference”是可迁移的结构。[^mdpo]

Hard sync 的同步周期不应由 wall-clock 或任意固定 epoch 决定，而应由 inner residual 决定。令 $e_{k,m}$ 为当前 student 到固定 mirror target 的残差，在局部二次模型中有

$$
\lVert e_{k,m}\rVert_{A_k}\le q_k^m\lVert e_{k,0}\rVert_{A_k}.
$$

选择最小的 $M_k$ 使得

$$
q_k^{M_k}\lVert e_{k,0}\rVert_{A_k}\le c\,\eta_k^2
$$

其中 $c$ 是允许的常数，就能把 outer 稳态误差 $\varepsilon_k/\eta_k$ 控制在 $O(\eta_k)$。如果只要求固定误差地板 $\bar\varepsilon$，右侧可改为 $c\eta_k$；但此时减小 $\eta_k$ 不会自动提高最终精度，因为误差项会被 $1/\eta_k$ 放大。

TRPO 也在一批 rollout 内冻结 old policy，但它只做一次受 KL 约束的 natural-gradient/line-search 更新，然后用新 rollout 刷新 old policy；PPO 则在同一批 rollout 上做多 epoch，并用 old/new ratio clipping 近似限制漂移。[^trpo][^ppo] 这两种算法都支持“采样批次边界同步”，但 PPO 的 clip 约束不等于 OPD 的 KL mirror target，不能把 clip ratio 当成 $\eta$ 的直接替代品。

### Delayed sync：把 reference 当作 lagged target

若每个 outer batch 都刷新过于昂贵，可以让同一份 reference 跨 $L>1$ 个 batch 保持不变。此时第 $j$ 个 batch 实际求解的是以旧 anchor $\bar\pi$ 为中心的镜像问题，误差递推包含额外的 anchor drift：

$$
\varepsilon_{k+j}^{\mathrm{total}}
\le \varepsilon_{k+j}^{\mathrm{inner}}
 +\operatorname{osc}_a\log\frac{\pi_{k+j}(a\mid s)}{\bar\pi(a\mid s)}.
$$

因此 delayed sync 只有在每个 batch 的 policy drift 很小、或者 replay 数据本来就以同一行为策略生成时才有理论依据。对 OPD，teacher ratio 本身依赖当前 anchor；跨 batch 不刷新会让 $r=\log T-\log\bar\pi$ 变成 stale signal。建议把 delayed sync 作为吞吐优化，而不是默认算法。

### Soft/Polyak sync：稳定但改变算法

soft sync 形如

$$
\theta_k^{\mathrm{ref}}
\leftarrow(1-\tau)\theta_{k-1}^{\mathrm{ref}}+\tau\theta_k^{\mathrm{student}},
\qquad 0<\tau<1.
$$

它可以降低 reference 的批间噪声，却不再等价于 $\pi_k$ 的 hard PMD。参数空间线性插值也不等于策略分布的几何插值；即使 logits 线性插值，softmax 后的策略仍不是 $\pi_{k-1}^{1-\tau}\pi_k^\tau$ 的一般实现。若使用 soft sync，必须把 reference 作为独立状态并分析 two-timescale 动力学；不能继续使用 hard-sync 的 exact OPD theorem。对于当前 OPD，soft sync 适合在实验中作为 ablation，不适合作为主理论对象。

### 每个 inner step 都 sync

若每个 inner step 后令 reference 等于 student，KL gradient 在新的 reference 处反复归零，参数优化只剩下随时间变化的 teacher direction。tabular 中若每次都能精确求解，连续执行微小的 geometric teacher interpolation 仍可产生类似的总路径；神经网络中却会叠加 stale-gradient、optimizer momentum 和 Fisher 重估误差，已不是“固定 reference 的一次 PMD step”。因此实现上应把同步点放在 outer loop，而不是 optimizer step hook 中。

## $\eta$ 的数学作用与选择

### 精确 OPD 中的三种含义

在 $p_\eta(a)\propto\pi_k(a)\exp(\eta r_k(a))$ 中，$\eta$ 同时决定三个量：teacher 插值比例、projective 收缩率和 KL 子问题曲率。对固定 teacher，$0<\eta\le1$ 时每个 outer 的 projective 收缩为 $1-\eta$；局部参数目标的 KL Hessian 是 $A_k/\eta$；若普通 GD 学习率是 $\gamma$，稳定条件是 $\gamma<2\eta/M_k$。所以减小 $\eta$ 会让单个 outer 更保守，同时让 inner 目标更硬，要求把参数学习率按 $\gamma\propto\eta$ 缩放。

若使用精确 mirror outer 且 teacher 固定，常数 $\eta$ 的达到误差 $\epsilon$ 所需 outer steps 约为

$$
K\ge\frac{\log(\Delta_0/\epsilon)}{-\log(1-\eta)}
\simeq\frac1\eta\log\frac{\Delta_0}{\epsilon}
\quad(\eta\ll1).
$$

因此很小的 $\eta$ 不是免费的稳定性：它把计算从 inner 转移到更多 outer refreshes。另一方面，$\eta=1$ 虽然在 exact tabular 情形一步到 teacher，却要求 inner 求解、teacher logits、访问测度和参数表示都足够准确；在有限样本神经网络中通常会放大 projection error。

### 用 KL budget 选择 $\eta$

比直接固定 $\eta$ 更稳健的规则是指定 anchor KL budget $\delta_k$，在每个 prefix 解一维方程

$$
\operatorname{KL}(p_\eta(\cdot\mid s)\Vert\pi_k(\cdot\mid s))=\delta_k.
$$

该方程可解是因为

$$
\operatorname{KL}(p_\eta\Vert\pi_k)
=\eta\,\mathbb E_{p_\eta}[r_k]-\log Z_k(\eta),
$$

且

$$
\frac{\mathrm d}{\mathrm d\eta}
\operatorname{KL}(p_\eta\Vert\pi_k)
=\eta\operatorname{Var}_{p_\eta}(r_k)\ge0.
$$

所以在 $0\le\eta\le1$ 上 KL 随 $\eta$ 单调增加，可以用 bisection 得到唯一的 $\eta_k$，再对所有 prefix 取分位数或保守最小值。小步长下

$$
\operatorname{KL}(p_\eta\Vert\pi_k)
=\frac{\eta^2}{2}\operatorname{Var}_{\pi_k}(r_k)+O(\eta^3),
$$

给出无需逐点求根的初值

$$
\eta_k\approx
\sqrt{\frac{2\delta_k}
{\widehat{\operatorname{Var}}_{\pi_k}(r_k)+\epsilon_{\mathrm{var}}}}.
$$

这条规则自动对 teacher/student 分歧大的 prefix 减小步长，对分歧小的 prefix 放大步长；最后仍应截断到 $[\eta_{\min},1]$。如果只能用 sampled actions，必须对方差估计和 $\eta$ 加上 confidence margin，避免少数尖锐 ratio 造成过大的实际 KL。

### 适合 OPD 的默认 schedule

当前阶段建议把 hard sync 作为主算法，先固定 $\eta_k=\eta$ 做机制实验，再引入 KL-budget adaptation。初始 sweep 可以覆盖 $\eta\in\{0.1,0.3,0.5,0.7,1.0\}$，但选择标准应是三条曲线同时满足：anchor KL 不超过预算、projective residual 按预期衰减、inner residual 在同步前达到阈值。若 $\eta=1$ 的 inner residual 明显高于较小 $\eta$，应优先增加 $M_k$ 或降低参数学习率，而不是继续堆 outer steps。

若需要随训练进度改变 $\eta_k$，应保持 $0\le\eta_k\le1$ 且 $\sum_k\eta_k=\infty$；否则即使每个 inner 都精确，$\prod_k(1-\eta_k)$ 也可能不趋于零。一个保守 schedule 是前期使用 KL-budget 控制的常数区间，后期在 residual 已小于阈值后再逐渐减小 $\eta_k$ 以降低噪声。MDPO 原文的 $t_k=1-k/K$ 是其有限 horizon 的理论启发式，不能直接作为 OPD 的通用 schedule，尤其不能在未知训练总步数时照搬。

## 对 OPD 实验的具体落地

每个 outer 记录 anchor 到 student 的实际 forward KL、精确或估计的 mirror-target KL、$\widehat{\operatorname{Var}}(r_k)$ 和 inner residual。用实际 target KL 校验训练代码是否真的执行了 hard sync；只看权重是否复制成功不够，因为 optimizer momentum、EMA 和 mixed-precision master weights 也可能保留旧状态。比较同步策略时至少做 hard sync、delayed sync 和 soft sync 三组，保持总 token/update 预算不变；否则更快的 wall-clock 可能只是少做了 inner updates。

最小可验证实验是固定一个 teacher 和一批 prefix，分别运行五个 $\eta$ 值，每个 outer 用 residual stopping 而不是固定 $M$。如果理论正确，应观察到：exact-table 版本的 projective residual 斜率接近 $\log(1-\eta)$；参数版本的偏差由 $q^M$ 和 Fisher 近似误差主导；soft/delayed sync 的偏差随 anchor drift 增大。这个实验能把“同步策略的影响”和“teacher 质量、rollout 方差、模型容量”分离开。

## 固定 $\eta$ 时的完整 pipeline

固定 $\eta$ 只固定每个 outer prox 子问题中的 KL 系数。它不固定 value function、advantage、anchor policy、state occupancy 或 rollout 数据。标准迭代仍然是

$$
\pi_0\to\text{估计 }Q^{\pi_0}\to\pi_1
\to\text{估计 }Q^{\pi_1}\to\pi_2\to\cdots.
$$

如果只用 $\pi_0$ 生成一批数据、做完 inner 后就停止，那只是完成了 $k=0$ 的一次 mirror/prox update；如果继续用这批数据和 $Q^{\pi_0}$ 更新很多 outer steps，得到的是固定 surrogate 上的重复优化，不是 PMD。

### Exact PMD

Lan 的 deterministic PMD 每轮直接执行

$$
\pi_{k+1}(\cdot\mid s)=
\arg\min_{p\in\Delta(\mathcal A)}
\left\{\eta_k\left[\langle Q^{\pi_k}(s,\cdot),p\rangle+h(p)\right]
 +D_{\pi_k}(p)\right\}.
$$

因此它的逻辑是：先有当前策略 $\pi_k$，得到当前策略的 action-value $Q^{\pi_k}$，对每个状态解一次 prox map，得到 $\pi_{k+1}$，再进入下一轮。若 $Q^{\pi_k}$ 可精确计算，算法没有额外的 optimizer inner loop；所谓 inner 只出现在近似求解 prox map 或估计 $Q^{\pi_k}$ 时。PMD 的 stochastic 版本在每个 $k$ 使用针对 $\pi_k$ 的新随机估计量 $Q^{\pi_k,\xi_k}$，而不是把 $\pi_0$ 的估计永久复用。[^pmd]

PMD 论文的 constant-step-size theorem 是 $\eta_k=\eta$，但 theorem 中的 $Q^{\pi_k}$ 和 reference $\pi_k$ 仍然随 $k$ 变化。固定的是系数，不是整个 objective。强凸 regularizer 下它可以得到线性收敛；一般凸 regularizer 下是相应的次线性结论。近似 subproblem 的误差会直接进入每一轮的误差项，因此“inner 解得不准”与“outer 没有刷新”是两个不同问题。

### MDPO on-policy

MDPO 把 PMD 的一轮写成可用神经网络执行的 inner 优化。第 $k$ 轮的 pipeline 是：

1. 当前 student 是 $\pi_k=\pi_{\theta_k}$。从它采样一批 trajectories，得到访问状态和 advantage 估计 $\widehat A^{\pi_k}$。
2. 保存并冻结 old policy $\pi_k$，同时固定这批 trajectories、advantage 和 state measure。
3. 初始化 $\theta_k^{(0)}=\theta_k$，在同一个目标
   $$
   \Psi_k(\theta)=
   \mathbb E_{s\sim\rho_{\pi_k},a\sim\pi_\theta}
   \left[\widehat A^{\pi_k}(s,a)
   -\frac1\eta\operatorname{KL}(\pi_\theta\Vert\pi_k)\right]
   $$
   上做 $m$ 个普通 SGD steps。
4. 令 $\theta_{k+1}=\theta_k^{(m)}$，丢弃或标记这批 on-policy 数据过期，重新从 $\pi_{k+1}$ 采样下一轮。

MDPO 论文中 $m=10$ 是其 on-policy 实验的经验选择；它不是 outer 迭代次数，也不是“固定数据训练到全局最优”的要求。$m$ 只是近似求解第 $k$ 个固定-reference 子问题的计算预算。论文还强调，$m=1$ 时 anchor KL 的一阶梯度在初始化点为零，更新退化成 vanilla policy gradient。[^mdpo]

### MDPO off-policy

off-policy MDPO 从 replay buffer 取状态和 transition，用 critic 估计当前 policy 的 $Q^{\pi_k}$ 或 advantage，再在固定 old-policy copy $\pi_k$ 下优化 actor objective。由于每个 environment step 都做 actor update，论文还讨论了让 old-policy copy 跨若干 gradient updates 保持不变，用较少的显式 outer refresh 模拟多 inner steps。这个版本不应与 on-policy 的“每轮一批完整 rollout”混同：它的 data measure 是 replay 分布，必须单独分析 state/action correction 和 policy lag。[^mdpo]

### 你的 OPD

你的 OPD 可以直接写成以下时间线：

$$
\begin{aligned}
&\text{同步得到 }\pi_k;\\
&\text{从 }\pi_k\text{ 采样 prompt--response rollouts，或从 replay 校正到 }d_{\pi_k};\\
&\text{冻结 }\pi_k,T,\ r_k=\log T-\log\pi_k;\\
&\text{固定 }r_k\text{ 和 anchor，在 inner 中优化 }M_k\text{ 次};\\
&\text{结束后同步 student 为 }\pi_{k+1};\\
&\text{用 }\pi_{k+1}\text{ 重新生成下一轮数据。}
\end{aligned}
$$

在 exact token-level 情形，$r_k$ 依赖当前 anchor，所以每一轮的 target 都会移动：

$$
\pi_{k+1}^{\star}\propto
\pi_k^{1-\eta}T^\eta.
$$

如果把 $\pi_0$ 的 rollout 永久保留并且仍使用 $r_0=\log T-\log\pi_0$，你优化的是固定 target $\pi_0^{1-\eta}T^\eta$ 的投影；这不是连续 OPD。只有在 off-policy 版本中显式估计 prefix occupancy ratio、action ratio 或使用完整词表 expectation，旧数据才可以作为下一轮 $\pi_k$ 目标的无偏或可控偏差估计。

因此，对你的问题最直接的回答是：**固定 $\eta$ 时，每个 outer 都重新产生或重新校正数据；只在该 outer 内复用同一批数据做多个 inner steps。** 固定的是每一轮的镜像步长，不是 $\pi_0$、$Q^{\pi_0}$ 或 rollout batch。

## $\eta=1$：理想情形下的一步 teacher 投影

对 OPD 的固定-prefix 子问题，令

$$
J_k(p)=\mathbb E_{a\sim p}[\log T(a\mid s)-\log\pi_k(a\mid s)]
-\frac1\eta\operatorname{KL}(p\Vert\pi_k).
$$

在 $\eta=1$ 时，KL 展开后 anchor 项完全抵消：

$$
\begin{aligned}
J_k(p)
&=\mathbb E_p[\log T-\log\pi_k]
 -\mathbb E_p[\log p-\log\pi_k]\\
&=\mathbb E_p[\log T-\log p]
=-\operatorname{KL}(p\Vert T).
\end{aligned}
$$

因此，只要 inner 在策略分布空间中求到全局最优，唯一解就是

$$
p^\star(\cdot\mid s)=T(\cdot\mid s).
$$

这也直接来自一般 Gibbs 解

$$
p^\star_\eta(a\mid s)
=\frac{\pi_k(a\mid s)^{1-\eta}T(a\mid s)^\eta}
{\sum_b\pi_k(b\mid s)^{1-\eta}T(b\mid s)^\eta},
$$

因为 $\eta=1$ 时分子为 $T(a\mid s)$，配分函数为 $1$。所以在 exact tabular、完整词表、全 prefix 覆盖和无限 inner budget 的理想模型里，$\pi_0$ 只需要产生第一轮访问测度；第一个 outer 结束后就有 $\pi_1=T$，之后 $r_1=0$，策略保持不变。

这并不意味着实际神经网络训练只要把超参数设为 $\eta=1$ 就一定得到 teacher。需要同时满足以下条件：

- inner optimizer 找到 $-\operatorname{KL}(\pi_\theta\Vert T)$ 在参数族中的全局最优，而不是局部驻点或受容量限制的投影；
- 所有需要匹配的 prefix 都被访问，或者 loss 使用完整 prefix/vocabulary expectation；
- teacher 在 anchor 的 support 上有有限 log-prob，不能直接出现 $\log T=-\infty$；
- sampled action、prefix occupancy、teacher logits 和 normalization 的估计误差足够小。

如果只在一批 $\pi_0$ rollout 上训练，inner 收敛的对象是这些 rollout prefix 上的 teacher 投影。未访问 prefix 的参数没有被约束，因而不能推出全局的 $\pi_1=T$。如果使用参数化 student，实际极限是

$$
\pi_{\theta_1}\in
\arg\min_{\theta}\mathbb E_{s\sim\text{training measure}}
\operatorname{KL}(\pi_\theta(\cdot\mid s)\Vert T(\cdot\mid s)),
$$

而不是逐状态精确等于 teacher。

对 $0<\eta<1$，inner 完全收敛时得到的是 teacher 的几何插值，而不是 teacher 本身；选择较小 $\eta$ 的作用是缩短单个 target 的距离、降低有限样本和参数投影的风险。代价是需要更多 outer refreshes。因而 $\eta=1$ 应被理解为“最激进的一次 teacher projection”，而不是自动正确的默认值。

### 参考文献

[^mdpo]: Manan Tomar, Lior Shani, Yonathan Efroni, and Mohammad Ghavamzadeh. _Mirror Descent Policy Optimization_. arXiv 2005.09814, 2020. [论文](https://arxiv.org/abs/2005.09814)

[^mirror]: Amir Beck and Marc Teboulle. _Mirror Descent and Nonlinear Projected Subgradient Methods for Convex Optimization_. Operations Research Letters 31(3), 2003. [DOI](<https://doi.org/10.1016/S0167-6377(02)00231-6>)

[^pmd]: Guanghui Lan. _Policy Mirror Descent for Reinforcement Learning: Linear Convergence, New Sampling Complexity, and Generalized Problem Classes_. arXiv 2102.00135, 2021. [论文](https://arxiv.org/abs/2102.00135)

[^gpmd]: Wenhao Zhan, Shicong Cen, Baihe Huang, Yuxin Chen, Jason D. Lee, and Yuejie Chi. _Policy Mirror Descent for Regularized Reinforcement Learning: A Generalized Framework with Linear Convergence_. arXiv 2105.11066, 2021. [论文](https://arxiv.org/abs/2105.11066)

[^bgpo]: Feihu Huang, Shangqian Gao, and Heng Huang. _Bregman Gradient Policy Optimization_. ICLR 2022. [论文](https://arxiv.org/abs/2106.12112)

[^npg]: Sajad Khodadadian, Prakirt Raj Jhunjhunwala, Sushil Mahavir Varma, and Siva Theja Maguluri. _On the Linear Convergence of Natural Policy Gradient Algorithm_. arXiv 2105.01424, 2021. [论文](https://arxiv.org/abs/2105.01424)

[^amari]: Shun-ichi Amari. _Natural Gradient Works Efficiently in Learning_. Neural Computation 10(2), 1998. [DOI](https://doi.org/10.1162/089976698300017746)

[^trpo]: John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel. _Trust Region Policy Optimization_. arXiv 1502.05477, 2015. [论文](https://arxiv.org/abs/1502.05477)

[^ppo]: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. _Proximal Policy Optimization Algorithms_. arXiv 1707.06347, 2017. [论文](https://arxiv.org/abs/1707.06347)
