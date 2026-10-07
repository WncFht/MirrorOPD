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

## 访问测度与 OPD 的边界

如果 loss integrand 只依赖当前 prefix $s_t=(x,y_{<t})$ 和当前 action $a_t$，目标是 $d_{\pi_k}(s_t)\pi_k(a_t\mid s_t)$，而数据来自行为测度 $\nu$，只需使用历史权重

$$
\Omega_k(s_t)=\frac{d_{\pi_k}(s_t)}{d_\nu(s_t)}.
$$

未来 suffix 不出现在 integrand 中，额外乘 suffix ratio 会改变目标并增大方差。若回放数据同时改变了当前 action 测度，则还要单独处理 action importance ratio；这和 history ratio 是两个不同的校正问题。

对 OPD，最干净的理论对象是：固定 prompt batch、固定 anchor rollout、固定 $T$ 和 $r_k$，在 anchor prefix 测度下优化一个局部 token objective。若想把结论推广到完整 sequence-level objective，必须额外控制 prefix occupancy drift。现有 TeX 中的 sequence experiment 已经显示，未访问 prefix 的误差不会因其他 prefix 的 inner 收敛而自动消失；这也是 LLM OPD 与独立 token toy 的主要差别。

## 可直接采用的分析路线

建议把 OPD 的证明分成三层。第一层在每个固定 prefix 上证明 Gibbs target 和 projective contraction；第二层在固定 reference 的参数空间证明 Natural-GD/普通 GD 的 inner 逼近误差；第三层再用 occupancy mismatch、teacher approximation 和采样误差把局部误差汇总到 sequence-level。第一层已经是精确结论，第二层需要 Fisher 非退化或阻尼后的谱界，第三层才需要 RL 特有的 concentrability 或 prefix-occupancy 假设。

实现上，outer loop 应显式保存 `anchor_state_dict`，所有 inner steps 读取同一份 $pi_k$ 的 log-prob；inner 结束后再同步。记录以下三个量足以诊断动力学：当前策略到精确 mirror target 的 KL、projective log-ratio 振幅 $\Delta_k$、以及 inner 梯度预条件后的残差。只看 teacher KL 无法区分“target 本身在移动”和“inner 没有求解好”这两种情况。

### 参考文献

[^mdpo]: Manan Tomar, Lior Shani, Yonathan Efroni, and Mohammad Ghavamzadeh. *Mirror Descent Policy Optimization*. arXiv 2005.09814, 2020. [论文](https://arxiv.org/abs/2005.09814)
[^mirror]: Amir Beck and Marc Teboulle. *Mirror Descent and Nonlinear Projected Subgradient Methods for Convex Optimization*. Operations Research Letters 31(3), 2003. [DOI](https://doi.org/10.1016/S0167-6377(02)00231-6)
[^pmd]: Guanghui Lan. *Policy Mirror Descent for Reinforcement Learning: Linear Convergence, New Sampling Complexity, and Generalized Problem Classes*. arXiv 2102.00135, 2021. [论文](https://arxiv.org/abs/2102.00135)
[^gpmd]: Wenhao Zhan, Shicong Cen, Baihe Huang, Yuxin Chen, Jason D. Lee, and Yuejie Chi. *Policy Mirror Descent for Regularized Reinforcement Learning: A Generalized Framework with Linear Convergence*. arXiv 2105.11066, 2021. [论文](https://arxiv.org/abs/2105.11066)
[^bgpo]: Feihu Huang, Shangqian Gao, and Heng Huang. *Bregman Gradient Policy Optimization*. ICLR 2022. [论文](https://arxiv.org/abs/2106.12112)
[^npg]: Sajad Khodadadian, Prakirt Raj Jhunjhunwala, Sushil Mahavir Varma, and Siva Theja Maguluri. *On the Linear Convergence of Natural Policy Gradient Algorithm*. arXiv 2105.01424, 2021. [论文](https://arxiv.org/abs/2105.01424)
[^amari]: Shun-ichi Amari. *Natural Gradient Works Efficiently in Learning*. Neural Computation 10(2), 1998. [DOI](https://doi.org/10.1162/089976698300017746)
