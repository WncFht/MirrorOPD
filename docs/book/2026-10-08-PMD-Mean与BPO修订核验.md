# PMD-Mean 与 BPO 修订核验

## 原始目标与近似目标

修订依据是 [2602.05933 的本地中文 LaTeX 源](../pmd-theory/2602.05933_zh_CN/main_arxiv.tex) 中的 $\mathcal L_{\mathrm{mean}}$ 定义与附录约束优化。固定一个 history，令 $q$ 为 anchor，$p$ 为 student，$f$ 为冻结信号，$\eta>0$。原始 mirror 问题为

$$
\max_p\;\mathbb E_p[f]-\eta^{-1}\operatorname{KL}(p\Vert q).
$$

其解满足 $U^*=\eta f-\log Z$，其中 $U=\log(p/q)$，$Z=\mathbb E_q[e^{\eta f}]$。原论文用 $\tau=\eta^{-1}$；本文平方损失与原论文相差正的整体尺度，乘子相应换算，最优分布一致。

原推导把 student log-ratio 也做了条件中心化，改变了 Mean 的目标，并将它变成缩放后的 BPO。按原论文只中心化 reward，令 $G=\eta(f-\mathbb E_qf)$，正确损失为

$$
L_{\mathrm{Mean}}=\tfrac12\mathbb E_q[(U-G)^2],\qquad
L_{\mathrm{BPO}}=\tfrac1{2\eta}\mathbb E_q[(G-U-K)^2],\qquad
K=\operatorname{KL}(q\Vert p)=-\mathbb E_qU.
$$

由 $\mathbb E_q[G-U-K]=0$，展开平方得到

$$
L_{\mathrm{Mean}}=\eta L_{\mathrm{BPO}}+\tfrac12K^2,\qquad
\nabla L_{\mathrm{Mean}}=\eta\nabla L_{\mathrm{BPO}}+K\nabla K.
$$

初始化 $p=q$ 时方向一致，尺度相差 $\eta$；离开 anchor 后有额外梯度项。冻结 anchor 下的平均 score 通常非零，不能把 $K\nabla K$ 当作 baseline 抵消。

Mean 的约束是 $\mathbb E_q[e^U]=1$。拉格朗日一阶条件及解为

$$
U-G+\lambda e^U=0,\qquad
U=G-W_0(\lambda e^G),\qquad
\mathbb E_q[e^U]=1.
$$

对一阶条件求 anchor 均值得 $\lambda=K$。正文给出了归一化根存在唯一性、拉格朗日函数严格凸性及全局最优性的完整证明。非恒定信号下，该解与 Gibbs 解不同。

## OPD 梯度与 Algorithm 5

记 $r=\log(T/q)$、$\Delta=\log(p/q)$、$K^T=\operatorname{KL}(q\Vert T)$、$K^\theta=\operatorname{KL}(q\Vert p)$。损失最小化的实际梯度中，Mean 与 BPO 的 score 权重分别为

$$
w_{\mathrm{Mean}}=\Delta-\eta(r+K^T),\qquad
w_{\mathrm{BPO}}=\eta^{-1}(\Delta+K^\theta)-r-K^T.
$$

Algorithm 5 统一使用 $\theta\gets\theta-\alpha G$。BPO 的负号与 $\eta^{-1}$ 已乘入权重，Mean 分支删除 student KL。原始损失 autograd 保留 BPO student center 的梯度；解析 score 实现则停止整个权重的梯度。

## 有限样本与数值核验

独立主动作及 $m$ 个独立 center 动作构造的 BPO 平方 residual，其期望为 $(1+1/m)\operatorname{Var}_q(\eta f-U)$。直接对该平方反向传播需乘 $m/(m+1)$；leave-one-out 使用 $m-1$ 个 center 动作时乘 $(m-1)/m$。Mean reward center 不依赖 student，其采样误差只给期望平方损失增加参数无关项。

[test_mean_bpo.py](experiments/test_mean_bpo.py) 检查了 48 组随机问题：词表维数为 2、4、8，$\eta$ 为 0.1、0.5、1、2。解析梯度、逐坐标有限差分、损失及梯度恒等式、Mean 归一化/KKT/驻点、Part/BPO 的 Gibbs 零损失、初始化方向尺度、恒定信号、完整枚举的有限样本修正和 off-behavior 双样本权重全部通过。

二动作例子 $q=(0.5,0.5)$、$T=(0.75,0.25)$、$\eta=1$ 的 Mean 解为 $(0.7291869550174757,0.2708130449825245)$，乘子为 $0.117928665874$；Gibbs 解为 teacher。该反例排除了 Mean 在 $\eta=1$ 时一步精确恢复 teacher 的旧结论。

[toy_token_inner.csv](experiments/toy_token_inner.csv) 记录修订后的 400 步实验。$\eta=0.8$ 时，Mean 到自身 Lambert-W 解的 TV 误差为 $4.11\times10^{-5}$，Lambert-W 与 Gibbs 解之间的 TV 为 $0.1904$。BPO 在相同学习率和预算下到 Gibbs 解的 TV 为 $4.53\times10^{-3}$。实验表区分优化误差与总体目标差异。

第二章拆分为通用推导与 [OPD 实例](chapters/02_opd_instantiation.tex)，原章节源文件加载后者，章节及算法编号连续。第三章修正 Mean/BPO 的测度与 KL 方向；第四章区分 Gibbs 与 Lambert-W outer 更新，并用 log-ratio 振幅推导有限 inner 的几何累积界。

复现命令（在本目录运行）：

```bash
uv run --with torch --with numpy python experiments/test_mean_bpo.py
uv run --with torch --with numpy python experiments/toy_token_inner.py
xelatex -interaction=nonstopmode -halt-on-error 2026-10-08-固定测度Token级MirrorOPD.tex
xelatex -interaction=nonstopmode -halt-on-error 2026-10-08-固定测度Token级MirrorOPD.tex
```
