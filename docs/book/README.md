# 固定测度 Token 级 MirrorOPD

本目录给出固定 anchor、固定 token history 下的 token-level 分析。PMD-Part、未线性化 BPO 与 MDPO 的总体解是 categorical mirror objective 的 Gibbs 分布；PMD-Mean 按原论文仅中心化 reward，用均值近似对数配分函数，其总体解是 Lambert-$W$ 分布。

主文档是 [`2026-10-08-固定测度Token级MirrorOPD.tex`](2026-10-08-固定测度Token级MirrorOPD.tex)，编译后的 PDF 同目录提供。文档包含：

- exact mirror target (p_k^+\propto\mu_k^{1-η}T^η) 及 η=1 的 teacher 极限；
- 四种 inner 方法的目标、梯度、Fisher 几何和固定测度采样语义；
- K4 reverse-KL 估计器、importance correction、Top-k 截断和方差界；
- hard-sync outer recursion、有限 inner 误差的几何累积界、EMA 和 history-measure drift；
- 可枚举 categorical toy experiment。

运行小实验：

```bash
uv run --with torch python experiments/toy_token_inner.py
uv run --with torch python experiments/test_kl_estimators.py
uv run --with torch python experiments/test_mean_bpo.py
```

第二个脚本用 200,000 个样本同时检查 K4 的 value 和 autograd gradient；当前误差分别为 (5.6\times10^{-5}) 和 (1.5\times10^{-3})。这里暂不进行大模型 sweep；正式 LLM 实验必须先通过文档第五章列出的 oracle tests。

Mean/BPO 的修订依据、完整恒等式和数值核验见 [2026-10-08-PMD-Mean与BPO修订核验](2026-10-08-PMD-Mean与BPO修订核验.md)。第二章的 OPD 实例与 Algorithm 5 位于 `chapters/02_opd_instantiation.tex`，由第二章主源文件加载。
