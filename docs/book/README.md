# 固定测度 Token 级 MirrorOPD

本目录给出一个统一的 token-level 分析：在 outer step (k) 内固定 anchor μ_k、固定 token history (s=(x,y_{<t}))，PMD-Part、PMD-Mean、BPO、MDPO 都被定义为同一个 categorical mirror objective 的不同求解器或估计器。

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
```

第二个脚本用 200,000 个样本同时检查 K4 的 value 和 autograd gradient；当前误差分别为 (5.6\times10^{-5}) 和 (1.5\times10^{-3})。这里暂不进行大模型 sweep；正式 LLM 实验必须先通过文档第五章列出的 oracle tests。
