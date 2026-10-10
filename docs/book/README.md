# 固定测度 Token 级 MirrorOPD

[LaTeX 主稿](2026-10-08-固定测度Token级MirrorOPD.tex) 与 [PDF](2026-10-08-固定测度Token级MirrorOPD.pdf) 是当前维护入口。主稿只负责样式、统一宏和八章装配，`chapters/` 负责章节入口，`sections/` 负责具体定义、证明、算法与证据。旧稿原样保存在 [book-old](../book-old/README.md)。

八章依次展开条件目标与镜像步、归一化投影、梯度与优化几何、信息接口、随机积分与自动微分、OPD 实例化、外层动力学和实验证据。固定锚点 `q`、当前学生 `p` 与教师 `t` 的角色贯穿全书；一般 KL 估计另用有序分布 `P_theta,Q`。

数学显示环境使用 `equation`、`align` 及对应无编号环境。独立源文件不超过 1000 行。完整内容迁移和新增结论见 [重构记录](2026-10-10-全书重构与内容迁移.md)。

从仓库根编译：

```bash
make book
```

新 CPU 数学核验和图的复现命令：

```bash
uv run --with torch --with numpy python experiments/toy/book_reconstruction_audit.py
uv run --with matplotlib --with numpy python experiments/plotting/book_reconstruction.py
```

实验说明见 [数学核验报告](../experiments/2026-10-10-书稿重构数学核验.md)，机器可读结果见 [metadata](../../results/book-reconstruction/metadata.json)。这些实验检验有限词表数学，不承担神经网络训练效果比较。
