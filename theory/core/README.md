# MirrorOPD 核心推导

使用 ElegantBook 的中文理论稿。前三章从 Mirror Descent 子问题出发，定义冻结信号 f 与 advantage，推导加权教师 m 的显式形式，再反推出教师目标 J。后三章讨论内层拟合、参数性质与有限拟合边界。八组附录先陈述每项定理、命题或引理的条件与结论，再展开证明，并比较 TOP-D 的目标教师退化及相关性质。

- [完整 PDF](2026-10-15-MirrorOPD核心推导.pdf)
- [正文阅读版](2026-10-15-MirrorOPD核心推导-正文.pdf)
- [LaTeX 主源](2026-10-15-MirrorOPD核心推导.tex)

完整稿与正文阅读版使用同一份正文。附录的内容与来源对应见完整稿末尾，原五组理论来源保留。

在仓库根执行：

```bash
make theory-core
python -m pytest -q tests/reference/test_core_monograph.py
```

也可在本目录运行 `make`。需要 XeLaTeX、latexmk、Biber 及现有书稿所需字体和宏包；构建通过 `TEXINPUTS` 复用 `../book/elegantbook.cls`。正文阅读版保留完整稿的附录编号，详细证明可按编号查阅完整 PDF。`make clean` 清理辅助文件并保留 PDF。
