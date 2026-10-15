# MirrorOPD 核心推导

使用 ElegantBook 的中文理论稿。正文从带 κ、η 的加权散度问题出发，连接目标教师、教师目标 J、内层拟合、参数性质与有限拟合边界。八组附录整理完整证明、优化几何、信息与估计、条件中心、动力学和 TOP-D 目标教师的退化。

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
