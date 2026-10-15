# MirrorOPD 核心推导

使用 ElegantBook 的中文理论稿。前三章从 Mirror Descent 子问题出发，定义冻结信号 f 与 advantage，推导加权教师 m 的显式形式，再反推出教师目标 J。后三章讨论内层拟合、参数性质与有限拟合边界。八组附录先陈述每项定理、命题或引理的条件与结论，再展开证明，并比较 TOP-D 的目标教师退化及相关性质。

- [完整 PDF](2026-10-15-MirrorOPD核心推导.pdf)
- [正文阅读版](2026-10-15-MirrorOPD核心推导-正文.pdf)
- [LaTeX 主源](2026-10-15-MirrorOPD核心推导.tex)

完整稿与正文阅读版使用同一份正文。旧稿的数学内容按主题合并；来源通过 `source-map.csv` 中的固定 Git 版本和校验和追溯，补充主题的具体命题对应在 `coverage-additions.csv`。旧目录已删除，历史模型实验与配置不纳入本稿。

在仓库根执行：

```bash
make book
python scripts/book_source_audit.py
python -m pytest -q tests/reference/test_core_monograph.py tests/reference/test_core_extensions.py
```

也可在本目录运行 `make`。需要 XeLaTeX、latexmk、Biber 及相应字体和宏包；模板随稿维护于 `elegantbook.cls`，保留其 LPPL 1.3c 许可头。正文阅读版保留完整稿的附录编号，详细证明可按编号查阅完整 PDF。`make clean` 清理辅助文件并保留 PDF。
