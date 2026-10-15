# AGENTS.md — 理论推导

长期维护稿在 `core/`，使用本目录内的 ElegantBook 模板。正文保持 Mirror Descent 子问题、信号与 advantage、加权教师 m、目标 J、inner 拟合、参数性质的推导顺序。附录按数学主题组织，每项独立结论先写完整条件与命题，再写证明；来源重复时合并，不直接复制旧稿。

用户已授权提取数学内容后删除 `book/`、`book-old/`、`Notes/`、`paper/`。历史数学来源由 `core/source-map.csv` 的固定 Git 版本和校验和追溯，不恢复为并行维护稿。核心稿不纳入历史模型实验内容和配置。`alpha-divergence/` 与 `alpha-beta-gamma-zh/` 保留为独立文献材料。

理论正文不适用工程 Markdown 自动格式化。修改推导时核实条件、尺度、积分测度和停止梯度语义，数值测试不能替代证明。运行 `make book` 同步两个 PDF；有限单纯形检查由 `tests/reference/test_core_monograph.py` 和 `tests/reference/test_core_extensions.py` 收集。单个 TeX 文件不超过 1000 行，按主题通过 input 组装。
