# AGENTS.md

## 项目范围与研究纪律

本仓库维护 MirrorOPD 的理论文档、KL 估计实验以及基于 TA-OPD/verl 的训练改动。公式推导必须写清测度、冻结锚点与停止梯度；Mean/BPO 的冻结 rollout 权重与 MDPO 的重要性比率保持分别处理。研究结果必须保留数据来源、模型、采样温度、误差定义与复现命令。

实际训练沿用上游框架，不独立重造 trainer。`upstream/` 是本机依赖 checkout，不进入主仓库；可复现改动通过固定上游版本及补丁保存。训练大文件、模型缓存与 scratch 放磁盘目录，archbox 的 `/tmp` 是容量受限的 tmpfs。

## 代码、记录与文档归属

- `src/mirroropd/` 保存可复用数学参考内核；`tests/` 保存正确性与集成验证。
- `training/` 保存正式训练入口；`patches/` 保存固定上游版本上的源码改动。
- `experiments/` 只保存实验程序，按 rollout、scoring、analysis、plotting、toy 分组。
- prefix/prompt 原始记录进入 `data/`，统计表、图与分析 metadata 进入 `results/`，报告进入 `docs/experiments/`。
- 日志、checkpoint、scratch 和临时验证输出进入忽略的 `runs/` 或远端运行目录，禁止写入程序目录。
- 移动文件时更新导入、文档相对链接、Makefile、CI、数据 manifest 与 LaTeX 图像路径；通过原数据 SHA-256 和集成测试核对。

## 格式化工具链

`git commit` 经过 `.githooks/pre-commit` 的 20 MiB 文件守卫与伪代码检查，再调用 pre-commit。formatter 经 git-format-staged 改写暂存内容并同步工作区；check 类 hook 失败才拦。初始化运行 `npm ci`、`brew install autocorrect shfmt shellcheck ruff gitleaks actionlint taplo` 与 `git config core.hooksPath .githooks`。版本以 `package.json` 为唯一事实源。

- Python：Ruff format 与 Ruff check；规则全开，研究脚本适用的豁免逐条写在 `ruff.toml`。
- Bash：shfmt 与 shellcheck；zsh 根据 shebang 透传。
- Markdown：markdownlint 原地修复，再接 autocorrect 与 prettier；原地修复后重新暂存。
- YAML：prettier 与散文注释显示宽度检查。
- JSON 配置与 TOML：prettier 与 taplo；实验 JSON 数据保持原始记录。
- GitHub Actions：actionlint；凭据检查：gitleaks。

CI 与本地检查同源。批量格式化独立提交，提交 hash 记入 `.git-blame-ignore-revs`。

## Agent 入口约定

`AGENTS.md` 与 `.agents/` 是入库的唯一事实源；skill 本体位于 `.agents/skills/`，各带 `agents/openai.yaml` 元数据。`.claude/skills/` 是指向 `.agents/skills/` 的软链，属本机便利层；clone 后运行 `scripts/agent-links.sh` 重建。预览链接运行 `scripts/crossnote-links.sh` 重建，工作区需要信任后才能注入预览 runtime。

## 提交与协作

使用英文 Conventional Commits，一个提交对应一个逻辑变更。保留其他人的未提交修改；并行修改按文件范围分工。文档和实验数据原地维护，不建立 v2/v3 副本。
