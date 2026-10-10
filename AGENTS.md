# AGENTS.md

## 项目范围与首要文档

本仓库维护 MirrorOPD 理论、KL 估计实验和基于 TA-OPD/verl 的训练实现。最重要的理论文档固定为 `docs/book/2026-10-08-固定测度Token级MirrorOPD.tex`，章节由它组织，PDF 从源文件编译生成。继续修改这份主稿及其章节，不因日期、修订轮次另建主稿或 v2/v3 副本。本约定覆盖通用的正文更新后自动改名规则。

理论结论以主稿中写清假设、定义和逐步证明的版本为准；实际算法行为以被测试的源码为准。两者不一致时明确差异并核对推导与实现，不能隐藏矛盾。保留完整推导，不把修订变成摘要。`docs/book-old/` 完整保存 2026-10-10 重构前书稿，作为只读历史来源；`docs/earlier/` 保存早期工程，当前理论维护入口是 `docs/book/`。

完整维护规范见 `docs/operations/2026-10-08-仓库维护规范.md`。

## 目录职责与唯一维护位置

- `src/mirroropd/training/`：生产训练的 Mirror 缓存和损失内核，在主仓库直接维护。
- `src/mirroropd/integrations/verl/`：verl 张量布局与接口适配；框架参数显式传入。
- `src/mirroropd/` 其他模块：数学参考实现与研究内核，明确参考和生产职责，测试一致性。
- `tests/`：数学恒等式、估计量、缓存生命周期、outer/inner 和模型集成验证。
- `training/`：正式训练与 GPU canary 启动入口，不在脚本中复制损失公式。
- `experiments/`：实验程序，按 rollout、scoring、analysis、plotting、toy 分组；复用核心包。
- `data/`：prompt、trajectory、prefix 等输入和原始记录；`data-manifest.json` 保存恢复及校验信息。
- `results/`：可引用的统计表、图、分析 metadata 和训练验证记录；报告进入 `docs/experiments/`。
- `docs/book/`：当前主稿、章节及同步 PDF；`docs/book-old/`：重构前完整书稿；`docs/operations/`：维护与运行规范；`docs/earlier/`：早期材料。
- `scripts/`：初始化、工具链和仓库维护；科学实验程序不放这里。
- `upstream/`：Git 忽略的第三方 checkout；`upstream-manifest.json` 固定版本，`patches/` 保存必要的集成改动。核心算法不得仅留在忽略目录或补丁里。
- `runs/`：忽略的日志、checkpoint、scratch 和临时输出。模型和大文件不放程序目录。

移动文件时同步更新导入、启动命令、文档链接、Makefile、CI、manifest 和 LaTeX 图片路径。校验原始数据 SHA-256，保留无关的未提交工作。

## 理论、训练与实验证据

公式明确采样测度、KL 方向、冻结锚点、停止梯度、outer/inner 索引与同步时点。Mean/BPO 的冻结 rollout 权重与 MDPO 的重要性比率分别处理；KL 数值估计与 KL 梯度估计分别定义。Normalized、Truncated、EMA、Tail-aware 估计量明确支持集、归一化和修正。

生产核心保持可在 CPU 独立导入；Ray、CUDA、verl 耦合放在适配层和上游集成中。保留第三方来源和许可证，不整体复制 trainer。修改上游接口后按固定基线更新补丁，在干净基线上重放验证。检查数学语义、停止梯度、缓存冻结和 hard sync，再验证模型链路。

实验先记录问题、判定标准和配置。结果记录数据与 split、模型版本、温度、top-p/top-k、长度限制、种子、样本数、估计量、误差定义、代码版本和复现命令。CPU 验证、GPU canary、完整训练和正式 benchmark 分别报告；截断 rollout 和小样本结果不能冒充完整 GSM8K 效果。

三机源码目录统一为 `~/src/mirroropd` 和 `~/src/MD-OPD`，后续开发与训练在 archbox 直接进行，运行产物留在 archbox。MirrorOPD 现有 TA-OPD 后端环境保留在 `~/src/mirroropd-taopd`。本机提交不等于远端部署；同步核心包、集成补丁及启动入口后记录远端实际版本。模型缓存和大 scratch 放磁盘，禁止把容量受限的 `/tmp` 当实验数据盘。路径与保留状态见 `docs/operations/2026-10-10-三机仓库路径与同步.md`。

## 检查与工具链

新 checkout 运行 `make bootstrap`；需要上游运行 `make upstream`；数据恢复运行 `make data`。训练语义修改运行 `make test`，书稿或图像引用修改运行 `make book`，提交前运行 `make check`。PDF 与对应源文件共同提交，核对引用、编号和图片。

提交经过 `.githooks/pre-commit` 的 20 MiB 文件守卫和伪代码检查，再调用 pre-commit。formatter 经 git-format-staged 同步暂存内容与工作区；检查失败须修复。依赖版本由仓库配置和锁文件确定，不能绕过检查提交。

- Python：Ruff format/check；研究豁免逐条写在 `ruff.toml`。
- Shell：shfmt/shellcheck，zsh 根据 shebang 透传。
- Markdown：markdownlint、autocorrect/prettier；YAML：prettier 与注释宽度检查。
- 配置 JSON/TOML：prettier/taplo；原始实验数据保持数值内容与校验关系。
- GitHub Actions：actionlint；凭据：gitleaks；CI 与本地检查同源。

## Agent 入口与提交

`AGENTS.md` 与 `.agents/` 是入库的唯一事实源。`.claude/skills/` 是忽略的本机软链，通过 `scripts/agent-links.sh` 重建；预览链接通过 `scripts/crossnote-links.sh` 重建，信任工作区后启用 runtime。

使用英文 Conventional Commits，一个提交对应一个逻辑变化。保留他人的未提交修改，并行修改按文件范围分工。纯格式化独立提交，hash 记入 `.git-blame-ignore-revs`。研究证据、数据和正文原地维护；新实验条件以清楚的配置标识区分，不复制整套工程。
