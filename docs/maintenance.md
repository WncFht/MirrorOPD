# 开发维护指南

MirrorOPD 将文档、文字与长期决策维护接入现有工具链。三个技能、共享参考和模板保存在 `.agents/`，来源与许可见[引入说明](../.agents/maintenance/README.md)。产品行为仍由数值、运行层和后端集成测试验证。

## 本地准备

需要 Python 3.12、Node 22 和 Git。使用独立的开发环境安装维护依赖，再配置当前仓库入口：

```bash
python -m pip install -r requirements-dev.txt
npm ci
python scripts/setup.py
```

此外需要 AutoCorrect、Gitleaks、shfmt、ShellCheck、actionlint 和 Taplo。现有系统包管理器可提供这些工具；CI 的安装方式见[工作流](../.github/workflows/ci.yml)。维护环境不代替训练所需的 PyTorch、CUDA、vLLM、Ray 和 FlashAttention。

setup 复用 `.githooks` 并创建忽略的 `.claude/skills` 链接，重复运行不会改写已有 hook。外来 hooksPath 或技能入口会导致失败，保留原内容。它不调用 `pre-commit install`，因为该命令与已配置的 hooksPath 冲突。可选的预览链接仍由 `scripts/crossnote-links.sh` 配置，需要本机存在相应事实源；基本维护检查不依赖该预览环境。

## 使用入口

| 命令                     | 验证范围                                                   |
| ------------------------ | ---------------------------------------------------------- |
| `make check-docs`        | 当前 Markdown 本地链接、skill 元数据、Notes 结构和归档历史 |
| `make test-maintenance`  | 维护检查的有效与违规输入、暂存隔离、安装边界               |
| `make check-maintenance` | 文档契约与维护测试，供 pre-push 与 CI 复用                 |
| `make check`             | 文档契约和现有格式、lint、secrets、全文件 pre-commit 检查  |
| `make test`              | pytest 收集的维护、数学、运行层与集成测试                  |
| `make book`              | 构建 `theory/book/` 中保留的分章书稿                       |

通过 `PYTHON=/path/to/python make …` 指定解释器。调度器中的 `python`、`python3` 命令使用同一解释器；其他命令从当前 PATH 查找。原生集成测试需要准备固定后端，再执行 `MIRROR_TEST_BACKEND=upstream/TA-OPD-v2 make test`；工具接入不启动 GPU 训练。

`python scripts/run_checks.py --config scripts/checks.json --mode ci` 运行完整维护检查，包括间距、Shell 和 Actions 检查；它不代替产品测试。命令使用参数数组，不经 shell；依赖失败使后代 skipped，阻断项 skipped 同样失败。调度器只限制直接子进程的数量与超时，后台进程由所属测试负责清理。

## 提交与推送

pre-commit 保留原有 20 MiB 文件限制及本机可用时的伪代码检查，再运行格式、lint、secrets 和暂存契约。Markdown fixer 可能要求重新暂存；其他 formatter 通过 git-format-staged 操作暂存内容。全文件 pre-commit 检查可能修复格式，应审阅差异后再提交。

暂存契约将完整 index 导出到 `runs/maintenance-index-*` 隔离目录，检查实际提交的链接和 Notes，再与本地 HEAD 中的封存历史比较。工作区中未暂存的修复不能掩盖提交错误。快照使用项目磁盘目录，退出后清理，不将版本化研究材料复制到受限 `/tmp`。

pre-push 执行 `make check-maintenance`。该检查适合本地推送前执行；完整产品测试和格式检查由开发者按变更选择，并在 CI 全量运行。setup 只配置本仓库，未安装全局 hooks。

## Agent Notes

[Agent Notes](../.agents/notes/README.md) 保存代码和当前文档不能充分解释的长期理由、真实替代方案、代价与验证。三个 skill 的调用范围见[根指令](../AGENTS.md)；不是每项机械修改都要创建记录。

接入时只建立生命周期规则和目录入口，没有重写历史记录或将旧报告转换为 Notes。新记录依照 proposed、implemented、rejected 的状态维护；只对符合规则的已实施记录执行归档，归档正文和附件逐项封存。

封存哈希用于检测当前字节变化，可信 Git 基线用于拦截正文和当前哈希同时重写。本地基线是 HEAD；CI 使用 PR base 或 push before。PR 的 archive-history job 从目标分支取出检查器，避免仅用待审版本证明自身保护有效。

初次接入时，目标分支尚无检查器，该 job 要求当前没有封存记录与清单。工具链合入后再开始封存；已有封存记录后，不能退回初始化路径。远端 required checks 与分支保护须在托管平台核验，工作流文件本身不证明合并保护已生效。

## 理论来源与旧文档边界

[theory](../theory/README.md) 保存从 `docs/book/`、`docs/book-old/`、`docs/Notes/`、`docs/paper/` 迁出的推导来源。正文、PDF 和内部层级继续纳入 Git；目录迁移只调整路径引用与构建入口。用户的 review 与稿件合并仍待后续启动。`theory/Notes/` 与工程 `.agents/notes/` 分开维护。

理论来源、第三方材料、运行目录和冻结 Notes 退出工程文档扫描及自动格式化。原始数据、结果与文献检索证据不参与新增文档和间距检查；当前文档指向这些产物的链接仍须有效。Markdown 检查仍遍历自有隐藏规则目录，检查文件目标是否存在及是否越出仓库；不验证外部网站、锚点、HTML 链接、命令行为或自然语言正确性。

旧工程文档正文与历史决策重写按用户要求延后。[延期清单](../scripts/maintenance-deferred-docs.json) 仅登记接入时 `docs/operations/`、`docs/experiments/`、`docs/earlier/` 中的既有 Markdown 及 SHA-256；必要的迁移链接修复完成后固定基线。仅字节相同的旧文件暂缓新增链接和间距检查，输出明确报告延期数量。新增文件或正文改变后进入检查；清单不能豁免根规则、skills 或新增维护文档。延期不表示旧文档已经核验，后续整理应修复问题并移除相应登记。

现有格式与 lint 配置继续运行；理论来源和冻结 Notes 被统一排除。旧文档中的旧接口、命令与结论不在接入任务中改写。

## CI 与构建产物

CI 保留 CPU 数学及运行层测试，新增维护行为测试、文档契约、可信归档基线和 wheel 安装验证。wheel 在源码目录外安装并运行 CLI plan，检查内置 profiles、后端补丁、判分来源和许可证，构建产物上传到当前 workflow 的 artifact。

维护包没有部署或发布实现，本仓也未配置自动部署、包索引上传或 GitHub Release。构建与 artifact 保存不等于发布；若以后需要发布，应明确版本来源、目标、凭据和触发方式后再接入。
