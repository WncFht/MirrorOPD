# 开发维护指南

MirrorOPD 将文档、文字与长期决策维护接入现有工具链。三个技能、共享参考和模板保存在 `.agents/`。产品行为仍由数值、运行层和后端集成测试验证。

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

提交使用英文 Conventional Commits，每个提交只处理一个逻辑变化，代码、必要测试、后端补丁和说明一起维护。精确暂存自己的文件，保留他人未提交工作。纯格式整理单独提交，必要时将提交加入 `.git-blame-ignore-revs`，不以绕过 hook 代替修复检查失败。

pre-commit 保留原有 20 MiB 文件限制及本机可用时的伪代码检查，再运行格式、lint、secrets 和暂存契约。Markdown fixer 可能要求重新暂存；其他 formatter 通过 git-format-staged 操作暂存内容。全文件 pre-commit 检查可能修复格式，应审阅差异后再提交。

暂存契约将完整 index 导出到 `runs/maintenance-index-*` 隔离目录，检查实际提交的链接和 Notes，再与本地 HEAD 中的封存历史比较。工作区中未暂存的修复不能掩盖提交错误。快照使用项目磁盘目录，退出后清理，不将版本化研究材料复制到受限 `/tmp`。

pre-push 执行 `make check-maintenance`。该检查适合本地推送前执行；完整产品测试和格式检查由开发者按变更选择，并在 CI 全量运行。setup 只配置本仓库，未安装全局 hooks。

## Agent Notes

[Agent Notes](../.agents/notes/README.md) 保存代码和当前文档不能充分解释的长期理由、真实替代方案、代价与验证。三个 skill 的调用范围见[根指令](../AGENTS.md)；不是每项机械修改都要创建记录。

新记录依照 proposed、implemented、rejected 的状态维护；只对符合规则的已实施记录执行归档，归档正文和附件逐项封存。[单仓交付与固定后端](../.agents/notes/implemented/architecture/2026-10-12-单仓交付与固定后端.md)保存当前职责、真实备选和历史理由缺口。运行说明、实验报告和未实施历史提案有各自归属，不因篇幅长就整体转换为 Notes。

封存哈希用于检测当前字节变化，可信 Git 基线用于拦截正文和当前哈希同时重写。本地基线是 HEAD；CI 使用 PR base 或 push before。PR 的 archive-history job 从目标分支取出检查器，避免仅用待审版本证明自身保护有效。

初次接入时，目标分支尚无检查器，该 job 要求当前没有封存记录与清单。工具链合入后再开始封存；已有封存记录后，不能退回初始化路径。远端 required checks 与分支保护须在托管平台核验，工作流文件本身不证明合并保护已生效。

## 理论来源与旧文档边界

[theory](../theory/README.md) 保存从 `docs/book/`、`docs/book-old/`、`docs/Notes/`、`docs/paper/` 迁出的推导来源。正文、PDF 和内部层级继续纳入 Git；目录迁移只调整路径引用与构建入口。用户的 review 与稿件合并仍待后续启动。`theory/Notes/` 与工程 `.agents/notes/` 分开维护。

理论来源、第三方材料、运行目录和冻结 Notes 退出工程文档扫描及自动格式化。原始数据、结果与文献检索证据不参与新增文档和间距检查；当前文档指向这些产物的链接仍须有效。Markdown 检查仍遍历自有隐藏规则目录，检查文件目标是否存在及是否越出仓库；不验证外部网站、锚点、HTML 链接、命令行为或自然语言正确性。

旧工程文档与历史决定按主题逐项核验。[延期清单](../scripts/maintenance-deferred-docs.json) 仅登记接入时 `docs/operations/`、`docs/experiments/`、`docs/earlier/` 中尚未核验的既有 Markdown 及 SHA-256。仅字节相同的旧文件暂缓新增链接和间距检查，输出明确报告延期数量。新增文件或正文改变后进入检查；清单不能豁免根规则、skills 或新增维护文档。已完成主题整理的文件退出清单，其他旧记录仅修复迁移入链时保留其延期身份，并在工作记录说明机械变更范围。

现有格式与 lint 配置继续运行；理论来源和冻结 Notes 被统一排除。文档的当前或历史身份需在其入口说明，不用检查延期替代状态判断。

## 变更验证

修改前核对所属源码、当前测试和有效决定。行为变化同步接口注释、使用说明与生成来源，先改来源再重建派生物。维护脚本的正常输入和违规反例都要证明实际返回状态，产品与后端变更则验证对应执行路径。

| 变更                       | 必要证据                                                                                                 |
| -------------------------- | -------------------------------------------------------------------------------------------------------- |
| 当前工程文档、导航和 Notes | 按展示方式执行相关命令；链接、Notes、归档与格式检查；审阅完整 diff                                       |
| 核心目标、缓存、词表和适配 | 独立目标梯度、缓存生命周期、padding/EOS、预算末 action、全局 token 归约；随后按风险做真实 GPU 小规模验证 |
| 后端控制流或版本           | 从固定基线重放完整补丁，执行原生合同测试，再验证受影响的 FSDP/vLLM 实际链路                              |
| 性能                       | 相同输入、范围、预热和测量规则下的基线及新结果，包含时间、内存和数值误差                                 |
| 理论稿或书稿图片           | 保留条件、完整推导和来源，运行相关数学核验与 `make book`，同步源文件和 PDF                               |
| 数据或结果迁移             | 文件 SHA-256、数值内容、配置、生成程序、图片和所有引用一致                                               |

原生集成测试从 `MIRROR_TEST_BACKEND` 指向的固定后端加载源码，不能用任意旧 checkout 的存在证明兼容。CPU 测试、模型链路检查、真实 GPU 验证和正式任务比较分别报告；小样本结果不能替代完整任务成绩。测试通过后只为新变化、失败或未解决的问题扩大验证。

上游行为需要修改时，在干净基线上导出相对 [backend.json](../src/mirroropd/integrations/verl/backend.json) 的全部必要改动，并与第一方适配、测试和说明共同维护。`make upstream` 的准备入口不能代替最终运行前版本核验，忽略目录中的唯一修改也不算交付。当前产品结构见[架构](架构.md)，机器与输入迁移见[运行与复现](运行与复现.md)。

## 实验证据与数据

实验开始前记录待验证命题、成功与否决条件、输入规模和比较预算。数据与 split、prompt 来源、筛选、排除规则、实际样本数及训练测试重叠应可追溯。模型与 tokenizer 的实际内容、模板、EOS、采样温度、支持集、截断、种子、outer/inner、更新次数和同步顺序随运行保存，不能只留一个模型名称或 commit。

估计量写明随机性来自 prefix、rollout action、额外采样还是跨实验重复；区分条件于固定前缀的误差和前缀间差异。bias、variance、RMSE、KL 方向、reward 中心、聚合分母和单位按实际对象命名。Normalized、Truncated、EMA、Top-J 与 tail 等名称不能代替支持集、归一化和测度定义。公式与当前实现发生差异时，先说明条件和近似，再修正文档或实现。

原始证据在 `data/`，统计、图及 metadata 在 `results/`，程序在 `experiments/` 或 `scripts/` 的对应职责下维护。数值图由真实数据与程序生成，保留轴、单位、样本数和统计方法，不手改图中数字。重绘时同步报告和书稿引用，理论材料仍服从用户 review 边界。

压缩证据由 `data-manifest.json` 记录原始路径、压缩文件、大小与 SHA-256。[恢复工具](../scripts/restore-data.py)拒绝覆盖校验不同的现有文件；新增证据时验证压缩恢复与原始记录一致。checkpoint、完整训练日志和 scratch 留在忽略的运行目录，支持结论的正式证据另行归档。

跨机器运行记录实际导入路径、执行源码、后端、模型、数据、命令和环境；本地提交或单个 launcher 的复制不证明远端已执行该版本。大型下载、模型、checkpoint 和 scratch 放磁盘目录，archbox 的受限 `/tmp` 不承载这些文件。

当前操作、模块关系、决策理由、版本证据和理论来源的归属见[文档入口](README.md)。过程记录放在忽略的 `runs/`，不新增平行进展报告。独立报告按日期与主题命名，长期入口保持稳定；理论文件的完整性和实验总报告的单文件例外由分层 `AGENTS.md` 约束。

## CI 与构建产物

CI 保留 CPU 数学及运行层测试，新增维护行为测试、文档契约、可信归档基线和 wheel 安装验证。wheel 在源码目录外安装并运行 CLI plan，检查内置 profiles、后端补丁、判分来源和许可证，构建产物上传到当前 workflow 的 artifact。

维护包没有部署或发布实现，本仓也未配置自动部署、包索引上传或 GitHub Release。构建与 artifact 保存不等于发布；若以后需要发布，应明确版本来源、目标、凭据和触发方式后再接入。

## 来源与许可

维护材料引自 `agent-maintenance` 提交 `20bb1e5`，原始方法来自 DeepSeek Harness 固定提交 `571902028a75c64c1d767f4ae5fd099485f0ddf0`[^harness]。中文规则经过提取、裁剪与重组，Python 检查工具由维护包新增；原始版权与许可保存在 [.agents/LICENSE](../.agents/LICENSE)。MirrorOPD 的适配方式与执行边界在本指南维护，不另设维护体系介绍目录。

### 参考文献

[^harness]: DeepSeek AI. DeepSeek Harness. GitHub 2026. [固定版本源码与许可证](https://github.com/deepseek-ai/deepseek-harness/tree/571902028a75c64c1d767f4ae5fd099485f0ddf0).
