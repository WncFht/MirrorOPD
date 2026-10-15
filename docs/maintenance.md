# 开发维护指南

MirrorOPD 将文档、文字与长期决策维护接入现有工具链。三个技能、共享参考和模板保存在 `.agents/`。产品行为仍由数值、运行层和后端集成测试验证。

`paper-search-reports/`、`paper-search-logs/`和`environments/`是本地检索与环境记录，均由 Git 忽略，不作为 checkout 所需文件。复现依赖以包配置、固定后端及已保存的运行证据核验。

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
| `make book`              | 构建 `theory/core/` 的完整稿与正文阅读版                   |

通过 `PYTHON=/path/to/python make …` 指定解释器。调度器中的 `python`、`python3` 命令使用同一解释器；其他命令从当前 PATH 查找。原生集成测试需要准备固定后端，再执行 `MIRROR_TEST_BACKEND=upstream/TA-OPD-v2 make test`；工具接入不启动 GPU 训练。

`python scripts/run_checks.py --config scripts/checks.json --mode ci` 运行完整维护检查，包括间距、Shell 和 Actions 检查；它不代替产品测试。命令使用参数数组，不经 shell；依赖失败使后代 skipped，阻断项 skipped 同样失败。调度器只限制直接子进程的数量与超时，后台进程由所属测试负责清理。

## 提交与推送

pre-commit 保留原有 20 MiB 文件限制及本机可用时的伪代码检查，再运行格式、lint、secrets 和暂存契约。Markdown fixer 可能要求重新暂存；其他 formatter 通过 git-format-staged 操作暂存内容。全文件 pre-commit 检查可能修复格式，应审阅差异后再提交。

暂存契约将完整 index 导出到 `runs/maintenance-index-*` 隔离目录，检查实际提交的链接和 Notes，再与本地 HEAD 中的封存历史比较。工作区中未暂存的修复不能掩盖提交错误。快照使用项目磁盘目录，退出后清理，不将版本化研究材料复制到受限 `/tmp`。

pre-push 执行 `make check-maintenance`。该检查适合本地推送前执行；完整产品测试和格式检查由开发者按变更选择，并在 CI 全量运行。setup 只配置本仓库，未安装全局 hooks。

## Agent Notes

[Agent Notes](../.agents/notes/README.md) 保存代码和当前文档不能充分解释的长期理由、真实替代方案、代价与验证。三个 skill 的调用范围见[根指令](../AGENTS.md)；不是每项机械修改都要创建记录。

文档整理只将有长期指导价值且有依据的决策理由提取为 Notes，不把实施事实、实验进度或未执行建议批量转换为决策。新记录依照 proposed、implemented、rejected 的状态维护；只对符合规则的已实施记录执行归档，归档正文和附件逐项封存。

封存哈希用于检测当前字节变化，可信 Git 基线用于拦截正文和当前哈希同时重写。本地基线是 HEAD；CI 使用 PR base 或 push before。PR 的 archive-history job 从目标分支取出检查器，避免仅用待审版本证明自身保护有效。

初次接入时，目标分支尚无检查器，该 job 要求当前没有封存记录与清单。工具链合入后再开始封存；已有封存记录后，不能退回初始化路径。远端 required checks 与分支保护须在托管平台核验，工作流文件本身不证明合并保护已生效。

## 理论来源与旧文档边界

[theory](../theory/README.md) 的长期维护稿位于 `core/`。旧书稿、理论笔记和论文稿已按用户要求提取数学内容、合并并删除；核心稿的来源表记录固定 Git 版本、原路径及校验和。理论推导与工程 `.agents/notes/` 分开维护，历史实验不作为核心稿的内容或配置来源。

理论来源、第三方材料、运行目录和冻结 Notes 退出工程文档扫描及自动格式化。原始数据、结果与文献检索证据不参与新增文档和间距检查；当前文档指向这些产物的链接仍须有效。Markdown 检查仍遍历自有隐藏规则目录，检查文件目标是否存在及是否越出仓库；不验证外部网站、锚点、HTML 链接、命令行为或自然语言正确性。

历史工程文档已按主题重新整理，接入时的[延期清单](../scripts/maintenance-deferred-docs.json)清空。当前工程 Markdown 全部接受链接和间距检查；理论源稿与冻结第三方正文仍保留自己的审定边界。格式检查不能识别过时结论、错误归因或无信息段落，内容审核仍需阅读实现与证据。

## 实验报告的日常维护

按[报告维护流程](../.agents/references/experiment-reports.md)确定问题、协议和证据，必要时使用[骨架](../.agents/templates/experiment-report.md)。同一问题更新原报告，执行流水留`runs/`；数据与图表分别在`data/`、`results/`，对应生成程序和校验信息必须可定位。

报告修订同时核对实际样本数、版本、计量单位、统计权重和未完成条件。推翻旧解释时改正结论，保留纠正依据及必要负结果。当前操作、历史重跑和只读汇总复算分别标注，不把当前 CPU 测试写成历史 GPU 重跑。

## 历史源码与证据恢复

0.2 退役的 launcher 和分析脚本保存在[固定源码归档](../data/refactor-20261015/legacy-source.tar.gz)，另有 Git 标签`archive/pre-refactor-20261015`。标签是完整旧仓库入口，tar 仅保存 manifest 中列出的退役源码，两者不等同于某一历史 run 实际执行的全部字节。先核对 run 的 provenance，再选择对应 source 快照与后端。

在仓库根执行下列只读检查和隔离恢复：

```bash
python - <<'PYCODE'
from pathlib import Path
import hashlib
import gzip
import json
import tarfile

root = Path.cwd()
archive = root / "data/refactor-20261015/legacy-source.tar.gz"
manifest = json.loads((root / "data-manifest.json").read_text())
record = next(row for row in manifest["files"] if row["archive"] == str(archive.relative_to(root)))
with gzip.open(archive, "rb") as stream:
    restored = stream.read()
assert len(restored) == record["bytes"]
assert hashlib.sha256(restored).hexdigest() == record["sha256"]
destination = root / "runs/legacy-source-review"
destination.mkdir(parents=True, exist_ok=False)
with tarfile.open(archive) as stream:
    stream.extractall(destination, filter="data")
print(destination)
PYCODE
```

恢复目录必须尚不存在。`filter="data"`要求支持该安全解压过滤器的 Python；项目基线为 Python3.12。`make data`按 manifest 恢复压缩原始记录，并拒绝覆盖校验不一致的文件。tar 成员引用如`legacy-source.tar.gz#training/...`只是定位信息；先解压才能执行文件。

恢复源码不等于恢复运行环境或输入。旧模型、缓存、checkpoint、后端修改和外部数据依各 run 记录提供；缺失时只能复算现有汇总，不宣称完整复现。旧程序可能依赖旧包与邻仓，将其放在独立历史环境，避免与当前可安装包混用。

## CI 与构建产物

CI 保留 CPU 数学及运行层测试，新增维护行为测试、文档契约、可信归档基线和 wheel 安装验证。wheel 在源码目录外安装并运行 CLI plan，检查内置 profiles、后端补丁、判分来源和许可证，构建产物上传到当前 workflow 的 artifact。

维护包没有部署或发布实现，本仓也未配置自动部署、包索引上传或 GitHub Release。构建与 artifact 保存不等于发布；若以后需要发布，应明确版本来源、目标、凭据和触发方式后再接入。

## 来源与许可

维护材料引自 `agent-maintenance` 提交 `20bb1e5`，原始方法来自 DeepSeek Harness 固定提交 `571902028a75c64c1d767f4ae5fd099485f0ddf0`[^harness]。中文规则经过提取、裁剪与重组，Python 检查工具由维护包新增；原始版权与许可保存在 [.agents/LICENSE](../.agents/LICENSE)。MirrorOPD 的适配方式与执行边界在本指南维护，不另设维护体系介绍目录。

### 参考文献

[^harness]: DeepSeek AI. DeepSeek Harness. GitHub 2026. [固定版本源码与许可证](https://github.com/deepseek-ai/deepseek-harness/tree/571902028a75c64c1d767f4ae5fd099485f0ddf0).
