# AGENTS.md

## 当前接口与维护边界

MirrorOPD 当前维护 0.2 接口。用户明确要求全面重构、清理旧内容，并尽量不提供兼容层。生产算法仅为 `on_policy`、`mdpo`、`bpo`；旧模式、旧缓存与实验专用 launcher 已退役。历史源码保存在 Git 标签 `archive/pre-refactor-20261015` 和可校验归档中，不恢复为当前入口。

目录职责：

- `src/mirroropd/core/`：框架独立的 signal、缓存、概率、目标和诊断。
- `src/mirroropd/runtime/`：配置、数据、来源、资源、进程、观测和审计。
- `src/mirroropd/integrations/verl/`：最小后端适配；`backend.json` 与 `backend.patch` 是唯一的后端版本事实源。
- `src/mirroropd/scoring/`：独立 scorer；vendor 保留来源与完整许可证。
- `src/mirroropd/reference/`：独立数学参照，不复制为生产接口或兼容别名。
- `src/mirroropd/profiles/`：可组合的模型、数据集、训练配方。
- `tests/`：所有验证均需被 pytest 收集，不把断言只放在脚本 main 中。
- `experiments/benchmarks/`：可复现的性能及实现验证程序。
- `data/`、`results/`：原始证据与统计；`data-manifest.json` 记录压缩恢复及 SHA-256。
- `runs/`：忽略的运行目录、快照、checkpoint 和 scratch。

不在脚本复制损失公式。配置只有一个解析、校验和默认值来源，拒绝未知字段与版本。后端只提供调度和必要调用点，不整体复制第三方 trainer。

## 数学与验证

理论推导维护在 `theory/core/2026-10-15-MirrorOPD核心推导.tex`，使用 ElegantBook，`make book` 构建完整稿与正文阅读版。旧书稿、理论笔记与论文稿已按用户要求提取数学内容并合并，来源由 Git 固定版本追溯。保持完整推导、假设和来源；修改理论源文件时同步其 PDF。核心稿不引用历史模型实验及配置。历史实验的有效结论、方法、负结果和数据保留；旧正文按用户授权逐篇重写、合并或删除，不能把保留证据解释为保留每篇文章。

公式明确冻结锚点、教师、当前学生、prefix/action 测度、停止梯度、更新和同步顺序。区分 score surrogate 与条件目标、reward 中心与普通 KL、训练 token 加权与评测按题平均。on-policy 每个新 batch 只做一次完整更新。

核心/适配修改先验证独立目标梯度、缓存生命周期、padding/EOS、预算末 action、全局 token 归约，再运行真实 GPU canary。小样本链路检查不能称为正式任务成绩。性能改动先留基线和输入，再报告相同范围的时间、内存和数值误差。

后端补丁从 `backend.json` 的固定基线导出全部必要改动，在干净基线上重放；保留版权和许可证。运行源码冻结并记录 byte hash，禁止仅凭 commit 名称描述未提交代码。

## 可观测性与资源

结构化事件使用当前 schema；记录 run/source/config 身份、phase、outer/inner/update、计量语义、有效 token、结束原因和错误。每个 producer 单独写流，rank0 只写已经全局归约的优化记录。缺样本、重复记录、非有限数或版本不一致应阻止完成判定。

GPU 列表、模型目录、数据路径、后端和运行目录显式配置。只清理经 PID/start-time 验证的本任务进程，禁止全局 ray stop 或清理其他训练。预约不因 heartbeat 过期而被窃取；借用预约须核对原 owner 仍为同一暂停进程。

大文件、checkpoint 和安装产物使用磁盘目录；archbox 的 /tmp 是受限 tmpfs。家目录配置和密钥遵守全局规则，不把环境密钥、.env、SSH 私钥或 netrc 内容放进 Git 或日志。

## 检查、文档与提交

文档维护使用 [maintain-docs](.agents/skills/maintain-docs/SKILL.md)，文字与注释使用 [maintain-prose](.agents/skills/maintain-prose/SKILL.md)，长期决策使用 [maintain-decision-notes](.agents/skills/maintain-decision-notes/SKILL.md)。先读对应区域指令：[文档](docs/AGENTS.md)、[理论来源](theory/AGENTS.md)、[脚本](scripts/AGENTS.md)、[CI](.github/AGENTS.md)、[Agent Notes](.agents/notes/AGENTS.md)。技能和引用材料以 `.agents/` 为事实源，本机链接不入库。

维护接入、检查边界及来源见[维护指南](docs/maintenance.md)。`make check-docs` 检查当前文档与 Notes，`make test-maintenance` 验证维护工具，`make check-maintenance` 供 pre-push 与 CI 执行；产品测试仍由 `make test` 负责。pre-commit 在框架保护未暂存内容之前先格式化 index，再检查真实暂存快照；不能用未暂存修复证明提交正确。使用 `scripts/setup.py` 保留两阶段顺序及所选维护 Python，不用普通 `pre-commit install` 覆盖 `.githooks`。

用户已启动旧文档与历史决策整理。实验报告按研究问题维护，规范见 docs/AGENTS.md；提取独有有效内容后删除被替代正文，不建立平行历史汇编。理论推导维护在 `theory/core/`，工程决策维护在 `.agents/notes/`。

使用英文 Conventional Commits，独立逻辑变更独立提交，不绕过 hook。保留用户已有的未提交工作。每项维护抽象与配置都需有当前消费者；失败后明确状态与清理责任，停止新工作并等待已启动工作结束后再释放依赖。默认运行 `make test`；原生源码集成测试通过 `MIRROR_TEST_BACKEND` 指定已准备的固定后端。提交前执行 `make check`。理论或书稿图片变化执行 `make book`。

按用户最新要求，全面重构只交付一份最终说明文档；不新增多个平行进展报告。过程记录放在忽略的运行目录。旧报告保留其历史语境，删除源码的链接指向精确历史归档。
