# MirrorOPD

MirrorOPD 的独立研究与实现仓库。理论文档、KL 估计量实验、数值验证、上游训练补丁与训练入口在这里共同维护。仓库从 Obsidian 拆出，保留了 104 个项目历史提交；Obsidian 不再跟踪项目文件。

## 文件组织

| 路径                | 内容                                                        |
| ------------------- | ----------------------------------------------------------- |
| `src/mirroropd/`    | 生产训练内核、verl 适配与数学参考实现                       |
| `tests/`            | 数学恒等式、模型与上游集成测试                              |
| `training/`         | TA-OPD/verl 正式训练与 GPU canary 启动入口                  |
| `experiments/`      | 按 rollout、scoring、analysis、plotting、toy 分组的实验程序 |
| `data/`             | prefix/prompt 原始记录与可恢复压缩数据                      |
| `results/`          | 分实验保存的统计表、图、metadata 与 GPU 验证记录            |
| `docs/experiments/` | 实验设计、分析报告、实现审计和运行说明                      |
| `docs/book/`        | 当前 ElegantBook 源文件与 PDF                               |
| `docs/earlier/`     | 完整保留的早期工程与原始笔记                                |
| `patches/`          | 上游训练项目的可重放源码修改                                |
| `upstream/`         | 固定版本的本地第三方 checkout，由 manifest 重建，Git 忽略   |
| `runs/`             | 本机日志、构建检查与临时输出，Git 忽略                      |
| `scripts/`          | 仓库维护、数据恢复与工具链入口                              |

项目主文档与理论唯一入口是 [固定测度 Token 级 MirrorOPD](docs/book/2026-10-08-固定测度Token级MirrorOPD.tex)，对应 [PDF](docs/book/2026-10-08-固定测度Token级MirrorOPD.pdf)。八章从条件 KL 与负熵镜像步出发，依次推导归一化投影、梯度几何、信息接口、估计与自动微分、OPD 实例化、同步动力学和实验证据。实现与 outer--inner 伪代码由 [OPD 章](docs/book/chapters/06_opd.tex) 组织。重构前完整书稿保存在 [book-old](docs/book-old/README.md)。

## 获取代码与检查

需要 Git、Node/npm、Python、uv、pre-commit。系统检查工具为 autocorrect、ruff、shfmt、shellcheck、gitleaks、actionlint、taplo；编译文档还需要 TeX Live 的 xelatex、latexmk 和中文字体。

```bash
make bootstrap
make upstream
make data
make test
make book
make check
```

`make upstream` 按 manifest 获取 MDPO、OpenKimi、One-Shot-OPD 与 TA-OPD，并应用保存的补丁。已经存在的本地修改会保留；补丁冲突直接报错。上游源码的许可证仍属于对应项目，本仓库不把完整第三方源码重复提交。

`make test` 包括数学恒等式、outer/inner 动力学、上游函数语义、trainer cache 路由和小型 causal-LM 集成验证。测试直接导入本仓库生产内核，并通过 AST 执行上游薄包装和原始审计函数，CPU 测试与真实 GPU 训练的证据分别记录，详见 [目标完成核验](docs/experiments/2026-10-08-%E7%9B%AE%E6%A0%87%E5%AE%8C%E6%88%90%E6%A0%B8%E9%AA%8C.md)。

## 训练入口与当前证据

固定监督下的全局标量学习率分析、精确 BPO 推导和 GPU 0–3 验证见 [BPO 标量调度分析与验证](docs/experiments/2026-10-12-BPO标量调度分析与验证.md)。报告区分完整目标与 sampled/Top-k 更新场，并同时记录 backward、forward、耗时和后期反弹。

当前数学实验使用 DeepSeek-R1-Distill-Qwen-1.5B 学生和 JustRL-DeepSeek-1.5B 教师，在四张 H800 上比较 OPD、MDPO、BPO。设置、日志审计和图表见[数学三算法主报告](docs/experiments/2026-10-12-数学三算法日志审计与学习效率比较.md)，字段、事件轴和公式见[统一指标规范](docs/operations/2026-10-12-蒸馏统一日志与评测指标规范.md)。

代码分层、实际 GPU 环境清单、模型与数据要求、可移植 canary 和正式启动命令见[代码架构与跨机器复现](docs/operations/2026-10-12-代码架构与跨机器复现.md)。新默认数学评测已改为每题 16 次采样和独立长预算；旧正式比较采用单次 greedy。新默认配置尚未完成 GPU canary。`make bootstrap` 不安装 GPU 训练栈，数学模型和 parquet 需要单独准备。

正式训练使用 **TA-OPD/verl** 调度，MirrorOPD 算法源码由本仓库直接维护。核心 loss 位于 [training/losses.py](src/mirroropd/training/losses.py)，冻结缓存位于 [training/cache.py](src/mirroropd/training/cache.py)，prediction-position 与 nested-tensor 适配位于 [integrations/verl/fsdp.py](src/mirroropd/integrations/verl/fsdp.py)。上游只保留导入、薄包装和框架接入；outer rollout 与权重版本检查仍在 `upstream/TA-OPD/verl/verl/trainer/ppo/ray_trainer.py`。本仓库的 [训练入口](training/run_upstream_mirror_training.sh) 调用该框架，没有另外实现分布式训练器。

三机源码目录统一为 `~/src/mirroropd` 和 `~/src/MD-OPD`，后续开发与训练在 archbox 进行；运行产物留在 archbox。MirrorOPD 已验证的 TA-OPD 后端仍位于 `~/src/mirroropd-taopd`，保留其虚拟环境和运行依赖，第一方核心从 `~/src/mirroropd/src` 加载。路径、版本及本地修改见 [三机仓库路径与同步](docs/operations/2026-10-10-三机仓库路径与同步.md)。生产入口支持 `mirror_pmd_mean`、`mirror_bpo`、`mirror_mdpo`、`mirror_opd_onpolicy`、`mirror_opd_is` 和 `mirror_opd_is_no_baseline`；PMD-Part 尚未接入生产入口：

```bash
MIRROR_UPSTREAM_ROOT="$HOME/src/mirroropd-taopd" \
  bash training/run_upstream_mirror_training.sh \
  mirror_bpo /absolute/path/train.parquet /absolute/path/val.parquet \
  trainer.total_training_steps=2
```

已跑过 Qwen3-0.6B student / Qwen3-1.7B teacher 的 LoRA GPU 验证，包含两个 outer step、每轮两个 inner epoch、硬同步权重版本检查与 checkpoint 恢复。这些验证证明训练链路能够运行；正式 GSM8K 效果比较尚未完成。更长 GSM8K prompt/response 必须在入口中覆盖长度、token budget 和训练步数，默认值是 12 GB GPU 验证配置。环境与已完成运行见 [训练说明](docs/experiments/2026-10-08-%E4%B8%8A%E6%B8%B8%E8%AE%AD%E7%BB%83%E8%BF%90%E8%A1%8C%E8%AF%B4%E6%98%8E.md) 和 [GPU 验证记录](results/training/2026-10-08-upstream-gpu-validation.json)。

Pythia-14M student / Pythia-70M teacher 已完成两组全参数 MDPO 动力学实验：每组 512 条 WikiText prompt、32 outer × 4 inner。固定验证前缀的全词表 KL 与真实续写 NLL 揭示了对齐和语言质量的分歧，详见 [Pythia 实测报告](docs/experiments/2026-10-08-Pythia-MDPO训练设置与实测.md)。固定入口为 `training/run_pythia_mdpo.sh`，checkpoint 评价入口为 `training/score_pythia_mdpo.sh`；这两组运行的 checkpoint 在 archbox 的 `~/src/mirroropd/runs/`，完整可引用日志和评分归档在 `results/training/pythia-mdpo*/`。

实验统计图与汇总 CSV 保存在 `results/`，prefix/prompt 原始记录保存在 `data/`。已有的三份大数据文件以 gzip 压缩后版本化，`make data` 根据 SHA-256 校验并恢复原文件；本地原文件保留并由 Git 忽略。checkpoint 和临时训练日志留在忽略的运行目录；正式实验需要引用的日志、metadata、图和汇总结果归档后跟踪。迁移范围和数据校验记录见 [独立仓库迁移](docs/operations/2026-10-08-独立仓库迁移.md)。

目录边界、输入输出和复现命令见 [实验组织说明](docs/operations/2026-10-08-实验目录组织与训练代码导航.md)。运行需要参考内核的独立脚本时，在仓库根添加 `PYTHONPATH=src`；`make test` 已设置该路径。也可以用 `python -m pip install -e . --no-deps` 安装本地包。

仓库维护规则见 [AGENTS.md](AGENTS.md) 和 [维护规范](docs/operations/2026-10-08-仓库维护规范.md)。MD-OPD 的接入设计见 [MirrorOPD 接入 MD-OPD 代码审查](docs/operations/2026-10-10-MirrorOPD接入MD-OPD代码审查.md)，本次尚未实现该接入。主文档固定维护在现有 TeX 路径；新推导、算法和实验结论更新到对应章节，PDF 由源文件编译。
