# MirrorOPD

MirrorOPD 的独立研究与实现仓库。理论文档、KL 估计量实验、数值验证、上游训练补丁与训练入口在这里共同维护。仓库从 Obsidian 拆出，保留了 104 个项目历史提交；Obsidian 不再跟踪项目文件。

## 文件组织

| 路径                | 内容                                                        |
| ------------------- | ----------------------------------------------------------- |
| `src/mirroropd/`    | 可复用的数学参考实现与 CPU 模型验证内核                     |
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

**项目主文档与理论唯一入口**是 [固定测度 Token 级 MirrorOPD](docs/book/2026-10-08-固定测度Token级MirrorOPD.tex)，对应 [PDF](docs/book/2026-10-08-固定测度Token级MirrorOPD.pdf)。第二章的 OPD 展开和完整 Algorithm 5 位于 [02_opd_instantiation.tex](docs/book/chapters/02_opd_instantiation.tex)。正文讨论固定 anchor、固定 history 的 inner 目标、有限 inner 与 outer 同步动力学，以及四类 Top-k KL 的偏差、方差与实测结果。

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

正式训练使用 **TA-OPD/verl** 调度，MirrorOPD 算法源码由本仓库直接维护。核心 loss 位于 [training/losses.py](src/mirroropd/training/losses.py)，冻结缓存位于 [training/cache.py](src/mirroropd/training/cache.py)，prediction-position 与 nested-tensor 适配位于 [integrations/verl/fsdp.py](src/mirroropd/integrations/verl/fsdp.py)。上游只保留导入、薄包装和框架接入；outer rollout 与权重版本检查仍在 `upstream/TA-OPD/verl/verl/trainer/ppo/ray_trainer.py`。本仓库的 [训练入口](training/run_upstream_mirror_training.sh) 调用该框架，没有另外实现分布式训练器。

archbox 上已验证的运行目录是 `~/src/mirroropd-taopd`，虚拟环境、checkpoint、日志和 scratch 都在该目录。此次本机 Git 拆分保留这个运行目录。支持 `mirror_pmd_mean`、`mirror_bpo`、`mirror_mdpo`：

```bash
MIRROR_UPSTREAM_ROOT="$HOME/src/mirroropd-taopd" \
  bash training/run_upstream_mirror_training.sh \
  mirror_bpo /absolute/path/train.parquet /absolute/path/val.parquet \
  trainer.total_training_steps=2
```

已跑过 Qwen3-0.6B student / Qwen3-1.7B teacher 的 LoRA GPU 验证，包含两个 outer step、每轮两个 inner epoch、硬同步权重版本检查与 checkpoint 恢复。这些验证证明训练链路能够运行；正式 GSM8K 效果比较尚未完成。更长 GSM8K prompt/response 必须在入口中覆盖长度、token budget 和训练步数，默认值是 12 GB GPU 验证配置。环境与已完成运行见 [训练说明](docs/experiments/2026-10-08-%E4%B8%8A%E6%B8%B8%E8%AE%AD%E7%BB%83%E8%BF%90%E8%A1%8C%E8%AF%B4%E6%98%8E.md) 和 [GPU 验证记录](results/training/2026-10-08-upstream-gpu-validation.json)。

实验统计图与汇总 CSV 保存在 `results/`，prefix/prompt 原始记录保存在 `data/`。已有的三份大数据文件以 gzip 压缩后版本化，`make data` 根据 SHA-256 校验并恢复原文件；本地原文件保留并由 Git 忽略。训练日志和 checkpoint 只保留在本地，对应 metadata、生成脚本、图和汇总结果继续跟踪。迁移范围和数据校验记录见 [独立仓库迁移](docs/operations/2026-10-08-独立仓库迁移.md)。

目录边界、输入输出和复现命令见 [实验组织说明](docs/operations/2026-10-08-实验目录组织与训练代码导航.md)。运行需要参考内核的独立脚本时，在仓库根添加 `PYTHONPATH=src`；`make test` 已设置该路径。也可以用 `python -m pip install -e . --no-deps` 安装本地包。

仓库维护规则见 [AGENTS.md](AGENTS.md) 和 [维护规范](docs/operations/2026-10-08-仓库维护规范.md)。主文档固定维护在现有 TeX 路径；新推导、算法和实验结论更新到对应章节，PDF 由源文件编译。
