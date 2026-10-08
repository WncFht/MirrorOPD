# MirrorOPD

MirrorOPD 的独立研究与实现仓库。理论文档、KL 估计量实验、数值验证、上游训练补丁与训练入口在这里共同维护。仓库从 Obsidian 拆出，保留了 104 个项目历史提交；Obsidian 不再跟踪项目文件。

## 文件组织

| 路径 | 内容 |
| --- | --- |
| `docs/book/` | 当前 ElegantBook 文档、章节源文件与编译后的 PDF |
| `experiments/` | 数值验证、rollout/scoring/绘图脚本、实验报告与汇总数据 |
| `experiments/upstream_patches/` | 在原始训练项目上的可重放修改 |
| `upstream-manifest.json` | 上游 URL、基线提交、迁移时 HEAD 与补丁对应关系 |
| `upstream/` | 本地上游 checkout，Git 忽略；通过 bootstrap 重建 |
| `docs/earlier/` | 早期访问测度、统一理论工程与原始笔记，完整保留 |
| `scripts/` | 上游获取、格式化、提交检查与预览环境接线 |

当前正文是 [固定测度 Token 级 MirrorOPD](docs/book/2026-10-08-固定测度Token级MirrorOPD.tex)，对应 [PDF](docs/book/2026-10-08-固定测度Token级MirrorOPD.pdf)。第二章的 OPD 展开和完整 Algorithm 5 位于 [02_opd_instantiation.tex](docs/book/chapters/02_opd_instantiation.tex)。正文讨论固定 anchor、固定 history 的 inner 目标、有限 inner 与 outer 同步动力学，以及四类 Top-k KL 的偏差、方差与实测结果。

## 获取代码与检查

需要 Git、Node/npm、Python、uv、pre-commit。系统检查工具为 autocorrect、ruff、shfmt、shellcheck、gitleaks、actionlint、taplo；编译文档还需要 TeX Live 的 xelatex、latexmk 和中文字体。

```bash
make bootstrap
make upstream
make test
make book
make check
```

`make upstream` 按 manifest 获取 MDPO、OpenKimi、One-Shot-OPD 与 TA-OPD，并应用保存的补丁。已经存在的本地修改会保留；补丁冲突直接报错。上游源码的许可证仍属于对应项目，本仓库不把完整第三方源码重复提交。

`make test` 包括数学恒等式、outer/inner 动力学、上游函数语义、trainer cache 路由和小型 causal-LM 集成验证。上游 AST 测试执行原始函数体，CPU 测试与真实 GPU 训练的证据分别记录，详见 [目标完成核验](experiments/2026-10-08-目标完成核验.md)。

## 训练入口与当前证据

正式训练基于修改后的 **TA-OPD/verl**。核心实现是 `upstream/TA-OPD/verl/verl/trainer/distillation/fsdp/losses.py`，outer rollout、缓存和版本检查在 `upstream/TA-OPD/verl/verl/trainer/ppo/ray_trainer.py`。本仓库的 [训练入口](experiments/run_upstream_mirror_training.sh) 调用该框架，没有另外实现分布式训练器。

archbox 上已验证的运行目录是 `~/src/mirroropd-taopd`，虚拟环境、checkpoint、日志和 scratch 都在该目录。此次本机 Git 拆分保留这个运行目录。支持 `mirror_pmd_mean`、`mirror_bpo`、`mirror_mdpo`：

```bash
MIRROR_UPSTREAM_ROOT="$HOME/src/mirroropd-taopd" \
  bash experiments/run_upstream_mirror_training.sh \
  mirror_bpo /absolute/path/train.parquet /absolute/path/val.parquet \
  trainer.total_training_steps=2
```

已跑过 Qwen3-0.6B student / Qwen3-1.7B teacher 的 LoRA GPU 验证，包含两个 outer step、每轮两个 inner epoch、硬同步权重版本检查与 checkpoint 恢复。这些验证证明训练链路能够运行；正式 GSM8K 效果比较尚未完成。更长 GSM8K prompt/response 必须在入口中覆盖长度、token budget 和训练步数，默认值是 12 GB GPU 验证配置。环境与已完成运行见 [训练说明](experiments/2026-10-08-上游训练运行说明.md) 和 [GPU 验证记录](experiments/2026-10-08-upstream-gpu-validation.json)。

实验汇总图与 CSV 保留在 `experiments/`。原始轨迹、逐 prefix 大文件、日志和 checkpoint 保留在本地并由 Git 忽略，避免 Git 承载训练运行数据；对应 metadata、生成脚本、图和汇总结果继续跟踪。迁移范围和数据校验记录见 [独立仓库迁移](docs/operations/2026-10-08-独立仓库迁移.md)。
