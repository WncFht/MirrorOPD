# MirrorOPD

MirrorOPD 0.2 提供 on-policy、MDPO、BPO 的幂均值蒸馏内核、可复用实验运行器和固定版本的 verl 接入。当前接口不提供 0.1 的模式别名、旧缓存或旧 launcher 兼容层。

## 安装与验证

使用 Python 3.12，在独立环境安装：

```bash
python -m pip install -e '.[data,test,math]'
make upstream
MIRROR_TEST_BACKEND=upstream/TA-OPD-v2 make test
make check
```

GPU 训练还需要匹配的 CUDA、vLLM、FlashAttention 和 Ray 环境。`environments/` 保存已验证环境的依赖与 wheel 校验信息；安装本包不自动替换 GPU 软件栈。

## 配置与运行

模型对、数据集、训练配方分别位于包内 `profiles/models`、`profiles/datasets`、`profiles/recipes`。可以使用内置名称或自己的 JSON 文件。变量必须显式提供，路径不依赖父目录 shell 配置。

```bash
mirroropd plan --recipe math-four-gpu --models deepseek-justrl --dataset dapo-math \
  --var model_root=/path/to/models --var data_root=/path/to/data --algorithm on_policy

mirroropd run --recipe math-four-gpu --models deepseek-justrl --dataset dapo-math \
  --var model_root=/path/to/models --var data_root=/path/to/data \
  --algorithm on_policy --set training.outer_steps=2 --set evaluation.question_limit=4 \
  --backend upstream/TA-OPD-v2 --output runs/my-canary

mirroropd status runs/my-canary
mirroropd audit runs/my-canary
```

`run --prepare-only` 完成输入核验、源码快照和后端配置生成，不启动 GPU 训练。`suite` 使用同样配置，依次执行三种算法。on-policy 强制每个 fresh batch 一次更新；MDPO/BPO 的 inner 数由配方指定。

数学配方默认 κ=0.5、η=0.2，训练 batch=64、response=7168；测试每题采样 16 次、temperature=0.7、top-p=0.9、response=31744。主要评测指标是每题样本正确率的平均值，另记录 pass@16。采样 seed 按 phase、权重版本、题目 UID 和 sample index 独立确定。

## 代码与证据

- `core/`：纯张量 reward、缓存、条件目标和诊断；不依赖 Ray/verl。
- `runtime/`：严格配置、数据准备、指纹、源码快照、GPU 预约、进程管理、事件、metrics 和审计。
- `integrations/verl/`：张量布局适配、后端编译器、固定基线及补丁。
- `scoring/`：独立判分器及带来源、许可证的数学 grader。
- `reference/`：用于独立检验的数学参照，不是生产训练入口。
- `tests/`：统一 pytest 收集的数学、模型、缓存、运行与集成验证。

运行目录保存 `job.json`、`inputs.json`、`backend.yaml`、`run.json`、冻结源码、分来源事件流、GPU 采样、逐题评测和最终验证结果。W&B 是可选展示，本地事件和原始回答是完成审计的依据。

理论推导来源保存在 [theory](theory/README.md)，包含尚待 review 与合并的书稿、理论笔记和论文稿。历史实验报告和数据保留；旧源码通过 Git 标签 `archive/pre-refactor-20261015` 或 [源码归档](data/refactor-20261015/legacy-source.tar.gz) 恢复，旧报告中的命令不代表当前 API。
