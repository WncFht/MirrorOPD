# MirrorOPD

MirrorOPD 提供 `on_policy`、`mdpo`、`bpo` 三种 token 级蒸馏算法、共享实验运行器和固定版本的 verl 接入。第一方内核维护损失与冻结缓存，后端负责 Ray、FSDP 和 vLLM 执行。当前 0.2 接口不接受旧算法别名、缓存或 launcher。

## 安装与配置检查

使用 Python 3.12，在独立环境安装本包及数据、测试、数学判分依赖：

```bash
python -m pip install -e '.[data,test,math]'
python -m mirroropd --version
```

GPU 训练另需匹配的 PyTorch、CUDA、vLLM、FlashAttention 和 Ray 环境；本包安装不替换完整 GPU 软件栈。环境记录、后端准备与输入要求见[运行与复现](docs/运行与复现.md)。

先解析一份配置，不读取模型权重或启动训练。示例路径只用于展示 profile 的展开规则：

```bash
mirroropd plan --recipe math-four-gpu --models deepseek-justrl --dataset dapo-math \
  --var model_root=/models --var data_root=/data --algorithm on_policy
```

`plan` 输出完整 Job 与 `config_id`，但不证明模型、数据或 GPU 可用。准备真实输入后，再按运行指南执行 `run --prepare-only`、小规模训练和完成审计；每一步的成功含义不同。

## 文档与研究材料

[文档入口](docs/README.md) 导航当前操作、架构、开发维护与验证证据。[架构说明](docs/架构.md) 解释模块职责和一次更新的生命周期；[开发维护指南](docs/maintenance.md) 说明检查与提交要求。版本切换和已保存的验证证据见 [0.2 全面重构说明](docs/2026-10-15-MirrorOPD全面重构说明.md)。

[theory](theory/README.md) 保存尚待 review 与合并的书稿、理论笔记和论文稿，继续纳入 Git。历史实验报告与数据保留各自运行条件；旧命令和成绩不能作为当前接口或新配置的执行结果。

重构前源码可通过 Git 标签 `archive/pre-refactor-20261015` 或[源码归档](data/refactor-20261015/legacy-source.tar.gz) 恢复。数据恢复、旧接口和历史提案的边界见文档入口。
