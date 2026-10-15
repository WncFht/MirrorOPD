# MirrorOPD

MirrorOPD 0.2 提供 on-policy、MDPO、BPO 的幂均值蒸馏内核、严格配置的实验运行器及固定 verl 后端接入。当前 API 不提供旧模式、缓存与 launcher 的兼容层。

## 使用

使用 Python3.12 独立环境安装本包；GPU 软件栈还需核对 PyTorch、vLLM 与驱动的兼容性：

```bash
python -m pip install -e '.[data,test,math]'
python -m mirroropd --version
```

[运行与复现](docs/运行与复现.md)给出后端准备、模型/数据要求、CLI 配置、训练及审计步骤。[算法与实现](docs/算法与实现.md)解释当前三算法和估计器；[日志与评测](docs/日志与评测.md)定义结果口径。完整导航见[文档入口](docs/README.md)。

## 开发与研究

[架构](docs/架构.md)说明第一方包与后端边界，[开发维护](docs/maintenance.md)说明测试、skills、hooks 和 CI/CD。[0.2 交付说明](docs/2026-10-15-MirrorOPD全面重构说明.md)保存版本验证证据与限制。

理论推导来源在[theory](theory/README.md)，仍待用户 review 和合并。实验报告按研究问题维护，在[文档入口](docs/README.md)列明范围；原始证据及统计在[data](data)和[results](results)。历史源码恢复见[维护指南](docs/maintenance.md)，旧实验不自动代表当前接口已通过验证。
