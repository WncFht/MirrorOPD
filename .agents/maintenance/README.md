# 维护体系来源与适配

维护材料从同级 `agent-maintenance` 仓库的提交 `20bb1e5` 引入，原始方法来自 DeepSeek Harness 固定提交 `571902028a75c64c1d767f4ae5fd099485f0ddf0`。中文规则经过提取、裁剪与重组；Python 检查工具由维护包新增。原始来源见 DeepSeek Harness[^source]，引入的版权和许可完整保存在 [LICENSE](LICENSE)。

## 引入范围

引入范围是三个 skill、六份共享参考、七份模板、Agent Notes 生命周期与分层指令、文档与归档检查、暂存快照、调度器和行为测试。维护包自身的过程决策未迁入 MirrorOPD。

## 项目适配

MirrorOPD 复用既有 Makefile、`.githooks`、格式工具和 pytest。setup 保留既有 hooksPath；维护测试纳入 `tests/maintenance/`；CI 使用项目的 Python 3.12；解释器沿用调度器所在环境。共享参考中的执行入口适配到[维护指南](../../docs/maintenance.md)。理论来源与冻结记录退出工程文档格式化，旧文档的延期按接入时文件 SHA-256 限定。

来源版本用于后续比较更新，不从同级目录动态加载技能或检查器；普通 checkout 已包含全部维护资产。执行边界、首次归档基线与未启用的发布能力见维护指南。

### 参考文献

[^source]: DeepSeek AI. DeepSeek Harness. GitHub 2026. [固定版本源码与许可证](https://github.com/deepseek-ai/deepseek-harness/tree/571902028a75c64c1d767f4ae5fd099485f0ddf0).
