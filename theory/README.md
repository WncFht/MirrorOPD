# MirrorOPD 理论推导

[核心推导](core/README.md)是长期维护的 ElegantBook 文档。正文从 Mirror Descent 推导目标教师和 J，附录维护完整证明、估计与误差分析，并比较 TOP-D 的目标教师退化。

| 目录 | 材料 |
| --- | --- |
| [core](core/README.md) | 核心推导、八组证明附录、来源对应与数学核验 |
| [alpha-divergence](alpha-divergence/README.md) | 密度比、α 散度及蒸馏梯度的中文重构稿 |
| [alpha-beta-gamma-zh](alpha-beta-gamma-zh/README.md) | Cichocki–Amari 散度族论文的中文译文与英文恢复稿 |

在仓库根执行 `make book` 或 `make theory-core`，同时构建完整 PDF 与正文阅读版。模板位于 `core/elegantbook.cls`，构建不依赖旧目录。

四组旧书稿、笔记和论文稿的独有数学内容已按主题补入附录，目录按用户要求删除。历史来源以 [source-map.csv](core/source-map.csv) 中的固定 Git 版本、路径及校验和定位；历史模型实验与配置不纳入核心稿。
