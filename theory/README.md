# 理论推导来源

本目录保存 MirrorOPD 的理论推导来源，包括尚未 review 和合并的书稿、推导笔记与论文稿。它们继续纳入 Git，保留各版本的正文和来源；工程文档在 [docs](../docs/AGENTS.md)，工程决策在 [Agent Notes](../.agents/notes/README.md)。

| 目录                                                 | 现有材料                                            |
| ---------------------------------------------------- | --------------------------------------------------- |
| [core](core/README.md)                               | ElegantBook 核心推导、完整证明附录与 TOP-D 目标教师比较 |
| [book](book/README.md)                               | 分章书稿及对应 PDF，保留现有构建入口                |
| [book-old](book-old/README.md)                       | 旧书稿及其历史说明                                  |
| [Notes](Notes/)                                      | 理论专著稿及对应 PDF                                |
| [paper](paper/)                                      | 论文叙事稿及对应 PDF                                |
| [alpha-divergence](alpha-divergence/README.md)       | 密度比、α 散度及蒸馏梯度的中文重构稿                |
| [alpha-beta-gamma-zh](alpha-beta-gamma-zh/README.md) | Cichocki–Amari 散度族论文的完整中文译文与英文恢复稿 |

`book/`、`book-old/`、`Notes/` 和 `paper/` 从 `docs/` 原样迁入，仅调整路径引用与构建入口。正文的正确性、重复关系和合并取舍留待用户 review。目录迁移不改变各材料的审定状态。

在仓库根执行 `make book` 构建现有分章书稿。其他稿件维持原来的独立来源，尚未建立统一装配入口。
