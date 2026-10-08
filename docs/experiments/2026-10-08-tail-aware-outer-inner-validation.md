# Tail-aware outer/inner 验证

## 已验证的范围

`opd_categorical.py` 实现冻结 anchor 的 outer/inner 循环。每个 outer step 从当前 anchor 采样，inner 从 anchor 初始化，teacher reward 为 `log teacher - log anchor`。Mean 使用 Tail-aware teacher KL center；BPO 同时使用 Tail-aware anchor–student center；MDPO 按当前 student 测度积分。中心与解析残差在 score-function surrogate 中停止梯度。hard sync 与 categorical 概率混合 soft sync 后重新采样。

`uv run --with torch python tests/test_opd_outer_inner.py` 通过 60 个随机解析梯度核对，以及 18 个覆盖三种方法、两种同步和三种支持集大小的 outer/inner 实验。完整支持集时，三种 surrogate 的梯度分别与原始 PMD-Mean、BPO、MDPO 损失一致。

四动作案例初始 teacher KL 为 0.917008。支持集大小为 2，8 个 outer step，每轮 40 个 inner step，学习率 0.08，mirror 参数 0.5。末端 inner student 的 teacher KL 如下。

| 方法     | Hard sync | Soft sync，概率混合系数 0.5 |
| -------- | --------: | --------------------------: |
| PMD-Mean |  0.015269 |                    0.112451 |
| BPO      |  0.000745 |                    0.022309 |
| MDPO     |  0.000620 |                    0.017448 |

这些是可枚举动作期望上的结果。rollout token 当前只验证 outer 采样契约，尚未用于 Monte Carlo inner 更新，不能称为真实 LLM 管线已经验证。

## 后续完成条件

1. 接入冻结 rollout action 的 Monte Carlo inner 路径，MDPO 保留 student/anchor importance ratio，比较大量重复采样的平均梯度与精确 oracle。
2. 接入 Transformers 训练和 vLLM rollout 后端，验证同一 tokenizer、temperature、支持集、模型权重同步和 token mask。
3. 验证缓存只存 sampled/head log-prob、tail mass 与 token ids，避免缓存完整词表；加入 checkpoint、恢复与有限梯度检查。
4. 审计 PMD、MDPO 原始实现并记录与本项目 fixed-measure OPD 的差别。
5. 完成本地小模型 outer/inner smoke test 后，再运行 archbox 真实模型实验。
