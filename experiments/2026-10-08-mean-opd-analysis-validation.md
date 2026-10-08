# PMD-Mean OPD 推导的数值核验

本次修订统一固定 anchor、student、teacher、prefix 测度、log-ratio、三个 residual、KL 方向、同步与采样计数。解析结果写在 `chapters/04_mean_opd_analysis.tex`，数值 oracle 使用未缩放平方损失的归一化乘子。

## 核验设计与结果

`test_opd_mean_analysis.py` 固定随机种子 20261008，枚举词表维度 2、4、8，镜像系数 0.1、0.5、1、2，每组八个独立正概率 anchor/teacher，共 96 组。检查归一化乘子上下界和 excess-loss 精确恒等式；在系数不超过 1 的组中检查 teacher KL 严格下降及 log-ratio 跨度收缩。全部通过。

三 token、三个独立 anchor action 的 leave-one-out 例子穷举全部 27 种样本结果，核对期望损失附加项及期望梯度。两式均通过。固定非二元 OPD reward，系数 10、50、200 时，最优分布与正中心化 log-ratio 极限分布的最大概率误差依次为 0.00399807、0.000821297、0.000206444，符合渐近结论。

`test_mean_bpo.py` 的 48 组随机 case 检查解析梯度、有限差分、Mean/BPO 损失与梯度恒等式、Lambert-W KKT 和归一化、Gibbs 零损失，以及有限样本 BPO 校正与离行为 pair weighting。全部通过。`test_kl_estimators.py` 的 KL 值绝对误差为 5.555e-5，梯度 L2 误差为 1.529e-3。

## 复现

在仓库根执行：

```sh
uv run --with torch --with numpy python experiments/test_opd_mean_analysis.py
uv run --with torch --with numpy python experiments/test_mean_bpo.py
uv run --with torch --with numpy python experiments/test_kl_estimators.py
```

这些检查使用有限 categorical 分布，验证解析公式和实现一致性。神经网络共享参数、全 prefix 一致界、有限优化与采样误差仍需满足正文列出的附加条件。
