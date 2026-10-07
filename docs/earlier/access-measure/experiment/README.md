# MirrorOPD 四种 inner 的最小实验

`mirroropd_algorithms.py` 提取了四种 token-level inner 的核心估计量：

- `pmd_part`: 回归精确 `h_k = eta * r_k - log Z_k`；
- `pmd_mean`: 回归中心化 reward `eta * (r_k - E_anchor[r_k])`；
- `bpo`: 对 residual 与 score 同时做 prefix 内中心化；
- `mdpo`: 对应 MD-OPD 仓库的 sampled-token KL 形式，不显式乘 action IS；`mdpo_is` 才是带 `rho = pi_theta / pi_anchor` 的理论无偏 action-level 对照。

`run_toy_experiment.py` 使用 `S` 个独立 prefix 和 `A` 个 token。每个 outer step 冻结 anchor，采样一批 anchor actions，执行固定数量的 inner updates，然后进入下一个 outer step。实验输出：

- `KL(student || teacher)`：outer 效果；
- `KL(student || mirror target)`：inner 是否跟上当前 target；
- inner coefficient variance：四种估计量的随机系数方差。

运行：

```bash
cd /Users/fanghaotian/src/obsidian/科研/pmd-theory/MirrorOPD/experiment
python3 run_toy_experiment.py --out results.json
```

这是一个验证估计量和 outer/inner 时间尺度的 toy 实验，不等同于真实 LLM 训练 benchmark。四种方法共享 teacher、anchor、samples、outer steps 和 inner steps，结果只用于检查理论中的梯度几何与方差趋势。

## Sequence experiment

`run_sequence_experiment.py` 使用真正的 prefix tree：每个 outer step 从冻结 anchor 采样完整长度为 `T` 的轨迹，inner 只更新 rollout 访问到的 prefix，因此可以同时测量 token KL 和 prefix occupancy drift。它默认使用 `V=6,T=6,N=512`，并将 BPO 的 complement-probability 权重、cap 和 mask 写入实际更新；同时报告 `mdpo`（官方 sampled-KL）与 `mdpo_is`（显式 action-IS oracle）。

```bash
python run_sequence_experiment.py --out sequence_results.json
```

该实验仍是可控的 categorical OPD 环境，但已经包含序列长度、访问测度偏移、未访问 prefix 和 inner budget 四个真实训练中缺失的因素。它不能替代 Qwen 等模型上的 benchmark，却能直接验证“prefix IS 是否需要、sampled-KL 与 action-IS 的差异、inner 是否追上每个 outer 的 mirror target”。
