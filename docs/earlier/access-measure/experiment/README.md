# MirrorOPD 四种 inner 的最小实验

`mirroropd_algorithms.py` 提取了四种 token-level inner 的核心估计量：

- `pmd_part`: 回归精确 `h_k = eta * r_k - log Z_k`；
- `pmd_mean`: 回归中心化 reward `eta * (r_k - E_anchor[r_k])`；
- `bpo`: 对 residual 与 score 同时做 prefix 内中心化；
- `mdpo`: 使用 `rho = pi_theta / pi_anchor` 的直接 mirror-ascent 梯度。

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
