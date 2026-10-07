# 四种 inner 的 outer/inner toy 实验

实验使用 `S=8` 个独立 prefix、每个 prefix `A=32` 个 token。teacher logits 固定；每个 outer step 将当前 student 冻结为 anchor，从 anchor 为每个 prefix 采样 `K=64` 个 token，然后执行 `40` 次 inner update。四种方法共享随机数据、`eta=0.35`、inner learning rate `0.25`，不使用 BPO clip、rollout IS 或 prefix IS。指标在每个 outer step 后计算。

运行环境是仓库已有的 `科研/pmd-theory/offpolicy-correction/.venv`（Python + NumPy；绘图若安装 matplotlib 则同时输出 PNG）。

```bash
cd /Users/fanghaotian/src/obsidian/科研/pmd-theory/MirrorOPD/experiment
/Users/fanghaotian/src/obsidian/科研/pmd-theory/offpolicy-correction/.venv/bin/python run_toy_experiment.py --out results.json
```

五个 seed 的最终结果如下。`teacher KL` 是 `KL(student || teacher)`，`target KL` 是当前 outer mirror target 与 student 的距离。

| inner | teacher KL 均值 ± 标准差 | target KL 均值 ± 标准差 |
|---|---:|---:|
| PMD-Part | 0.308867 ± 0.061431 | 0.016525 |
| PMD-Mean | 0.312115 ± 0.067683 | 0.023614 |
| BPO | 0.322014 ± 0.070368 | 0.017046 |
| MDPO | 0.173842 ± 0.026912 | 0.008525 |

这个 toy 中 MDPO 的结果最好，因为它直接沿当前 candidate policy 的 mirror objective 做 ascent；PMD-Part 与 BPO 追踪 mirror target 的能力接近，PMD-Mean 因为只使用中心化 reward、没有显式 `log Z`，target gap 略大。这个排序不能直接外推到 LLM：真实工程中 MDPO 还承担 action ratio、prefix occupancy 和 sampled-token KL 的方差，而 PMD-Part 可以利用 teacher logits 与 Top-k KL 把这些方差降下来。

实验文件：

- `mirroropd_algorithms.py`：四种单步估计量的核心代码；
- `run_toy_experiment.py`：outer/inner 循环、指标和绘图；
- `code_reference.md`：从 `/Users/fanghaotian/src/MD-OPD` 代码摘录的 MDPO、MOPD residual、BPO-shaped、teacher reward 与 top-k 核心实现；
- `results.json`、`results_5seeds.json`、`results.png`：运行产物。
