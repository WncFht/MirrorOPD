# 四种 inner 的 outer/inner toy 实验

实验使用 `S=8` 个独立 prefix、每个 prefix `A=32` 个 token。teacher logits 固定；每个 outer step 将当前 student 冻结为 anchor，从 anchor 为每个 prefix 采样 `K=64` 个 token，然后执行 `40` 次 inner update。四种方法共享随机数据、`eta=0.35`、inner learning rate `0.25`，不使用 BPO clip、rollout IS 或 prefix IS。指标在每个 outer step 后计算。

运行环境是仓库已有的 `科研/pmd-theory/offpolicy-correction/.venv`（Python + NumPy；绘图若安装 matplotlib 则同时输出 PNG）。

```bash
cd /Users/fanghaotian/src/mirroropd/docs/earlier/access-measure/experiment
/Users/fanghaotian/src/obsidian/科研/pmd-theory/offpolicy-correction/.venv/bin/python run_toy_experiment.py --out results.json
```

五个 seed 的最终结果如下。`teacher KL` 是 `KL(student || teacher)`，`target KL` 是当前 outer mirror target 与 student 的距离。

| inner           | teacher KL 均值 ± 标准差 | target KL 均值 ± 标准差 |
| --------------- | -----------------------: | ----------------------: |
| PMD-Part        |      0.308867 ± 0.061431 |                0.016525 |
| PMD-Mean        |      0.312115 ± 0.067683 |                0.023614 |
| BPO             |      0.322014 ± 0.070368 |                0.017046 |
| MDPO sampled-KL |      0.192158 ± 0.034002 |                0.009559 |

这个 toy 中 sampled-KL 版 MDPO 的结果最好，因为它直接沿当前 candidate policy 的 KL-regularized objective 做 sampled ascent；它仍然不是 action-IS 无偏估计。PMD-Part 与 BPO 追踪 mirror target 的能力接近，PMD-Mean 因为只使用中心化 reward、没有显式 `log Z`，target gap 略大。这个排序不能直接外推到 LLM：真实工程中 MDPO 还承担 sampled-token KL 与 prefix occupancy 的方差，而 PMD-Part 可以利用 teacher logits 与 Top-k KL 把这些方差降下来。

实验文件：

- `mirroropd_algorithms.py`：四种单步估计量的核心代码；
- `run_toy_experiment.py`：outer/inner 循环、指标和绘图；
- `code_reference.md`：从 `/Users/fanghaotian/src/MD-OPD` 代码摘录的 MDPO、MOPD residual、BPO-shaped、teacher reward 与 top-k 核心实现；
- `results.json`、`results_5seeds.json`、`results.png`：运行产物。

## Sequence-level prefix-tree experiment

为了检验 independent-row toy 遗漏的访问测度问题，新增 `run_sequence_experiment.py`。它在一个深度为 `T`、分支数为 `V` 的 prefix tree 上运行 outer/inner：每个 outer 冻结 anchor，从 root 采样完整轨迹；inner 只更新这些轨迹真正访问到的 prefix；下一次 outer 刷新 anchor。这样 student 和 anchor 的 prefix occupancy 会随策略变化，未访问 prefix 的误差不会被错误地当成已经训练好。

默认单 seed 配置为 `V=6,T=6,N=512,outer=10,inner=32,eta=.35,lr=.18`，结果为：

| inner                 | anchor-occupancy 下 `KL(student $\vert$ teacher)` | anchor-occupancy 下 `KL(student $\vert$ target)` | occupancy L1 |
| --------------------- | ------------------------------------------------: | -----------------------------------------------: | -----------: |
| PMD-Part              |                                          0.339804 |                                         0.030604 |     0.346673 |
| PMD-Mean              |                                          0.400846 |                                         0.038011 |     0.382680 |
| BPO practical         |                                          0.323232 |                                         0.035923 |     0.717551 |
| MDPO sampled-KL       |                                          0.300925 |                                         0.030693 |     0.425874 |
| MDPO action-IS oracle |                                          0.304841 |                                         0.031633 |     0.444186 |

为减少单次随机性的影响，另用 `V=6,T=5,N=256,outer=8,inner=24` 跑三个 seed：

| inner                 | `KL(student $\vert$ teacher)` 均值 ± 标准差 | `KL(student $\vert$ target)` 均值 ± 标准差 |
| --------------------- | ------------------------------------------: | -----------------------------------------: |
| PMD-Part              |                         0.448051 ± 0.034193 |                        0.036835 ± 0.002340 |
| PMD-Mean              |                         0.515765 ± 0.032573 |                        0.044312 ± 0.002008 |
| BPO practical         |                         0.366871 ± 0.018044 |                        0.034501 ± 0.002143 |
| MDPO sampled-KL       |                         0.362731 ± 0.013163 |                        0.034100 ± 0.001223 |
| MDPO action-IS oracle |                         0.364197 ± 0.017633 |                        0.033513 ± 0.001527 |

这里 `MDPO sampled-KL` 才是 MD-OPD 仓库的对应实现：它在行为 rollout token 上直接累积 KL estimator，没有显式 `rho`；`MDPO action-IS oracle` 才在 action 维度乘 `rho=pi/anchor`。两者在这个有限步数实验中接近，不能据此声称 sampled-KL 是无偏的。两种 MDPO 都没有 prefix-level `W_{<t}`，因此 prefix occupancy drift 仍然存在。

BPO 的原始论文没有公开代码仓库。sequence experiment 使用论文给出的 practical complement ratio
`omega=(1+epsilon-b)/(1+epsilon-pi)`、cap 和 sign mask；本地 `mopd_grpo` 只是 OPD 适配的参考实现，不能写成原始 BPO 官方代码。PMD-Mean 使用访问 prefix 内的 sampled mean，因此在有限 `N` 下有中心估计误差；PMD-Part 使用精确 `log Z`，所以它的主要误差来自参数投影和未访问 prefix。
