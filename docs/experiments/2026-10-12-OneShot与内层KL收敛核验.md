# One-Shot OPD 与内层 KL 收敛核验

## 结论

四卡 MDPO/BPO 的 `inner_student_center` 较小，与 One-Shot OPD 固定状态仍学习数百步的观察一致。两个量测量不同对象：前者是本轮冻结锚点到当前学生的 Tail-aware KL，论文主要观察学生与教师的对齐以及测试准确率。每个 outer 开始时，锚点等于学生，所以 student center 必须回到零；这一性质与教师监督是否已经吸收无关。

原始日志没有显示首个 inner 更新后目标已经收敛。MDPO 首轮 student center 从第一次实际更新后的 $4.2616\times10^{-4}$，继续增加到第 15 次更新后的 $6.2222\times10^{-4}$；对应梯度范数仍为 1.04495。BPO 首轮的变化为 $4.2979\times10^{-4}\to6.2468\times10^{-4}$，末测量梯度范数为 1.01840。全程 800 个 optimizer 更新不能按 50 个 outer 解释为 50 次学习。

首个更新后约 $4\times10^{-4}$ 的共同台阶确实需要数值解释。原始 1.5B 师生、单卡 FSDP 的短响应探针重现了该量级：BF16 前向首次 lr=$10^{-6}$ 更新后 student center 为 $4.0795\times10^{-4}$，FP32 前向为 $5.0819\times10^{-3}$。零学习率更新在两种精度下都保持 student center 为零。这个实验说明低精度前向会显著改变小步更新的可见效果；该台阶不能作为“监督已经学完”的证据。探针使用 SDPA、单条 128 token 响应，与正式四卡 FlashAttention 长响应训练的配置有差异。

损失公式、停止梯度、缓存时序和逐 token 归约的检查支持当前训练实现声明的目标。当前存在的诊断不足是：正式日志没有记录每个 inner 的当前完整词表师生 KL、论文式绝对 log-prob 差或镜像目标误差。因此这组日志不能给出 inner 达到最优的步数，也不能据此宣称相对论文缩短了 200～300 倍的吸收时间。

## 论文的固定状态实验

One-Shot OPD 第 5.3 节[^oneshot] 比较 always-on-policy 与 always-off-policy。后者复用初始学生从同一个 query 一次性采样得到的 64 条轨迹；一个 prompt 对应一批固定 response，每条 response 又包含大量 token 前缀。论文 Figure 10 报告 off-policy 测试准确率持续改善约 200 步，Top-16 overlap 与 overlap-token advantage 的改善也延续数百步。

论文第 5.1 节定义平均绝对 log-prob 差，并称为 distance：

$$
d_t=\frac{1}{|\mathcal T_t|}\sum_{i\in\mathcal T_t}\left|\log\pi_T(y_i\mid s_i)-\log\pi_{\theta_t}(y_i\mid s_i)\right|.
$$

绝对值保留动作上的正负差异，防止抵消。它和 KL 的有符号 Monte Carlo 估计具有不同的数值与衰减形状。论文第 5.2 节的 300 步分析仍保留 step 30 距离的 16%～22%；这里的“几百步”表示持续对齐的实验时间尺度，不能解释为一个统一阈值下的完全吸收。

## MD-OPD 的历史实验与代码

在 `MD-OPD/plots/runs/` 中找回了完整 W&B CSV 和配置 metadata。三个主要 run 如下；本次重新读取原始 CSV，未用既有报告中的表格替代数据。

| run        | 数据与训练方式                             | 历史长度 | W&B                   |
| ---------- | ------------------------------------------ | -------: | --------------------- |
| `5aub1k4h` | one-shot t22，online rollout，vanilla loss |      300 | One-Shot-OPD[^online] |
| `ceejzu26` | one-shot t22，frozen rollout，opd_noratio  |      300 | One-Shot-OPD[^frozen] |
| `mdxt04lm` | full DAPO，vanilla，历史基线               |      279 | One-Shot-OPD[^full]   |

前两组均使用学生 DeepSeek-R1-Distill-Qwen-1.5B、教师 JustRL-DeepSeek-1.5B、batch/mini-batch=64、ppo_epochs=1、lr=$10^{-6}$、response 上限 7168。每个 step 对同批 64 条轨迹执行一次全 batch optimizer 更新。full-data 历史基线的 prompt 长度和部分配置不同，不能作为只改变 rollout 的严格消融。

### 使用论文口径重算历史

旧报告 `2026-09-09-offpolicy-exp-analysis.md` 使用 $-\mathrm{mean}(A)$ 作为 signed distance proxy，其中 $A=\log\pi_T-\log\pi_\theta$。论文使用 $\mathrm{mean}(|A|)$。两者应分别读取 CSV 的 `opd/adv_mean` 与 `opd/adv_abs_mean`。

| run               | step | 平均绝对差 | 负平均 advantage | Top-16 overlap |
| ----------------- | ---: | ---------: | ---------------: | -------------: |
| frozen `ceejzu26` |    1 |   0.236357 |         0.103141 |       0.817851 |
| frozen `ceejzu26` |   30 |   0.195393 |         0.095805 |       0.840643 |
| frozen `ceejzu26` |  100 |   0.052585 |         0.008415 |       0.920601 |
| frozen `ceejzu26` |  200 |   0.029131 |         0.003125 |       0.933271 |
| frozen `ceejzu26` |  300 |   0.022049 |         0.001190 |       0.937154 |
| online `5aub1k4h` |   30 |   0.171506 |         0.063641 |       0.838426 |
| online `5aub1k4h` |  100 |   0.046863 |         0.005780 |       0.919153 |
| online `5aub1k4h` |  300 |   0.025069 |         0.001536 |       0.949680 |

固定 rollout 的平均绝对差在 step 100～300 又下降约 58.1%。这支持固定状态上继续学习数百步。其 step 300 绝对差为 step 30 的 11.28%；同一时点的负平均 advantage 只有 step 30 的 1.24%。只比较有符号 proxy 会把对齐程度看得更高。

对于 frozen run，动作仍来自初始分布，后续 $-\mathrm{mean}(A)$ 的期望是 $\mathbb E_{a\sim q}[\log p(a)-\log T(a)]$。它与当前分布的 reverse KL $\mathbb E_{a\sim p}[\log p(a)-\log T(a)]$ 有不同的采样测度；所以旧文档将整个 frozen 历史直接作为当前 reverse KL proxy 的解释需要加上此条件。

### 真实调用链

MD-OPD 当前 checkout 的 `verl/verl/trainer/ppo/ray_trainer.py` 在 frozen 模式复制第一次 rollout 的 tokens 与 states，每步重算当前学生 log-prob。`opd_trainer_mixin.py::compute_opd_token_reward` 用当前学生与教师的 log-prob 重新构造 advantage，并按 response mask 记录绝对差、均值和 RMS。教师权重冻结，教师评分与 Top-k 诊断仍重新执行。

`opd_core_algos.py::compute_policy_loss_opd_noratio` 的核心为

$$
\ell_i=-\operatorname{stopgrad}\big(\log T(a_i\mid s_i)-\log p_\theta(a_i\mid s_i)\big)\log p_\theta(a_i\mid s_i).
$$

动作固定，advantage 更新，且不带 PPO ratio。这是固定动作上的 score surrogate。它与 MirrorOPD 的每轮重采样、冻结 reward、16 次 inner 和镜像正则具有不同目标。历史 metadata 没有提供逐文件启动 SHA-256；这里的代码时序以当前 checkout 与保存配置相互核对，证据强度低于四卡实验的启动源码审计。

## 四卡日志的指标语义

### 分布、支持与测度

固定一个 outer，记冻结锚点为 $q$、当前学生为 $p$、教师为 $T$。轨迹由本轮学生采样，response 前缀在 16 次 inner 中固定。令 $H$ 为锚点选出的 Top-16，$R$ 为剩余词表，$q_R=\sum_{a\in R}q_a$；$p_R$ 与 $T_R$ 同理。

日志 `mirror/student_center/mean` 对应

$$
K_H(q,p)=\sum_{a\in H}q_a\log\frac{q_a}{p_a}+q_R\log\frac{q_R}{p_R}.
$$

它是锚点到当前学生的粗粒化 KL。W&B 名为 `04_kl_centers/inner_student_center/mean`。正式源码和所存档 W&B history 中没有字面字段 `inner_student_kl`；若图表标题采用这个简称，需要按实际字段确认。这里分析与描述相符的 student center 原始量。

日志 `mirror/teacher_center/mean` 对应 $K_H(q,T)$，由 `complete_mirror_teacher_cache` 在本轮更新前计算，此后冻结。MDPO/BPO 的 `mirror/sampled_reward` 是

$$
r(a,s)=\log T(a\mid s)-\log q(a\mid s),
$$

也在本轮固定。student center 每次学生前向重新计算；teacher center 与 sampled reward 在 inner 内恒定是缓存契约。

设每个 outer 的初始学生为 $p_0=q$，则逐个前缀满足

$$
K_H(q,p_0)=0,\qquad K_H(q,T)>0
$$

只要锚点尚未与教师相等，两个式子即可同时成立。学生到自己的距离为零，不推出学生到教师的距离为零。下一轮锚点重新设为上一轮结束的学生，student center 再次归零，800 个测量点因此包含 50 次锚点重置。

### Tail-aware KL 保留和丢失的信息

完整 KL 可以分解为

$$
\mathrm{KL}(q\Vert p)=K_H(q,p)+q_R\,\mathrm{KL}\big(q(\cdot\mid R)\Vert p(\cdot\mid R)\big).
$$

推导只需将尾部每项写为 $q_a=q_Rq(a\mid R)$ 与 $p_a=p_Rp(a\mid R)$，展开对数后分别求和。第二项非负，所以日志是完整 KL 的下界。尾部总质量相等时，尾部桶项为零，桶内部仍能有不同分布。首轮平均锚点尾部质量约 2.49%（MDPO）和 2.66%（BPO）；这些数值不能单独限制尾部条件 KL 的大小。

### 原始逐步数值

下表使用 zero-based `outer_call`。inner index 表示测量前已经完成的本轮 optimizer 更新数；记录是 pre-update，因此 index15 测量第 16 次更新之前的学生，日志不包含同一批上完成第 16 次更新后的快照。

| 方法 | outer_call | inner index | anchor→student KL | anchor→teacher KL | 梯度范数 |
| ---- | ---------: | ----------: | ----------------: | ----------------: | -------: |
| MDPO |          0 |           0 |              约 0 |          0.105989 | 1.275134 |
| MDPO |          0 |           1 |          0.000426 |          0.105989 | 1.274809 |
| MDPO |          0 |          15 |          0.000622 |          0.105989 | 1.044954 |
| MDPO |          4 |           1 |          0.000405 |          0.040866 | 0.459337 |
| MDPO |          4 |          15 |          0.003322 |          0.040866 | 0.224944 |
| MDPO |         49 |           1 |          0.000441 |          0.002646 | 0.053432 |
| MDPO |         49 |          15 |          0.000536 |          0.002646 | 0.042691 |
| BPO  |          0 |           0 |              约 0 |          0.104292 | 1.236035 |
| BPO  |          0 |           1 |          0.000430 |          0.104292 | 1.231237 |
| BPO  |          0 |          15 |          0.000625 |          0.104292 | 1.018401 |
| BPO  |          4 |           1 |          0.000415 |          0.039782 | 0.449914 |
| BPO  |          4 |          15 |          0.003317 |          0.039782 | 0.242985 |

50 轮的 inner index1 student center 平均值分别为 0.00043558（MDPO）与 0.00043522（BPO）。index15 的平均值分别为 0.00079843 与 0.00078698；MDPO 的 outer_call4 更从 index1 的 0.00040487 增至 index15 的 0.00332241，增长约 8.21 倍。把 50 轮内层叠在累计更新轴上，反复归零、相近的首步台阶以及早期少数较大峰值会掩盖单轮内层变化。

还可以从已有两个有符号均值重构

$$
\mathbb E_{q,\mathrm{fixed}\ s}[\log p(a\mid s)-\log T(a\mid s)]=\mathrm{mean}(u)-\mathrm{mean}(r),\qquad u=\log p-\log q.
$$

MDPO 首轮这个量由 0.115943 变为 index15 的 0.115455；outer_call4 由 0.043653 变为 0.038942；outer_call49 由 0.002422 变为 0.001823。它只能描述固定动作上的有符号差，不能命名为当前师生 KL，也不能用绝对值的均值代替绝对均值。完整重算表保存在 [inner-history.csv](../../results/oneshot-inner-audit/inner-history.csv)。

## 镜像内层的目标与收敛

### MDPO 的目标分布

代码的 MDPO 梯度系数是

$$
c_{\mathrm{MDPO}}(a)=\exp(u(a))\left(\frac{u(a)}{\eta}-r(a)\right),
$$

loss 为 $\operatorname{stopgrad}(c)\log p(a)$。动作在冻结锚点 $q$ 下采样；$\exp(u)=p/q$ 将动作期望转为 $p$。这里修正动作测度，前缀测度仍固定。按完整动作期望求梯度时，它对应

$$
J(p)=\frac{1}{\eta}\mathrm{KL}(p\Vert q)-\mathbb E_{a\sim p}r(a).
$$

把 $r=\log T-\log q$ 代入，令

$$
Z=\sum_a q_a\exp(\eta r_a),\qquad p_a^*=\frac{q_a\exp(\eta r_a)}{Z}=\frac{q_a^{1-\eta}T_a^\eta}{Z}.
$$

逐项展开可得

$$
\begin{aligned}
\eta J(p)&=\sum_a p_a\big(\log p_a-\log q_a-\eta r_a\big)\\
&=\sum_a p_a\big(\log p_a-\log p_a^*-\log Z\big)\\
&=\mathrm{KL}(p\Vert p^*)-\log Z.
\end{aligned}
$$

所以理想内层最优解是 $p^*$。当前 $\eta=0.3$，一次 outer 的目标为 $q^{0.7}T^{0.3}$ 的归一化分布。即使 inner 精确求解，学生与教师通常仍有差距。对任意动作 $a,b$，最优解满足

$$
\log\frac{p_a^*}{p_b^*}-\log\frac{T_a}{T_b}=0.7\left(\log\frac{q_a}{q_b}-\log\frac{T_a}{T_b}\right).
$$

该式仅说明独立可优化的单前缀概率分布在理想更新中的 log-odds 收缩。真实 LLM 参数共享、有限固定动作、Adam、BF16 与状态变化共同决定实际动力学，不能据此推算训练轮数。

### BPO 的精确目标与中心近似

用完整 KL 中心时，$K(q,p)=-\mathbb E_q u$，$K(q,T)=-\mathbb E_q r$。BPO 生产系数为

$$
c_{\mathrm{BPO}}=\frac{u+K(q,p)}{\eta}-r-K(q,T)=\frac{u-\mathbb E_q u}{\eta}-\big(r-\mathbb E_q r\big).
$$

它是目标

$$
J_{\mathrm{BPO}}(p)=\frac{1}{2\eta}\operatorname{Var}_{a\sim q}\big(u(a)-\eta r(a)\big)
$$

的 score 梯度系数。对固定 $q,r$，方差最小值为零要求 $u-\eta r$ 对所有动作为常数，再由归一化确定该常数为 $-\log Z$；精确 BPO 与 MDPO 的理想概率目标相同。

正式实现以 Tail-aware 中心替代完整中心。定义 $\delta_s=K_H(q,p)-K(q,p)$、$\delta_T=K_H(q,T)-K(q,T)$，生产系数与精确系数相差

$$
\Delta c=\frac{\delta_s}{\eta}-\delta_T.
$$

其期望梯度偏差是 $\Delta c\,\mathbb E_q\nabla_\theta\log p(a)$。当 $p=q$ 时该 score 期望为零，离开锚点后通常非零。这是已有实现中明确采用的中心近似，与“KL 很小所以学完”的判断分别处理。

一个二动作例子能直接区分内层结束与教师对齐。取 $q=(0.5,0.5)$、$T=(0.9,0.1)$、$\eta=0.3$，理想镜像解为 $p^*=(0.659073,0.340927)$。此时 $\mathrm{KL}(q\Vert p^*)=0.053357$，而 $\mathrm{KL}(p^*\Vert T)=0.212805$。镜像目标可以精确求解，教师距离仍然为正；用 anchor–student KL 判定教师吸收程度缺少必要条件。

## 代码与数据验证

本次重新验证了 MDPO 启动提交 `87ec5155f4ee4f7904aa275cf580261542350c82` 与 BPO 启动提交 `ac54a612fb9ef0692b5b65d6c7dae620bdbb39a9`。`losses.py`、`cache.py`、`metrics.py`、`tracking.py` 和 `integrations/verl/fsdp.py` 的提交内容 SHA-256 均与启动 provenance 相符。分析使用启动快照解释正式行为；当前 checkout 新增的 Top-k 成对 overlap 不能追溯填入正式日志。

两方法各有 800 条唯一 pre-update inner 记录，索引为 50×16，压缩归档解压内容与 data manifest 的 SHA-256 相符。全部 outer 内的 teacher center、sampled reward 和 anchor head mass 恒定；student center 在内层变化。MDPO 全部 800 条、BPO 存档中前 416 条在线 inner student center 与本地逐条相等，最大误差为零。BPO 完成时的在线 finished/851 history 行状态另见既有 final completion audit；本次逐条 parity 以实际保存的 history 范围为准。

重新运行 `tests/test_mean_bpo.py` 与 `tests/test_upstream_opd_semantics.py` 通过。前者包含解析梯度、有限差分、Gibbs 最优解、BPO Hessian 与 Fisher 动力学；后者包含生产 score 系数与完整词表目标的枚举梯度比较、Tail-aware BPO 中心误差、缓存与 jagged prediction-position 对齐。它们证明被覆盖的公式和接线关系，不证明正式 LLM 已达到最优。

短响应 GPU 探针使用相同原始模型、固定生成 tokens、Top-16、$\eta=0.3$、FP32 参数与 AdamW。记录初始策略、一次 lr=0 更新后的策略，以及随后三次 lr=$10^{-6}$ 更新后的策略。BF16/FP32 两臂使用相同输入，教师前向精度随条件变化；两臂均在零更新后保持 student center 为零。其完整数值在 [precision-probe.json](../../results/oneshot-inner-audit/precision-probe.json)，程序在 [oneshot_precision_probe.py](../../experiments/scoring/oneshot_precision_probe.py)。

BF16 量化还给出了可验证的参数级解释。原始 checkpoint 的权重本身为 BF16 数值。对第 0 层 Q projection 和第 12 层 MLP down projection 逐元素计算，将权重加减 $10^{-6}$ 再转回 BF16，能改变表示的元素比例分别只有 1.31% 与 1.10%。因此 FP32 master 参数发生小步变化时，大量元素的 BF16 前向值仍保持原值；后续累积更新跨过舍入边界时，可见变化才出现。这个比例是给定更新幅度的舍入界实验，不是正式训练实际变化元素数，结果保存在 [bf16-rounding-bound.json](../../results/oneshot-inner-audit/bf16-rounding-bound.json)。

## 后续比较的测量要求

公平检验固定监督的吸收步数，需要固定同一批初始轨迹、相同初始模型、相同学习率与 optimizer 更新计数，并在每次更新后独立评分同一批前缀。针对本问题，优先比较四个量：

1. 论文式动作绝对差 $\mathbb E_q|\log T-\log p|$，保持固定动作测度。
2. 完整词表当前师生 KL，分别报告 $\mathrm{KL}(p\Vert T)$ 与 $\mathrm{KL}(T\Vert p)$。
3. 固定镜像目标误差 $\mathrm{KL}(p\Vert p^*)$ 或 BPO 中心残差方差，确认 inner 是否接近其目标。
4. anchor–student KL 和重要性采样 ESS，说明偏离程度与动作重用有效性。

“inner 步数够用”需要目标误差下降轨迹和增大 inner 预算的对照；“监督已吸收”需要当前师生距离和固定测试任务。此次现有日志已经足以解释表面矛盾，尚不足以确定最佳 inner 次数。正式训练未被修改，也没有用单卡短响应探针代替四卡数学 benchmark。

复算命令：

```bash
python experiments/analysis/oneshot_inner_audit.py
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN PYTHONPATH=src \
  python experiments/scoring/oneshot_precision_probe.py
PYTHONPATH=src python experiments/scoring/oneshot_precision_probe.py --rounding-only
PYTHONPATH=src python tests/test_mean_bpo.py
PYTHONPATH=src python tests/test_upstream_opd_semantics.py
```

### 参考文献

[^oneshot]: Zixuan Fu, Bingxiang He, Yuxin Zuo, et al. Rethinking On-Policy Distillation of Large Language Models II: One Training Example. arXiv 2026. [arXiv:2609.04172](https://arxiv.org/html/2609.04172).

[^online]: WncFht. One-Shot-OPD online one-shot run 5aub1k4h. W&B 2026. [Run 5aub1k4h](https://wandb.ai/wncfht/One-Shot-OPD/runs/5aub1k4h).

[^frozen]: WncFht. One-Shot-OPD frozen rollout run ceejzu26. W&B 2026. [Run ceejzu26](https://wandb.ai/wncfht/One-Shot-OPD/runs/ceejzu26).

[^full]: WncFht. One-Shot-OPD full-data run mdxt04lm. W&B 2026. [Run mdxt04lm](https://wandb.ai/wncfht/One-Shot-OPD/runs/mdxt04lm).
