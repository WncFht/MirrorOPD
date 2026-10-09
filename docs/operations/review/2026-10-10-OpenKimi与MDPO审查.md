# OpenKimi 与传统 MDPO 审查

## 审查范围与固定身份

本审查只读取隔离目录 `~/.cache/mirroropd-review/`，没有修改仓库现有 `upstream/` checkout，也没有实现训练代码。OpenKimi 使用固定提交 `737515818bf950e0c3f3ee2483ae0c53be435385`，MDPO 使用固定提交 `c9bd3ba080e34f94e104ffe7633167e246dd63c6`。两套源码和论文通过 `link-get` 的 arXiv TeX 路径读取，论文文本保存在 review cache 中。

OpenKimi 固定提交对应 `horizon-llm/OpenKimi` 的 `main`，论文为《Approximation of Log-Partition Function in Policy Mirror Descent Induces Implicit Regularization for LLM Post-Training》[^openkimi-paper]。MDPO 固定提交对应 `manantomar/Mirror-Descent-Policy-Optimization` 的 `master`，论文为《Mirror Descent Policy Optimization》[^mdpo-paper]。源码链接均指向固定 commit，而不是随时间变化的默认分支。

## OpenKimi 的算法对象

### 论文定义

OpenKimi 论文把 LLM 训练写成 contextual bandit：prompt 是 state，完整 response 是 action，reward 是 response-level 标量。给定旧策略 `pi_t`，理想 PMD-part 目标是

$$
\min_\pi \frac12\mathbb E_{y\sim\pi_t}
\left[\log\frac{\pi(y)}{\pi_t(y)}-
\frac{r(y)}{\tau}+\log Z_t\right]^2,
\qquad
Z_t=\mathbb E_{\pi_t}\exp(r/\tau).
$$

PMD-mean 用旧策略下的 reward mean 代替 log-partition：

$$
\min_\pi \frac12\mathbb E_{\pi_t}
\left[\log\frac{\pi(y)}{\pi_t(y)}-
\frac{r(y)-\mathbb E_{\pi_t}r}{\tau}\right]^2.
$$

论文完整推导表明，PMD-mean 的 population 解不是 Gibbs target。它由 Lambert-W 给出，并等价于带自适应 `chi^2` 正则的 mirror subproblem。这个差异在无限样本时仍然存在；增加 rollout 数量只能消除均值估计误差，不能把 PMD-mean 变回 PMD-part。

论文的假设是旧策略在 response action 空间上严格正、policy class 可实现 target、优化误差有界、log-ratio 有界。论文实验使用 binary `+/-1` response reward、每 prompt 16 个 response、temperature 1、top-p 1。`top-p=1` 是支持假设的一部分，不能把截断 rollout 直接解释成同一个无偏 PMD 目标。

### verl 实现

固定 commit 的 [verl core algorithms](https://github.com/horizon-llm/OpenKimi/blob/737515818bf950e0c3f3ee2483ae0c53be435385/openkimi/backends/verl/core_algos.py) 注册了三类组件：`ploo` advantage、复用 verl `rloo` advantage、以及 `opmd` policy loss。

`ploo` 对同一个 prompt 的 response rewards 计算 leave-one-out log-mean-exp partition baseline：

$$
\widehat A_i=r_i-\tau\log\left(
\frac1{K-1}\sum_{j\ne i}e^{r_j/\tau}\right).
$$

`rloo` 使用 leave-one-out arithmetic mean。两者都在 response level 构造 scalar advantage，然后把同一个 advantage broadcast 到 response tokens。`opmd` 再把整条 response 的 current/old log-prob ratio 求和，形成 sequence-level `U_i`，并计算

$$
\frac{\tau}{2}\left(U_i-\frac{A_i}{\tau}\right)^2.
$$

默认 aggregation 是每条 response 的 token mean，再对 response 做 sample mean；`pmd_ratio_agg_mode=mean` 会改变 `U_i` 本身，而不是只改变 loss 的尺度。`partition_weights` 还可以在 prompt group 内按指数 reward 权重或 reward-range filter 筛选样本。

这条实现链的数学对象是 **response-level regression**，不是 MirrorOPD 的 **prefix/token-level fixed-measure objective**。把 `U_i` 换成每 token `u_{i,j}` 并不能保持等价，因为平方与求和次序不同：

$$
\left(\sum_j u_j-A\right)^2
\ne
\sum_j\left(u_j-a_j\right)^2
$$

除非另加非常强的独立、正交或特定 advantage 分解假设。统一实现必须把 `trajectory_integral` 与 `token_integral` 作为不同 recipe，而不能只增加一个 normalization flag。

### slime 实现

固定 commit 的 [slime PMD loss](https://github.com/horizon-llm/OpenKimi/blob/737515818bf950e0c3f3ee2483ae0c53be435385/openkimi/backends/slime/pmd_loss.py) 同样先对每条 response 的 `log_prob - old_log_prob` 按 loss mask 求和，再将 sequence ratio broadcast 回 token。默认 `pmd_length_normalization=true` 只在平方 loss 后乘 `1/response_length`，不改变 sequence ratio。实现还支持 adaptive tau：依据 advantage 的上下界把 tau 置为零或缩放，这改变每 token 的有效目标，不是普通数值稳定化。

slime 的 `get_sum_of_sample_mean` 对每个 sample 单独除以 response length，再跨 sample 求和；启用 per-token loss 时改为纯 token sum。context parallel 会先重建全 response 的 mask，再做 sample reduction。这个 reducer 处理的是 response sample 的公平权重，不等同于 MirrorOPD 的 prefix token sufficient statistics。

OpenKimi 还提供 partial rollout、dynamic reward-variance filter、weight-version staleness filter 和异步 weight update。它们解决吞吐与 stale sample admission，不改变 PMD objective；但 staleness filter 只限制 weight-version span，不提供 importance correction。复用其调度时，必须把 stale age 记录到 `ProposalRecord`，不能把“被过滤”误写成“已做 off-policy 校正”。

### 已验证的实现风险

`ploo` 用 shifted exp 的 group sum 减去自己的 exp 项计算 LOO。唯一高 reward response 的其余项远小于自身时，这个减法会丢失全部有效位。CPU float64 枚举 `scores=[1,-1]`、`tau=.005`，literal kernel 的 baseline 为 `[0.9078965963,1]`，advantage 为 `[0.0921034037,-2]`；直接 masked logsumexp 的 exact LOO baseline 为 `[-1,1]`，advantage 为 `[2,-2]`。即使 float64 仍然失败，因此不能只把 dtype 提高作为修复，应直接对排除自己的集合计算 logsumexp。

`opmd` 收到 token-wise `rollout_is_weights` 后，把这些权重按 response 做算术平均。若目标是 response action 的 anchor-measure integral，则一般需要完整 response 的 joint weight，即各 token ratio 的乘积；算术均值没有该语义。统一方案允许复现这种历史 correction recipe，但禁止标注为 exact trajectory correction。

slime example 使用 `advantage_estimator=grpo` 并关闭 std normalization。这是包含主样本的 group mean center，区别于 verl 的 RLOO mean center。对 $K$ 个同 prompt samples，group-mean advantage 是 RLOO advantage 的 $(K-1)/K$ 倍；在 PMD regression 中，该因子改变 target，而不只是整体 gradient scale。两个 backend 必须记录具体 center estimator，不应以同名 PMD 宣称逐样本等价。

OpenKimi theory 定义的 half-square 与 appendix experiment loss 的系数相差常数尺度；源码使用 `0.5*tau`，论文 appendix 写 `tau`。这不改变固定 target 的最优解，却改变相同 SGD learning rate 下的步幅；Adam、epsilon 和 clipping 下更不能将这种差异忽略。

### OpenKimi 代码与论文的差异

论文把 PMD-part/mean 写成完整 response action 的回归；源码把 advantage broadcast 到 response token，再由 sequence ratio 重建 action log-ratio。论文实验说明 reward 是 `+/-1`，源码配置允许一般 reward bounds 和 adaptive tau。论文的稳定性解释依赖 Lambert-W 的 population solution；源码的 finite group、length normalization、prompt filter 和 optimizer 误差没有进入该理论保证。论文没有证明异步 partial rollout 的 stale filter 保持原始 PMD 更新，因此该功能必须作为系统 heuristic 单独报告。

## 传统 MDPO 的算法对象

### 论文定义

传统 MDPO 论文研究连续控制 MDP，不是 LLM token distribution。它的 state measure 是 discounted occupancy `rho_pi`，action measure 是当前 policy 或 replay-induced distribution。On-policy MDPO 在旧策略 `pi_k` 采样 trajectory，固定 advantage `A^{pi_k}`，对如下 proximity objective 做多步 SGD：

$$
\max_\theta\;\mathbb E_{s\sim\rho_{\pi_k}}
\left[\mathbb E_{a\sim\pi_\theta}A^{\pi_k}(s,a)
-\frac1{t_k}\operatorname{KL}(\pi_\theta\|\pi_{\theta_k})\right].
$$

旧 policy 在每个 outer iteration 固定；inner 多步更新的目的正是让 KL curvature 在第一步以后发挥作用。论文明确指出，一步更新在初始化点没有 KL gradient，退化为普通 policy gradient。论文的 on-policy `t_k` 从约 1 退火到 0，传统代码中使用 `sgd_steps` 个更新。

Off-policy MDPO 从 replay buffer 取 state，critic 学习 `Q` 与 `V`，actor 用 reparameterization 优化 `log pi_theta - log pi_theta_k - t_k Q`。这不是 token-level teacher distillation，也不是 response reward 的简单 importance-weighted score；它需要连续 action、critic、target critic 和 replay state distribution。论文还分析了 Tsallis Bregman divergence 与 SAC 的关系。

### 传统代码

固定 commit 的 [on-policy implementation](https://github.com/manantomar/Mirror-Descent-Policy-Optimization/blob/c9bd3ba080e34f94e104ffe7633167e246dd63c6/mdpo_on/mdpo.py) 建立 `model` 与 `oldpi` 两套 policy，rollout generator 使用 old policy。每轮先计算 GAE/advantage，并标准化 advantage，然后执行 `sgd_steps` 次 actor update；完成后通过 `assign_old_eq_new` 硬同步 old policy。代码同时训练 value function，checkpoint 保存 TensorFlow 参数和超参数。

固定 commit 的 [off-policy implementation](https://github.com/manantomar/Mirror-Descent-Policy-Optimization/blob/c9bd3ba080e34f94e104ffe7633167e246dd63c6/mdpo_off/mdpo.py) 使用 replay buffer、double Q、value target 和 Polyak target update。actor、Q、V 使用独立优化器；policy old network 每隔训练周期硬同步。这个实现依赖 2020 年 TensorFlow 1.x、stable-baselines、MuJoCo 和旧版 Gym，源码的 `requirements.txt` 固定了 `tensorflow-gpu==1.15.0`、`torch==1.6.0`、`numpy==1.19.1` 等环境，不能直接作为 archbox 上的 LLM trainer dependency。

代码中 `mdpo_on` 的 `meankl` 是 Gaussian distribution closed-form KL；`mdpo_off` 的 policy proximity 通过 continuous Gaussian reparameterization 和 Q estimate 近似。两者都没有 token Top-K、sampled OPD、teacher requested IDs 或 vocabulary tail。它们可以贡献 outer/inner 生命周期与硬同步概念，不能贡献 MirrorOPD 的 token estimator。

`mdpo_off` 的 `gradient_steps` 实际用于 old-policy 硬同步间隔；主训练循环每个 eligible environment step 只执行一次 gradient update。这个行为对应论文说明的跨 environment steps 累积多步近似，区别于在一个固定 replay batch 上连做 $m$ 次。actor 使用 current `qf1_pi`，并随着 critic 更新继续变化，不能把该代码解释为整个 inner 中冻结 reward/critic 的 MirrorOPD 子问题。

当前源码还存在未完成或破损路径。off-policy `reparameterize=False` 分支引用未定义的 `logp_old` 和 `logp_pi_old_action`；on-policy TensorBoard writer 分支给 `compute_lossandgrad` 的 placeholder 输入数量少于其定义。save metadata 未完整保存所有 MDPO-specific configuration，也没有保存 replay buffer。它们不作为本方案可运行的实现来源；传统代码只承担公式与历史生命周期的对照。

### 传统 MDPO 的 paper-code 差异

论文的 on-policy objective 使用未标准化理论 advantage 和 state occupancy；代码默认 GAE、advantage standardization、gradient clipping 与学习率 schedule。论文的 off-policy objective使用 replay states 与 approximate critic；代码还加入 target-network Polyak update、double-Q 与 Gaussian-specific regularization。论文结论是连续控制 benchmark 的多步 SGD 近似，不能直接推导 LLM fixed-prefix convergence。

## 与 MirrorOPD 的逐项对照

| 维度            | OpenKimi                                            | 传统 MDPO                                      | MirrorOPD 当前实现                             | 统一结论                                               |
| --------------- | --------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------ |
| action 粒度     | 完整 response                                       | environment action                             | token at fixed prefix                          | trajectory 与 token recipe 分开                        |
| state measure   | prompt dataset / rollout groups                     | discounted occupancy 或 replay                 | frozen prefix population                       | measure 必须显式存储                                   |
| anchor          | old response policy                                 | old policy `pi_k`                              | outer anchor `q`                               | `PolicyVersion` 必须带 snapshot                        |
| target          | PMD-part 或 PMD-mean regression                     | advantage + KL trust region                    | Part/Mean/BPO/MDPO/OPD                         | objective 与 gradient recipe 分离                      |
| normalization   | sequence ratio sum；loss 可 length normalize        | batch/GAE normalization                        | token/DP sufficient stats                      | reducer 不是 objective 参数                            |
| baseline        | LOO arithmetic / partition reward baseline          | V / Q critic，GAE                              | teacher/student KL center、OPD scalar baseline | target centering、value baseline、control variate 分开 |
| sampled support | 论文要求 full response policy；源码 rollout top-p=1 | continuous Gaussian support                    | top-k partition + sampled action               | support coverage 是 capability gate                    |
| inner           | mini-step policy regression                         | multi-step SGD                                 | outer frozen cache + finite inner              | sync boundary 与 cache freeze 统一抽象                 |
| sync            | async/partial weight-version control                | old policy hard sync；off-policy Polyak target | hard/soft anchor sync                          | weight version 与 probability evidence 同步记录        |
| critic/value    | 无独立 value critic                                 | on-policy V；off-policy Q/V                    | 无 critic，KL/control diagnostics              | 不把 Q/V baseline 塞进 token center                    |
| checkpoint      | verl/slime trainer checkpoint                       | TensorFlow model/old/target state              | FSDP actor/checkpoint diagnostics              | resume 必须恢复 policy versions/cache policy           |
| 性能边界        | sequence full log-prob，异步 rollout                | continuous control legacy stack                | full local vocab，SP=1 cache adapter           | 先统一 semantic CPU kernel，再做 distributed backend   |

## 统一方案

### 共同最小接口

统一层应围绕固定 prefix 的积分契约建立，而不是围绕名字 `PMD` 或 `MDPO` 建立：

1. `PrefixBatch`：prefix IDs、prediction positions、valid mask、response grouping、prefix measure。
2. `PolicyVersion`：角色、参数 snapshot、temperature/logit transform、tokenizer/vocab identity、weight version。
3. `ProposalRecord`：实际 action、proposal log-prob、proposal support、是否 top-p/top-k、trajectory/token proposal。
4. `SupportPartition`：support selector、固定范围、head IDs、tail cell、head/tail masses。
5. `DistributionEvidence`：requested IDs、sampled action、partition masses、full-vocab availability 和数值 dtype。
6. `ObjectiveSpec`：integrand、integration measure、direction、normalization、target centers、是否 frozen。
7. `GradientProgram`：deterministic head、sampled residual、importance weight、stop-gradient policy、forward value semantics。
8. `ReductionSpec`：per-prefix、per-token、per-sequence、per-prompt、DP/CP aggregation；输出 raw sums/counts，不平均 micro means。

所有 recipe 都必须返回一对 `(forward_estimate, backward_program)` 和一份 semantic metadata。生产 surrogate 的 scalar 不得自动被解释为 objective value。

### Recipe 划分

- `mirror_pmd_part_token`：anchor measure，full/partition reward center，平方 residual；需要 exact log-partition 或声明 partition approximation。
- `mirror_pmd_mean_token`：anchor measure，teacher reward center frozen；完整 center 与 partition center 必须不同类型。
- `mirror_bpo_token`：anchor measure，student center + teacher center；approximate student center 会改变 gradient，必须报告 bias。
- `mirror_mdpo_token`：current student target measure，用 anchor action proposal 和 stopped `p/q` importance ratio；保留有限 eta 的 `U/eta` 项。
- `opd_current_measure`：current student measure、teacher-current reward、可选 scalar baseline；exact `p/rho` 下 baseline 只改方差。
- `opd_partition_scalar`：仅作为 partition KL value/control，不宣称 full KL。
- `hets_reverse_kl`：deterministic head score + sampled tail residual；head backward 常数与 tail K4 形式必须成对声明。
- `resopd_reverse_kl`：partition head、aggregate tail control 和 residual sample；不能降格为普通 scalar baseline。
- `response_pmd_mean/part`：OpenKimi 的完整 response action recipe，使用 response log-ratio 和 response reduction；不与 token recipe 共享目标名。

### Top-K、normalized、tail-aware、sampled 的保留规则

Top-K support 是离散 partition，不是“更便宜的 full KL”。应同时保留四种明确对象：

- truncated head：确定性 head value，通常有系统 tail bias；
- normalized support KL：在 selected support 上分别重归一化，目标是 conditional KL；
- tail-aware partition KL：head 加一个 aggregate tail cell，目标是 coarsened KL；
- sampled-tail residual：head 加实际 action tail residual，只有 proposal 覆盖残差时才无偏。

当前 production cache 的 head/tail center 属于第三类。它可以作为诊断或明确声明的 partition objective；当它放入 Mean/BPO target center 时，会改变目标。`opd_onpolicy` / `opd_is` 的 scalar baseline 则可在 exact measure correction 下只影响方差。EMA/HETS 的 backward 必须额外声明 head coefficient 是否 stopped，不能从 scalar forward 名字推导。

### 生命周期与恢复

每个 outer step 必须生成新的 `PolicyVersion`，冻结 anchor evidence、teacher requested-ID evidence、actual proposal 与 support partition；所有 inner updates 只刷新 current student evidence。hard sync 将 current student 参数变成下一个 anchor；soft sync 要记录概率空间或参数空间的具体定义。checkpoint 至少保存 outer/inner counters、anchor version、proposal transform、support selector、cache schema、tokenizer hash 和 reduction spec。恢复时若无法证明 cache 与 anchor version 一致，必须重新 score，而不是复用旧 cache。

## 必须先做的 CPU 审计

统一前先在 finite categorical vocabulary 上枚举完整 action：

1. 对每个 recipe 比较 exact objective gradient、stopped score gradient、sampled estimator 的 conditional mean。
2. 使用 proposal `rho != q != p` 检查 importance correction 和 missing-support bias。
3. 比较 full、truncated、normalized、tail-aware 和 sampled-tail 的 value bias、gradient bias、variance trace、ESS。
4. 对 Mean/BPO 单独扫 `C_t`、`C_s` 的 partial-center error，验证偏差分别为 `-eta delta_t(q-p)` 与 `(delta_s/eta-delta_t)(q-p)`。
5. 对 response-level OpenKimi loss 与 token-level Mirror loss 用相同 logits 反例验证平方和与和平方不可互换。
6. 检查 hard sync、soft sync、stale anchor、finite inner 的 outer recurrence；训练质量实验之前先报告这些误差。

本次已运行第 5 类最小反例：令两 token 的 log-ratio 为 `[0.4,-0.4]`，advantage 为零。response loss `0.5*(sum u)^2` 的 value 为零且 gradient 为 `[0,0]`；token loss `0.5*sum u^2` 的 value 为 `0.16` 且 gradient 为 `[0.4,-0.4]`。同一个后端 reduction 无法修复这种目标差异。

## 性能、依赖和维护结论

OpenKimi 是 Apache-2.0 项目，含 Apache-2.0 `LICENSE`，并通过 git submodule 引入 verl 与 slime；分发其派生改动时需要保留许可证、版权和 NOTICE 要求。它适合作为训练调度、sequence/context parallel reduction、async rollout 和 weight-version control 的参考，但第三方 submodule commit 必须继续由 manifest 锁定。

传统 MDPO 仓库没有单独的 LICENSE 文件，README 只给出论文引用；源码依赖旧 stable-baselines、TensorFlow 1.15、MuJoCo 和 Gym。它应作为论文算法参考和历史实现归档，不应把其代码直接复制进 MirrorOPD 生产包。若未来复用片段，先取得上游许可并保留归属记录。

最佳统一边界是：MirrorOPD 保留唯一第一方 token estimator 与 cache contract；OpenKimi 贡献 response-level PMD、异步/partial rollout 和 scalable reducer 的接口经验；传统 MDPO 贡献 outer/inner、old-policy freeze、多步优化、critic 与 target-network 的概念对照。三者不应合并为一个可任意拼接的 loss。

### 参考文献

[^openkimi-paper]: Zhenghao Xu, Qin Lu, Changlong Yu, Tuo Zhao. _Approximation of Log-Partition Function in Policy Mirror Descent Induces Implicit Regularization for LLM Post-Training_. arXiv 2602.05933, 2026. [arXiv](https://arxiv.org/abs/2602.05933)

[^mdpo-paper]: Manan Tomar, Lior Shani, Yonathan Efroni, Mohammad Ghavamzadeh. _Mirror Descent Policy Optimization_. arXiv 2005.09814, 2020. [arXiv](https://arxiv.org/abs/2005.09814)
