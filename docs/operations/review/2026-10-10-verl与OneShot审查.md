# verl 与 One-Shot-OPD 的训练契约审查

## 版本、范围与结论

本次核验的官方 verl 为 `fc72e2f384212470868a9928b62baedb9bcc3ff0`，One-Shot-OPD 为 `6b6b1782cc3f04749dd077de19353f839f6bc048`。独立源码位于 `~/.cache/mirroropd-review/verl` 与 `~/.cache/mirroropd-review/One-Shot-OPD`，没有覆盖 MD-OPD 工作区。本文中的源码事实均以这两个提交为准。One-Shot 对应论文是 Fu 等人的《Rethinking On-Policy Distillation of Large Language Models II: One Training Example》，arXiv:2609.04172，正文与附录已读取，相关公式与启动配置交叉核验。[^oneshot-paper]

统一应保留官方 verl 的模型执行、teacher server、数据布局、词表并行和权重同步设施，抽出框架无关的评分契约与目标内核。One-Shot 的 teacher-reward 路径和官方 direct distillation 路径共享模型评分能力，但它们的支持集、积分方向和反传规则分别定义。把二者压缩为一个 `top_k` 参数会丢失这些定义。

## 官方 verl 的实际调用链

官方 `distillation.enabled` 启动 `MultiTeacherModelManager`，按配置拆分 teacher resource pool，为每个 teacher 创建若干 inference replica。agent loop 在生成 student response 后，把完整 prompt+response token IDs 发给 teacher server 请求 `prompt_logprobs`。server 返回的 `teacher_ids` 与 `teacher_logprobs` 经 pad/unpad 后进入 actor engine。其 loss 有两个入口：完整 logits 处的 callback 计算 teacher-topk loss；最终 loss 处执行 sampled estimator 或聚合 callback 的每位置输出。[^verl-manager] [^verl-loss]

teacher 通过配置 key 路由，single-teacher 自动命中唯一模型；multi-teacher缺 routing key 或不存在的 key会报错。teacher服务管理器核验每个 replica 的 GPU数与节点边界，支持 vLLM 与 SGLang，并把Prometheus与RLInsight监控接到teacher replica。资源布局和teacher routing是执行后端的职责，算法侧只需要逐行teacher identity及固定版本。[^verl-manager] [^verl-model]

当前teacher请求始终使用temperature=1，非1配置只触发警告。它请求一个无关续写token，以获得全部prefix的prompt评分；续写不进入目标。vLLM的sampled模式用已有token LP，topk模式按返回rank挑teacher自己的topk并丢弃落在topk外的已有token；首位置缺LP，通过移位和末尾dummy把评分转到预测位置坐标。任意chosen IDs评分、温度重新归一化与尾部条件采样都没有作为该teacher manager的接口暴露。[^verl-manager] [^verl-vllm]

FSDP的teacher-topk direct kernel在current student完整logits上gather冻结teacher IDs。Ulysses先把teacher IDs/LP按sequence切分，loss与模型logits逐位置对齐，再回收完整每位置输出。可选chunked路径以FP32 logsumexp+gather替代完整log_softmax缓冲，减少forward临时峰值；它没有定制backward以保证全部saved tensors也同样有界，chunk大小不能当成训练显存上界。[^verl-fsdp]

Megatron保持词表分片。它用跨TP的MAX/SUM得到全词表normalizer，global token IDs映射到local shard，只有所属shard贡献teacher项。loss、teacher-active mass与diagnostics经collective聚合；custom backward构造每个local vocab coordinate的精确梯度。它还从各shard收集局部topk再选全局topk，以计算overlap。CP的THD/BSHD预处理与词表TP相互独立。[^verl-megatron]

## 支持矩阵

| 能力                                  | 官方 inference teacher + FSDP actor | 官方 inference teacher + Megatron actor | One-Shot FSDP teacher + FSDP actor                      |
| ------------------------------------- | ----------------------------------- | --------------------------------------- | ------------------------------------------------------- |
| 实际response action的teacher LP       | 已有                                | 已有                                    | 已有                                                    |
| teacher-own Top-K                     | 已有，冻结teacher IDs               | 已有，global IDs与local vocab mapping   | forward内部已有，最终reward没有以该支持集训练           |
| student/anchor指定IDs的teacher评分    | teacher manager未暴露               | teacher manager未暴露                   | 已有student IDs评分                                     |
| 非1 teacher评分温度                   | 该路径固定1                         | 该路径固定1                             | logits除以teacher温度                                   |
| teacher-topk raw FKL direct backward  | 已有                                | 已有custom backward                     | OPD reward入口不承担此任务；vendored direct入口另行存在 |
| sampled KL estimator转PG              | 已有                                | 已有                                    | k1差值冻结后转PG                                        |
| student Top-K冻结reward逐candidate PG | 原生direct入口没有此语义            | 原生direct入口没有此语义                | 已有，仅单epoch/full minibatch更新                      |
| tail mass作为独立积分单元             | 没有该目标入口                      | 没有该目标入口                          | 没有该目标入口                                          |
| 精确固定anchor期望/残差square         | 原生路径没有                        | 原生路径没有                            | 公共One-Shot版本没有；MD-OPD增量另行审查                |
| 多教师执行                            | 独立server pool和routing            | 独立server pool和routing                | actor GPU pool colocated FSDP teacher                   |
| teacher/student tokenizer映射         | 没有通用映射层                      | 没有通用映射层                          | 直接复用相同token IDs                                   |

所有“已有”均指源码存在相应路径，不代表本轮运行了GPU验证。异构teacher使用相同vocab大小还不足以保证语义一致；tokenizer词表、special IDs、chat模板和prefix tokenization必须核验并加入模型身份。

## 三种不同的 Top-K 数学对象

### teacher支持集上的raw forward KL

官方direct kernel令冻结teacher分布为$q$，current student为$p_\theta$，teacher选出的固定支持集为$S$。计算的是

$$
L_S(\theta)=\sum_{v\in S}q_v(\log q_v-\log p_{\theta,v}).
$$

定义$Q=\sum_{v\in S}q_v$。由$\partial\log p_v/\partial z_j=\mathbf1\{v=j\}-p_j$，逐步求导得到

$$
\frac{\partial L_S}{\partial z_j}=-\sum_{v\in S}q_v(\mathbf1\{v=j\}-p_j)=Qp_j-q_j\mathbf1\{j\in S\}.
$$

这正是Megatron的mass-scaled backward。若LP clamp触发，active集合$A$替换$S$，梯度为$Q_Ap_j-q_j\mathbf1\{j\in A\}$。上层又把每位置负loss clamp为0；clamp关闭/未触发时的等式不能自动推广到该组合。支持集概率没有重新归一化，因此它是raw截断积分，可以为负，不能标成完整KL或normalized KL。[^verl-megatron] [^verl-loss]

### student支持集上的冻结candidate reward

One-Shot选outer snapshot$b$的student Top-K，令$P_S=\sum_{v\in S}b_v$、$w_v=b_v/P_S$，逐candidate缓存

$$
A_v=w_v(\log q_v-\log b_v).
$$

更新时逐candidate计算$p_\theta(v)/b_v$的PPO ratio和clip，再求candidate和。无clip时的损失为

$$
L_{\mathrm{PG}}=-\sum_{v\in S}A_v\frac{p_{\theta,v}}{b_v}.
$$

在$\theta$等于outer snapshot时，其梯度为$-\sum_{v\in S}A_v\nabla\log p_{\theta,v}$。这里$A_v$和$S$均冻结。它既不同于上一节teacher加权FKL，也不同于直接对$\sum_v\tilde p_{\theta,v}(\log p_{\theta,v}-\log q_v)$求导，因为后者还要微分归一化权重。论文的标量advantage展示了各candidate贡献之和，代码则保留每candidate贡献供ratio与clip执行；训练契约必须记录这种粒度。[^oneshot-kernel] [^oneshot-paper]

One-Shot engine在每个train forward重新取student自己的Top-K，而非gather缓存IDs，因此只允许单epoch且mini batch覆盖全部rollout batch。这是现有实现的边界；冻结 `SupportSpec.ids` 后gather current LP可以移除candidate身份漂移，仍需另行定义多inner的目标和信任域。[^oneshot-mixin] [^oneshot-engine]

### 头部加尾部估计

本仓库 `03_kl_sampling_and_opd.tex` 明确目标测度$\nu$与采样测度$\rho$，`03_kl_autodiff.tex` 分开forward数值与backward向量。tail-aware与残差控制需要支持集原始质量、tail质量、采样来源及必要的重要性比率。官方teacher-topk输出的有限$q_v$允许求$q(S)$，但有限topk本身不能恢复任意被遗漏token的teacher LP或完整teacher温度变换normalizer。新估计量必须先检查后端能满足哪些score request，再选择算法；不能从一个`topk`布尔量推导后端能力。

## One-Shot 论文与发布代码的边界

论文完整分布目标写为$\mathbb E_{y\sim\pi_\theta}\sum_i\mathrm{KL}(\pi_\theta\Vert\pi_T)$。若把它解释为完整trajectory目标，导数含prefix访问测度随参数变化的项。代码固定已经采到的trajectory，只用局部冻结token reward做PG，没有该trajectory目标的完整return-to-go或访问测度微分。因此复现实验应命名为固定rollout上的局部更新，并在outer再采样；本仓库的固定测度理论与此契约对齐。论文目标公式不能替代计算图定义。[^oneshot-paper] [^oneshot-kernel]

论文数学one-shot主结果使用student Top-16；状态覆盖和absorption分析明确改用sampled-token，MOPD也用sampled-token。两类结果不能直接作为同一个loss的学习曲线拼接。论文absorption distance是同一batch上sampled action log-gap的绝对值，不是full-vocab KL；其跨step差分同时受visited states变化影响。用于比较统一内核时应增加固定prefix probe和同一分布上的gap，保留原指标作为实验复现指标。[^oneshot-paper]

论文MOPD附录指定`seq-mean-token-mean`，发布recipe继承common默认`token-mean`，没有在mopd recipe覆盖该值。实际复现需显式设置论文配置。论文和recipe的math response cap为7168，README摘要表写7680，README并非配置唯一事实源。[^oneshot-recipe] [^oneshot-paper]

论文固定state实验复用初始student产生的64条trajectory；公开One-Shot入口没有MD-OPD的`frozen_rollout`、replay buffer和anchor扩展。本文核验的公共版本不足以从现有recipe原样复现这些实验条件，MD-OPD本地扩展应单独保留版本与配置身份。

发布配置列出`only_stu/only_tch/intersection/union/union-intersection`，实际teacher helper除intersection外始终gather student IDs；reward的`only_tch`与`only_stu`同分支，union的2K mask又截回K，teacher-only候选没有进入loss。统一方案不能把这些配置项标为完整支持，应先令支持集组装成为独立纯函数，返回冻结IDs与valid mask，并让两个scorer在完全相同IDs上评分。[^oneshot-teacher]

## off-policy、归一化与同步

官方rollout correction把training旧snapshot LP与实际rollout LP的差作为log ratio，支持token或sequence权重、clipping、batch normalization和rejection。decoupled模式冻结old/rollout校正，PPO另用current/old ratio；bypass把oldLP设为rolloutLP，由loss处理current/rollout。它们解决不同身份之间的比率，不能再无条件乘上一遍Mirror/MDPO比率。[^verl-correction]

token action级校正并不能恢复参数变化后的prefix访问测度。有限支持采样的top-p/top-k删除动作时，有限重要性比率也不能恢复未知尾部。本仓库采样章节给出缺失支持项；统一配置编译器必须验证目标与行为分布的支持条件，或者明确目标只是已缓存prefix上的固定测度目标。

官方`agg_loss`的`token-mean`用全局有效response token数并乘DP size；sequence平均是另一种测度。candidate轴必须先按目标聚合，不能把K个候选当K个独立response token改变分母。本仓库 `sections/01_prefix_measure_and_reduction.tex` 证明各micro batch先局部均值再直接累积改变权重，也说明sequence长度依赖未来动作会改变条件动作分布。统一 `ReductionSpec` 要携带当前optimizer batch的token计数、DP平均约定、candidate reduction及一次optimizer step边界。[^verl-agg]

官方权重同步已有`CheckpointEngineManager.update_weights(global_steps)`：非naive模式关闭/abort请求，释放KV cache，建通信拓扑，发送actor与接收rollout权重，finalize，再恢复cache和unfinished requests。版本随weight update传递。这些设施应复用。Mirror需要在自己的outer完成后调用，并明确是否允许partial trajectory跨权重版本继续；若坚持一个outer对应一个行为snapshot，首版应禁止混合版本trajectory。[^verl-sync]

模型checkpoint和window恢复分别处理。官方actor/critic/dataloader checkpoint不等于保存Mirror评分cache、outer/inner索引、支持集或固定teacher中心。第一版限定outer完成且hard sync完成的恢复边界，可以减少持久化对象；跨outer replay和exact mid-inner resume要求额外保存cache、snapshot、sampler状态与身份hash。

## 对统一接口与实现路线的影响

1. `DistributionSpec` 分开behavior、proximal、center、teacher身份，包含checkpoint revision、temperature、tokenizer hash、action transform和LP convention。允许算法明确声明两角色同一分布，禁止靠字段名推断。
2. `SupportSpec` 冻结IDs、selector snapshot、valid mask与raw masses。`only_stu/only_tch/union/intersection`只改变选集合；normalized/truncated/tail-aware只改变积分规则，两轴独立。
3. `ScoreRequest` 声明sampled action、chosen IDs、own topk、tail mass或full moments；`ScoreBundle` 返回量和分布/位置/支持集身份。teacher backend输出评分，不输出某个算法的reward。
4. `ObjectivePlan` 同时声明forward诊断、backward规则、冻结字段、live字段、需要的score能力和reduction。scalar estimator、PG coefficient、residual-square拥有不同kernel，避免直接复用名字相近的KL loss。
5. `WindowState` 管理rollout、评分冻结、inner、同步、缓存释放与恢复。FSDP/SP/TP、server资源、collective排程、pad/unpad和snapshot物化留在adapter。

FSDP是首版Mirror生产接入的正确目标：已有完整current logits、student candidate extraction和CPU可对照的kernel。官方engine callback足以承载局部评分/目标，但缓存应由window拥有，不能把framework TensorDict变成CPU核心的公共API。Megatron第二阶段增加vocab-sharded gather、mass reduction与custom gradient；应借鉴现有FKL的TP ownership和CP布局，而不是先all-gather完整V再称为并行支持。teacher server方案适合sampled LP/teacher-own Top-K与部署分离，chosen-ID/full-moment要求需要新backend能力或FSDP fallback。

首个验证阶段使用小词表和固定prefix，比较loss、全logits gradient、stop-gradient、支持集identity以及microbatch/DP reduction。随后同一checkpoint的FSDP与server LP做batch级差异基准，记录dtype、temperature和token位置。性能基准记录forward/backward峰值、teacher FLOPs、通信bytes与每有效token walltime，再决定chunked/custom-autograd/TP优化。

## 依赖与版本策略

两个源码仓库均为Apache-2.0，One-Shot的released checkpoint另受base-model license约束。复用源码保留LICENSE、NOTICE与原文件copyright，新增kernel记录来源。teacher model权限和数据来源单独进入运行manifest。[^oneshot-readme] [^verl-license]

官方内部接口使用nested TensorDict、THD/BSHD、Ulysses、CP/TP、loss callback和checkpoint engine，公共One-Shot又在这些位置追加专有字段。统一不应发布一个绑定这些内部类的核心包。固定upstream commit及依赖锁，adapter契约测试覆盖callback输入/输出、shift、SP/TP shape、reduction和同步version；升级先在干净基线重放小补丁，再跑CPU数学/contract检查和GPU canary。每次实验保存核心包commit、upstream commit、patch hash、backend依赖版本与resolved config，避免源码目录一致被误认成运行版本一致。

### 参考文献

[^oneshot-paper]: Fu 等. Rethinking On-Policy Distillation of Large Language Models II: One Training Example. arXiv 2026. [2609.04172](https://arxiv.org/abs/2609.04172).

[^oneshot-readme]: Thinking-Space. One-Shot-OPD README. 固定提交 2026. [README](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/README.md).

[^oneshot-kernel]: Thinking-Space. OPD advantage 与 candidate PPO kernel. 固定提交 2026. [opd_core_algos.py](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/verl/verl/trainer/ppo/opd_core_algos.py#L128).

[^oneshot-teacher]: Thinking-Space. FSDP teacher、Top-K scoring与reward. 固定提交 2026. [fsdp_teacher_worker.py](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/verl/verl/workers/fsdp_teacher_worker.py#L104).

[^oneshot-mixin]: Thinking-Space. OPD trainer validation与reward编排. 固定提交 2026. [opd_trainer_mixin.py](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/verl/verl/trainer/ppo/opd_trainer_mixin.py#L145).

[^oneshot-engine]: Thinking-Space. FSDP student Top-K extraction. 固定提交 2026. [transformer_impl.py](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/verl/verl/workers/engine/fsdp/transformer_impl.py#L1303).

[^oneshot-recipe]: Thinking-Space. One-Shot公共启动配置. 固定提交 2026. [common recipe](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/recipe/_common.sh#L46).

[^verl-manager]: verl. Async teacher scoring manager. 固定提交 2026. [teacher_manager.py](https://github.com/volcengine/verl/blob/fc72e2f384212470868a9928b62baedb9bcc3ff0/verl/experimental/teacher_loop/teacher_manager.py#L35).

[^verl-model]: verl. Multi-teacher deployment与监控. 固定提交 2026. [teacher_model.py](https://github.com/volcengine/verl/blob/fc72e2f384212470868a9928b62baedb9bcc3ff0/verl/experimental/teacher_loop/teacher_model.py#L62).

[^verl-vllm]: verl. Prompt LP extraction及预测坐标对齐. 固定提交 2026. [vLLM utils](https://github.com/volcengine/verl/blob/fc72e2f384212470868a9928b62baedb9bcc3ff0/verl/workers/rollout/vllm_rollout/utils.py#L554).

[^verl-loss]: verl. Distillation registry、direct/PG选择与截断loss clamp. 固定提交 2026. [losses.py](https://github.com/volcengine/verl/blob/fc72e2f384212470868a9928b62baedb9bcc3ff0/verl/trainer/distillation/losses.py#L230).

[^verl-fsdp]: verl. FSDP Top-K forward KL. 固定提交 2026. [FSDP losses](https://github.com/volcengine/verl/blob/fc72e2f384212470868a9928b62baedb9bcc3ff0/verl/trainer/distillation/fsdp/losses.py#L75).

[^verl-megatron]: verl. Vocab-parallel forward KL及custom backward. 固定提交 2026. [Megatron losses](https://github.com/volcengine/verl/blob/fc72e2f384212470868a9928b62baedb9bcc3ff0/verl/trainer/distillation/megatron/losses.py#L65).

[^verl-correction]: verl. Rollout correction与rejection. 固定提交 2026. [rollout_corr_helper.py](https://github.com/volcengine/verl/blob/fc72e2f384212470868a9928b62baedb9bcc3ff0/verl/trainer/ppo/rollout_corr_helper.py#L786).

[^verl-agg]: verl. Loss aggregation. 固定提交 2026. [agg_loss](https://github.com/volcengine/verl/blob/fc72e2f384212470868a9928b62baedb9bcc3ff0/verl/trainer/ppo/core_algos.py#L1140).

[^verl-sync]: verl. Checkpoint engine与rollout权重同步. 固定提交 2026. [checkpoint engine](https://github.com/volcengine/verl/blob/fc72e2f384212470868a9928b62baedb9bcc3ff0/verl/checkpoint_engine/base.py#L510).

[^verl-license]: verl. Apache License 2.0. 固定提交 2026. [LICENSE](https://github.com/volcengine/verl/blob/fc72e2f384212470868a9928b62baedb9bcc3ff0/LICENSE).
