# 具身 MirrorOPD：异质策略蒸馏的文献依据与实验方案

更新日期：2026-10-09。本文中的 OPD 指在线策略蒸馏（On-Policy Distillation），MOPD 指本仓库的 MirrorOPD。文献中同一缩写也用于多教师在线策略蒸馏；二者需要分别命名。研究范围聚焦机器人操作，覆盖单臂、双臂以及跨机器人训练。

## 1. 判断与研究定位

这个方向具备可执行的起点。VLA-OPD 已经在机器人环境中实施学生 rollout、冻结教师逐动作反馈和反向 KL 更新，说明具身 OPD 已有直接实验先例。HPT、CrossFormer 等工作已经提供异质观测和动作空间的联合建模接口。因此第一版可以在已有策略架构上检验 MirrorOPD，无需先解决整个具身领域的表示统一问题。[^vla-opd][^hpt][^crossformer]

当前最有价值的研究命题是：在相同网络、相同教师和相同交互预算下，MirrorOPD 的冻结锚点与有限内层更新，能否提高异质任务联合蒸馏的样本效率，并减少某些任务或机器人的能力退化。这个命题中的效果需要实验建立。本仓库现有固定测度推导没有给出共享神经网络上的跨任务正迁移保证。

“OPD 能做，MirrorOPD 就能做”在接口兼容层面成立需要三个条件：同一机器人上的教师和学生定义在同一个动作空间；二者能对同一学生动作计算概率或密度；教师能接收学生实际到达的观测历史。训练效果还取决于教师在偏离示范的状态上是否可靠、学生表达能力以及共享参数的梯度冲突。

## 2. 直接相关的论文

下表列出最能决定研究路线的工作。年份优先表示首次公开年份；RLDG 同时标注 2025 年正式发表。各模型的论文性能来自各自设置，不能按表格顺序当作同预算排名。

| 工作                           | 已验证的内容                                                                    | 对本项目的用途                              | 接入 MirrorOPD 的条件                        |
| ------------------------------ | ------------------------------------------------------------------------------- | ------------------------------------------- | -------------------------------------------- |
| Open X-Embodiment / RT-X，2023 | 汇集 22 类机器人数据并展示跨机器人迁移                                          | 数据标准化与多机器人训练的起点              | 数据容器标准化后仍要核对动作语义             |
| OpenVLA，2024                  | 用离散动作 token 训练开放 VLA                                                   | 第一版概率接口明确的策略选择                | 教师、学生采用相同动作 tokenizer 和归一化    |
| Octo，2024                     | 模块化 Transformer 与 diffusion 动作头，支持适配新观测和动作空间                | 连续策略和模块化适配的参考                  | 原生 diffusion 动作头需要额外的概率处理      |
| CrossFormer，2024              | 共享 Transformer，按动作类别使用不同动作头，联合处理 20 类 embodiment           | 跨机器人架构与对照设计                      | 原论文连续 L1 动作头需改成可评分概率头       |
| HPT，2024                      | embodiment-specific stem、共享 trunk、任务或数据源相关 head，扩展到 52 个数据集 | 最直接的异质输入/输出接口参考               | 原论文 Huber 回归输出需替换为概率头          |
| π₀，2024                       | 视觉语言骨干结合 flow matching 动作生成器                                       | 后续连续动作推广的目标架构                  | 不能把速度场回归损失直接当成动作 log-prob    |
| FAST，2025                     | DCT 与 BPE 压缩动作块，支持多种动作维度和频率                                   | 保留离散概率接口并处理高频动作              | 统一 codec，同时保留机器人、频率与动作元数据 |
| RLDG，2024 / RSS 2025          | 用 RL 专家生成的数据改进通用机器人策略                                          | “专家产生数据，再训练 generalist”的重要基线 | 原方法的数据生成与学生闭环 OPD 分别比较      |
| VLA-OPD，2026-03               | 学生执行环境轨迹，教师反馈，反向 KL 蒸馏                                        | 最直接的算法 baseline                       | 先确认动作概率接口与可复现实现               |
| FastOPD，2026-10               | 在学生生成过程上监督 flow-map，压缩模型与采样步数                               | 连续生成器蒸馏的近期参考                    | 去噪过程 on-policy 和环境 on-policy 分别定义 |
| DriftOPD，2026-09              | kernel-smoothed 分布匹配与离线 critic，面向一步动作策略                         | 连续动作与长时序目标的邻近工作              | 其离线、无环境 rollout 设置单独报告          |

上述方法和规模分别来自论文或作者项目页。[^oxe][^openvla][^octo][^crossformer][^hpt][^pi0][^fast][^rldg][^vla-opd][^fastopd][^driftopd]

### 2.1. VLA-OPD：已经存在的直接 baseline

VLA-OPD 使用经过 RL 训练的 SimpleVLA-RL 作为冻结教师，在学生实际访问的环境状态上提供 token 分布监督。论文表 2 报告，LIBERO 每任务一条示范初始化时，学生平均成功率为 48.9%，纯蒸馏后为 87.4%，蒸馏后继续 GRPO 为 93.4%，教师为 93.9%。这组数字支持具身 OPD 的可行性；93.4% 对应额外 RL 阶段，不能全部归因于蒸馏。[^vla-opd]

RoboTwin 2.0 的实验设置不同。论文只选择四个双臂任务，学生以每任务 1,000 条示范初始化，平均成功率从 45.2% 提升到 71.1%，教师为 74.0%。它证明了所选双臂任务上的改进，尚不能据此判断整个 RoboTwin 任务集、多机器人联合训练或者新机器人的零样本迁移。[^vla-opd]

截至本文核对时，作者项目页的代码入口标为 “Code (Coming Soon)”。论文采用 OpenVLA-OFT 命名，同时算法写出 token logits 与 log-prob。OpenVLA-OFT 的最优配方包含连续 L1 动作头，因此复现时必须确认作者究竟使用哪个动作头，以及如何得到概率分布。已有论文证据与已验证的可运行开源 baseline 是两项不同交付。[^vla-opd-project][^oft]

论文关于反向 KL 优于前向 KL、硬标签的结果应在其任务和配置内解释。一般的反向 KL 没有统一有限上界；当教师对学生有正质量的动作赋零概率时，KL 可以发散。其局部固定测度最优解也可以等于教师，因而不能从 KL 方向推出“自动过滤教师错误”或“自动保持动作多样性”。本项目需要自己测量熵、动作覆盖与成功率。

### 2.2. HPT 与 CrossFormer：表示统一已经有可借用的方案

HPT 将各机器人本体感知与视觉输入映射成固定维度的少量 latent tokens，使用共享 Transformer trunk，再由各自 head 产生动作。原论文用归一化动作的 Huber 回归训练，证明的是异质表示预训练与迁移，而非基于概率比率的 MirrorOPD。借鉴其模块化接口后，要显式引入离散或连续概率输出头。[^hpt]

CrossFormer 通过模态 tokenizers 和动作 readout tokens 处理不同输入、不同动作维度，用共享骨干与不同动作类别的输出头联合训练。原论文包含单臂、双臂、导航与四足控制，说明物理动作无需全部映射为同一个七维向量。其连续动作 L1 回归头同样不能直接提供现有 MirrorOPD 所需的 log-prob。[^crossformer]

这两篇工作支持的架构选择是：共享能跨任务学习的部分，保留机器人相关的传感器接口和控制输出。它们也给出更严格的基线要求：MirrorOPD 必须与同架构、同数据混合的普通联合训练比较，才能将收益归因于优化方法。

### 2.3. FAST：统一编码不等于统一物理含义

FAST 先对连续动作块做离散余弦变换（Discrete Cosine Transform，DCT），再用字节对编码（Byte Pair Encoding，BPE）压缩。FAST+ 是在大规模真实机器人动作上训练的通用 tokenizer。它为高频、双臂动作保留自回归概率接口，是第二阶段值得采用的路线。[^fast]

codec 统一后，同一个 token 在不同机器人条件下仍可对应不同控制量。策略必须知道动作维度次序、控制频率、坐标系、归一化统计和动作模式。FAST 的贡献在于动作序列编码；跨 embodiment 的可执行语义仍由条件信息和控制接口确定。

### 2.4. 两类 on-policy 与近期连续动作蒸馏

具身中的 on-policy 至少有两个不同层次。

1. 环境轨迹 on-policy：学生动作真正进入环境，下一帧视觉和机器人状态由该动作产生。VLA-OPD 属于这一层次。
2. 动作生成过程 on-policy：在给定环境观测上，学生生成自己的带噪动作或去噪中间状态，再由教师监督生成器。FastOPD 重点处理这一层次。

第二层次能减小动作生成器内部的分布偏移，但只用静态示范观测时，环境状态分布仍来自离线数据。FastOPD 不能直接替代“学生把物体碰偏后，教师教它如何恢复”的环境闭环基线。[^fastopd]

DriftOPD 使用核平滑动作分布匹配与离线 critic，目标涉及动作块的未来影响。论文明确采用无需在线环境 rollout、无需独立教师的设置。它适合帮助分析连续密度难题和长时序信用分配，但其数据访问条件与本仓库冻结教师、环境 rollout 的设定不同。[^driftopd]

## 3. 需要统一的接口

### 3.1. 区分异质性的来源

| 异质性     | 例子                                     | 必须明确的接口                              | 第一版处理方式                     |
| ---------- | ---------------------------------------- | ------------------------------------------- | ---------------------------------- |
| 任务与场景 | 抓杯子、开抽屉；桌面材质变化             | 任务指令、成功条件、episode 划分            | 同机器人多任务，先检验优化差异     |
| 观测       | 单相机、多相机；不同 proprioception 维数 | 模态标识、时间戳、缺失 mask、输入适配器     | 共享视觉编码，保留输入适配器       |
| 控制       | 七维末端增量、十四维双臂关节目标         | 动作模式、维度顺序、单位、坐标系、有效 mask | 各 embodiment 保留本地动作头       |
| 时间       | 10 Hz 与 50 Hz；不同 action chunk        | 时间间隔、预测长度、实际执行长度            | 第一阶段固定，第二阶段显式条件化   |
| 数据质量   | 人类示范、RL 专家、失败轨迹              | 来源、质量标签、教师与采样策略版本          | 分层采样，分别报告效果             |
| 监督模态   | 人类视频没有机器人控制标签               | 可用监督种类、动作缺失标识                  | 作为独立辅助目标，先不计入动作 OPD |

episode 容器可以统一为观测、指令、机器人状态、动作、终止信号和 metadata。进一步要统一的是这些字段的含义。例如两份数组都长为七，不代表它们都在机器人末端坐标系表达米制位移；一个数据集的夹爪值越大代表张开，另一个可以采用相反约定。

控制频率尤其容易造成隐藏差异。若标签为速度，固定速度在 20 Hz 与 50 Hz 下单步积分产生不同位移；若标签为位置增量，同一数值在不同周期下对应不同实际速度。chunk 长度也必须结合控制周期解释，不能只对齐张量形状。

### 3.2. 推荐的共享策略

对 embodiment $e$，采用

$$
\pi_\theta^e(\cdot\mid h,g)
=H_{\theta,e}\!\left(F_\theta(E_{\theta,e}(h,g),m_e)\right).
$$

其中 $h$ 是观测和动作历史，$g$ 是任务条件，$E_e$ 是观测适配器，$F$ 是共享主干，$H_e$ 是输出本地动作分布的头，$m_e$ 是机器人与控制模式的 metadata。教师与学生可以有不同网络结构；概率比较要求它们在该 embodiment 内采用相同的动作变量和条件输入语义。

不同 embodiment 之间只共享参数及 latent 接口。Franka 的动作分布与双臂关节动作分布各自参与对应任务的损失，不直接互相计算 KL。对于多个动作模式相同的机器人，可以共享动作头并加入 embodiment 条件；其收益通过后续消融判断。

### 3.3. 不同教师的处理

第一版使用按机器人和任务明确路由的教师 $T_{e,j}$，避免将跨机器人语义对齐、教师选择与优化目标同时改动。教师须能在学生到达的历史上评分；只拥有教师离线示范不足以实现这条在线路径。

如果同一任务有多个教师，且它们共享完全相同的动作表示，可以在后续比较分布混合与 log-prob 聚合。两者目标不同：算术混合可以保留各教师的不同模式，几何聚合偏向教师共同支持的动作，支持集不相交时甚至无法归一化。跨机器人教师首先需要动作映射或本地教师，直接平均 logits 没有定义。

## 4. MirrorOPD 如何接到具身策略

### 4.1. 本地条件优化目标

沿用本仓库固定测度语义。在 outer step $k$ 冻结每个 embodiment 的 anchor $\pi_k^e$，在对应环境中执行它，得到观测历史经验测度 $\widehat d_k^e$。教师 $\pi_T^e$ 在本轮内冻结。对固定历史 $h$ 和离散动作变量 $a\in\mathcal A_e$，定义

$$
r_k^e(h,a)=\log\frac{\pi_T^e(a\mid h)}{\pi_k^e(a\mid h)}.
$$

本文暂将任务、控制频率等条件收入 $h$，以简化记号。需要对教师、anchor 的正概率支持作显式约定；实际 logits mask 与采样温度也属于策略定义。

对应条件 mirror 子问题为

$$
\max_{p\in\Delta(\mathcal A_e)}
\left\{\mathbb E_p[r_k^e]
-\frac1\eta\operatorname{KL}(p\Vert\pi_k^e)\right\}.
$$

固定 $e,h$ 并省略条件，令

$$
Z=\sum_{a\in\mathcal A_e}\pi_k(a)e^{\eta r_k(a)},
\qquad
q(a)=\frac{\pi_k(a)e^{\eta r_k(a)}}{Z}.
$$

若动作集合有限且 anchor、教师处处为正，则 $Z>0$ 且有限。代入 $r_k$ 得到

$$
q(a)=\frac{\pi_k(a)^{1-\eta}\pi_T(a)^\eta}{Z}.
$$

逐项展开 KL：

$$
\begin{aligned}
\operatorname{KL}(p\Vert q)
&=\sum_a p(a)\left[\log p(a)-\log\pi_k(a)-\eta r_k(a)+\log Z\right]\\
&=\operatorname{KL}(p\Vert\pi_k)-\eta\mathbb E_p[r_k]+\log Z.
\end{aligned}
$$

因此

$$
\mathbb E_p[r_k]-\eta^{-1}\operatorname{KL}(p\Vert\pi_k)
=\eta^{-1}\log Z-\eta^{-1}\operatorname{KL}(p\Vert q).
$$

KL 非负，且在 $p=q$ 时等于零，故 $q$ 是唯一最优解。这个结论分别在每个 embodiment 的本地动作空间内成立，完全不需要把不同机器人的动作维数设成相同。

对于 $0<\eta<1$，$q$ 是 anchor 和教师的几何插值；$\eta=1$ 时 $q=\pi_T$。插值约束的是固定历史上的分布目标。神经网络有限更新后的实际变化、其他任务性能和闭环轨迹表现仍需要测量。

### 4.2. 联合参数化训练

用固定的 embodiment 权重 $w_e$，$\sum_e w_e=1$，定义联合经验测度

$$
\widehat\mu_k(e,h,a)
=w_e\widehat d_k^e(h)\pi_k^e(a\mid h).
$$

将仓库的 Mean、BPO、MDPO 内核应用到各自本地概率上，再在 $\widehat\mu_k$ 下聚合。三个实施要点分别是：

1. rollout 的 anchor、教师、采样权重和教师相关 target 在 outer 内冻结；student 概率在各 inner step 重算。
2. Mean/BPO 的回归测度来自 anchor，MDPO 的 student 测度通过匹配的采样或重要性比率实现，不能把两种权重混写。
3. inner 结束后 hard sync，再重新进入环境生成 rollout；旧 rollout 保留作 replay 时另定义混合目标与采样纠正。

PMD-Part 的残差为

$$
R_{\mathrm{Part}}^e
=\log\frac{\pi_\theta^e}{\pi_k^e}-\eta r_k^e+\log Z_k^e.
$$

PMD-Mean 用冻结的 teacher/anchor KL 中心替代 partition 项，BPO 进一步中心化完整残差。现有主稿已证明 Mean 的总体最优解一般不等于 Gibbs 几何插值。因此实验结果必须标明具体内核，不能将上节 Gibbs 解当成所有 MirrorOPD 变体的精确目标。

采用 autoregressive 动作 token 时，$h$ 还包含当前动作块内的 token prefix，$a$ 表示下一个 token。采用整个动作块作为 $a$ 时，其空间是全部动作块；词表全求和的便利不再存在。第一版按 token 使用现有内核，同时固定动作块长度、执行长度与 prefix 权重。

### 4.3. 为什么异质联合训练更好仍然是待验证命题

共享主干的更新梯度写为

$$
\nabla_\theta\mathcal L=\sum_e w_e g_e,
\qquad g_e=\nabla_\theta\mathcal L_e.
$$

当 $g_e^\top g_f<0$ 时，两类数据在共享参数上的局部更新方向冲突。固定锚点会改变各任务的训练目标与更新幅度，但它没有强制 $g_e^\top g_f\ge0$。对联合损失下降的保证也不等于每项损失或每个机器人成功率同时改善。

MirrorOPD 的可检验动机在于：限制一次 outer 内偏离当前策略的程度，并利用固定 rollout 上的多次更新，提高每次昂贵环境交互与教师查询的利用率。如果主要瓶颈是表示错误、教师失效或持续负梯度冲突，这个机制不能独立解决问题。

不同数据源的动作维度和 token 长度会改变损失尺度。直接把所有 token 混在一起求平均，会让长动作块或高维机器人得到更多权重。实验应预先规定按 episode、控制决策或 token 哪个单位加权，报告实际 $w_e$；若用 task-wise 缩放，需承认它改变了联合目标或有效 mirror 温度。

### 4.4. 连续动作与 flow 路线

对具有共同基准测度的连续密度，上面的求和可以改成积分，但必须检查归一化积分存在、共同支持集以及所需矩。普通确定性 L1 回归输出不能直接形成这个密度比；diffusion/flow 的训练误差也不是终端动作 log-density。

有三条实施路线：

1. 离散动作路线：OpenVLA 风格 categorical action head 或 FAST，先复用现有 token 内核，明确量化误差与控制性能。
2. 显式密度路线：Gaussian、mixture density 或可评估密度的 flow，推导连续版目标，检查多模态动作表达能力；不能给确定性教师随意加噪后宣称目标保持不变。
3. 生成过程路线：对随机去噪 transition 的概率定义 mirror step，或对 velocity/score 采用有单独理论依据的目标。确定性 ODE transition 的 Dirac 分布没有通常的有限 transition KL，必须先建立适用的概率模型。

第二、三条路线会同时改变动作模型与理论假设，适合在离散路线得到清楚结论后开展。FastOPD 和 DriftOPD 是这一阶段的关键邻近文献。[^fastopd][^driftopd]

## 5. 最小可行实验与正式 benchmark

### 5.1. 开始前固定的问题与反例

第一版回答两个问题：MOPD 的概率接口与缓存同步能否正确迁移到闭环机器人策略；在相同任务混合与预算下，它能否比普通 OPD 提高成功率或达到同成功率所需的交互效率。

以下结果会否定或收窄当前动机：只改善教师 KL 而降低闭环成功率；收益完全由更大的 teacher 查询或 inner FLOPs 解释；联合训练收益与普通 OPD 无差别；仅平均任务改善而少数机器人持续退化；必须改变架构才能观察到优势。出现这些结果后应按证据修改研究命题。

### 5.2. 阶段零：概率和闭环接口验证

选择一个可执行的机器人任务和一个概率明确的 student/teacher 对，检查以下事项：

1. 学生动作真正作用于环境，教师接收对应的新观测和同一动作 token prefix。
2. action tokenizer、归一化统计、维度顺序、mask、采样温度和执行控制器一致。
3. exact-vocab、采样估计两条梯度在小 batch 上一致到预定容差；本体感知或任务条件改变时教师输出确实变化。
4. teacher、anchor 与 target 在 inner 内不变；student log-prob 更新；outer hard sync 后重新 rollout。
5. 分别记录动作执行次数、教师 forward 次数、生成 token 数、优化 FLOPs 估计与耗时。

阶段零只建立实现可执行与语义正确的证据，不报告 generalist 或跨 embodiment 结论。

### 5.3. 阶段一：同一 embodiment 上的任务异质性

先在 LIBERO 上固定一种机器人和动作概率头。开发集选六个任务，覆盖不同对象、空间关系和长程依赖；用两个训练种子、每任务 50 个固定评估初态，比较 SFT 初始化、普通 OPD 与 MOPD。每方法每 checkpoint 共 600 个评估 episode。这是排查接口和选择学习率范围的开发实验。

正式比较扩展到 LIBERO 的四套主要任务集合，至少五个训练种子、每任务每种子 100 个固定评估初态。任务划分、演示条数、初始化 checkpoint 与教师先冻结，开发任务不用于最终超参数胜负判定。不要通过选取各方法不同的最好 checkpoint 抬高结果；预先固定预算终点和 checkpoint 选择规则。

这一阶段可以判断优化方法是否有研究价值，但 LIBERO 四套任务本身不构成跨 embodiment 证明。

### 5.4. 阶段二：真正的跨 embodiment 联合训练

采用支持多机器人配置的仿真平台，在可兼容的任务和控制器上选择三种机器人，每种十个任务，训练一个带本地动作头的共享学生。robosuite 支持多机器人配置，但具体机器人、任务和控制器组合先通过阶段零检查。[^robosuite]

正式评估使用五个训练种子，每个机器人任务组合 100 个固定初态，每方法每 checkpoint 为 15,000 个 episode。先做两个机器人、各三个任务的开发实验，确认教师和控制器有效，再确定完整任务清单。

RoboTwin 2.0 的双臂任务可以作为动作维度变化的补充验证。只有实际联合训练不同机器人，并按机器人分别评估，才使用“跨 embodiment 联合训练”的结论。把单臂模型和双臂模型分别训练成功，不能代替这个证据。

若测试未见过的机器人，需将其独立列为 few-shot adaptation 设置：固定适配数据预算，重新初始化或适配输入和输出接口。留出机器人零样本迁移是更强的命题，第一版不预设达到这一能力。

### 5.5. 基线与预算

| 对照                                            | 检验的问题                     | 公平性约束                               |
| ----------------------------------------------- | ------------------------------ | ---------------------------------------- |
| 共享架构联合 BC/SFT                             | 收益是否来自共享表示和混合数据 | 与 MOPD 使用同样的主干、动作头与初始化   |
| 学生状态上教师标注的 DAgger / soft distillation | 收益是否仅来自学生状态覆盖     | 匹配环境交互和教师查询；明确硬/软监督    |
| 普通 reverse-KL OPD                             | mirror 目标的独立增益          | 相同 rollout、教师、概率表示、任务混合   |
| MOPD-Mean / BPO / MDPO                          | 收益来自哪个 inner 目标        | 超参数搜索范围与搜索预算一致             |
| 分机器人专用学生                                | 共享参数带来的正迁移和负迁移   | 同时报单模型与整组模型参数量、总训练预算 |
| RL 专家数据蒸馏                                 | 学生访问测度的作用             | 明确教师采样成本与离线数据量，参考 RLDG  |

DAgger 的核心是学生诱导状态上的专家标注与数据聚合；它可以用于连续动作，并不等同于某一种离散硬标签交叉熵。实现时将状态采集方式、标签形式和优化目标分开。[^dagger]

预算至少采用两种比较：匹配环境交互与教师查询，检验昂贵数据的利用率；匹配总训练算力或 wall-clock，检验实际效率。MOPD 使用更多 inner 更新时，两种预算不必同时相等，因此分别给出曲线。teacher 的预训练或 RL 成本单列，不能当作免费获得的能力。

在阶段零测得运行代价后，冻结三个交互预算点及相应 teacher 查询上限。开发实验用于决定是否能执行完整规模；正式实验前把数值、任务列表、种子、超参数搜索次数和终止规则写入配置，之后按此执行。本文提出 benchmark 结构，尚未运行这些训练。

### 5.6. 指标与判定规则

主要指标是闭环任务成功率、按机器人宏平均和最差机器人表现。辅助指标包括达到目标成功率的环境动作数、教师查询数、耗时、动作熵、执行平滑性、约束违反次数、teacher KL、anchor KL 和缓存复用率。跨任务共享主干的梯度夹角作为解释性指标，不当作成功率的替代物。

遗忘指标在训练前后使用相同的留出任务和初态计算。迁移指标在相同训练预算下比较联合学生与仅目标机器人数据训练的学生。测试集按 episode、场景及数据源隔离，不能把同一轨迹的相邻帧拆到训练与测试两边。

预设的研究推进门槛是：在匹配交互和教师预算下，相对普通 OPD 的成功率提高至少五个百分点，或者达到相同成功率所需的交互减少至少 20%；同时检查机器人分组差异，不接受平均改进掩盖超过三个百分点的单机器人退化。以上数值是本项目的决策门槛，不是文献结论，也不是对结果的预测。

用训练种子作为独立重复单位报告均值和置信区间，评估初态在方法之间配对。任务和初态存在分层结构时采用分层重采样；不能将所有 episode 当成相互独立的训练重复。将效应大小、区间与预设门槛共同用于决定是否进入下一阶段。

## 6. 推荐路线与贡献边界

推荐首先实现“离散动作概率头 + 共享 VLA 主干 + 固定 rollout 的 MirrorOPD”，在单机器人多任务上与普通 OPD、DAgger 比较。拿到闭环成功率和预算曲线后，再加入 HPT/CrossFormer 式本地动作接口，检验跨 embodiment 联合训练。

数据接入优先覆盖有动作标签、可查询教师和可执行环境的机器人任务。人类视频可以用于视觉、语言和表示辅助预训练；缺少机器人动作与本地教师时，将它直接混进动作 log-ratio 目标缺乏定义。跨模态辅助目标的收益应通过独立消融归因。

有说服力的最终贡献应同时给出固定测度下的清楚目标、真实闭环 OPD 对照、多机器人联合训练的分组结果，以及环境和教师预算上的收益。若离散路线有效，再推广到连续动作的概率或去噪过程目标；这一推广包含新的数学与实现工作。

### 参考文献

[^oxe]: Open X-Embodiment Collaboration. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv 2023 / ICRA 2024. [arXiv](https://arxiv.org/abs/2310.08864).

[^openvla]: Moo Jin Kim et al. OpenVLA: An Open-Source Vision-Language-Action Model. arXiv 2024. [arXiv](https://arxiv.org/abs/2406.09246).

[^octo]: Octo Model Team et al. Octo: An Open-Source Generalist Robot Policy. arXiv 2024. [arXiv](https://arxiv.org/abs/2405.12213).

[^crossformer]: Ria Doshi et al. Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation. arXiv 2024. [arXiv](https://arxiv.org/abs/2408.11812).

[^hpt]: Lirui Wang, Xinlei Chen, Jialiang Zhao, Kaiming He. Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers. NeurIPS 2024. [arXiv](https://arxiv.org/abs/2409.20537).

[^pi0]: Kevin Black et al. π₀: A Vision-Language-Action Flow Model for General Robot Control. arXiv 2024. [arXiv](https://arxiv.org/abs/2410.24164).

[^fast]: Karl Pertsch et al. FAST: Efficient Action Tokenization for Vision-Language-Action Models. arXiv 2025. [arXiv](https://arxiv.org/abs/2501.09747).

[^rldg]: Charles Xu et al. RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning. arXiv 2024 / RSS 2025. [arXiv](https://arxiv.org/abs/2412.09858).

[^vla-opd]: Zhide Zhong et al. VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation. arXiv 2026. [arXiv](https://arxiv.org/abs/2603.26666).

[^vla-opd-project]: IRPN Lab. VLA-OPD 项目页. 核对日期 2026-10-09. [项目页](https://irpn-lab.github.io/VLA-OPD/).

[^oft]: Moo Jin Kim, Chelsea Finn, Percy Liang. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. arXiv 2025. [arXiv](https://arxiv.org/abs/2502.19645).

[^fastopd]: Yoojin Oh et al. FastOPD: On-Policy Distillation for Lightweight VLA Deployment. arXiv 2026. [arXiv](https://arxiv.org/abs/2610.02832).

[^driftopd]: Youngjun Jun et al. DriftOPD: Sequence-Level Reverse-KL Distillation for One-Step VLA Policies. arXiv 2026. [arXiv](https://arxiv.org/abs/2610.00317).

[^dagger]: Stéphane Ross, Geoffrey Gordon, Drew Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011. [arXiv](https://arxiv.org/abs/1011.0686).

[^robosuite]: robosuite contributors. Robots. 官方文档，核对日期 2026-10-09. [Robots](https://robosuite.ai/docs/modules/robots.html).
