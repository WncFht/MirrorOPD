"""Render the single maintained OneShot/Mirror experiment report from evidence."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.analysis.oneshot_effective_time import (
    audit_completed_arm,
    audit_evaluation,
    collapse_statistics,
    history,
)
from training.oneshot_suite import selected_experiment_plan

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "runs/oneshot-effective-time-20261012"
OUT = ROOT / "results/oneshot-effective-time/consolidated"
ARCHIVE = ROOT / "data/oneshot-report-sources-20261013"
REPORT = ROOT / "docs/experiments/2026-10-13-OneShot与镜像蒸馏实验报告.md"
ALIASES = {
    "2026-10-12-OneShot与内层KL收敛核验.md": "kl-semantics",
    "2026-10-12-OneShot长期吸收与有效时间消融设计.md": "registered-scope",
    "2026-10-12-OneShot长期吸收实验进展.md": "frozen-lr",
    "2026-10-13-FP32消融结果与后续实验调整.md": "fp32-fixed",
    "2026-10-13-OneShot学习率上限的历史数据验证.md": "historical-lr",
    "2026-10-12-四卡数学蒸馏实验.md": "math-full",
    "2026-10-12-四卡数学蒸馏完整比较.md": "math-full",
    "2026-10-12-数学OPD五十步训练结果.md": "math-onpolicy",
    "2026-10-12-数学MDPO五十轮训练结果.md": "math-mdpo",
    "2026-10-12-数学BPO五十轮训练结果.md": "math-bpo",
}


def source(name):
    """Recover the exact superseded report without keeping duplicate Markdown."""
    raw = gzip.decompress((ARCHIVE / (name + ".gz")).read_bytes())
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    expected = next(r["sha256"] for r in manifest["sources"] if r["path"].endswith(name))
    assert hashlib.sha256(raw).hexdigest() == expected
    return raw.decode()


def section(text, heading, following=None):
    """Select a checked section boundary from a frozen source report."""
    start = text.index(heading)
    end = text.index(following, start + len(heading)) if following else len(text)
    return text[start:end].strip()


def below(text):
    """Move imported sections one level below the maintained experiment heading."""
    return "\n".join("#" + line if line.startswith("##") else line for line in text.splitlines())


def figure(fig, name):
    """Save shareable raster and vector figures from the same observations."""
    for ext in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, metadata=metadata)
    plt.close(fig)


def observations(config):
    """Read only committed, deduplicated updates from one arm."""
    return history(SUITE / config["id"] / "updates.jsonl", "update_before")


def full_rows(identifier):
    """Read full-vocabulary native readouts on the fixed 4096-prefix reservoir."""
    return history(SUITE / identifier / "fixed-native.jsonl", "update")


def distance(arm, step):
    """Use the post-final readout when the requested step is the terminal step."""
    return arm["distance"][arm["steps"].index(step)]


def draw(bf, fp_config, evaluations):
    """Plot complete learning-rate arms, stopped precision data and benchmarks."""
    colors = ("#2166ac", "#4d9221", "#b2182b")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for arm, color in zip(bf, colors, strict=True):
        for ax, field in zip(axes, ("steps", "effective_time"), strict=True):
            ax.plot(arm[field], arm["distance"], color=color, label=f"lr={arm['config']['lr']:g}")
            ax.set_yscale("log")
            ax.set_ylabel("Mean absolute action log-prob gap (395862 fixed actions)")
            ax.grid(alpha=0.2)
            ax.set_xlim(left=0)
    axes[0].set_xlabel("Completed optimizer updates")
    axes[1].set_xlabel("Effective time u = updates * lr / 1e-6")
    axes[0].legend()
    fig.suptitle("Fixed responses, seed 42: three completed BF16 runs")
    figure(fig, "frozen-effective-time")
    fp = observations(fp_config)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    stop = fp[-1]["update_before"]
    for identifier, label, color in (
        (bf[0]["config"]["id"], "BF16 training", "#2166ac"),
        (fp_config["id"], "FP32 training (stopped)", "#b2182b"),
    ):
        rows = history(SUITE / identifier / "updates.jsonl", "update_before")
        selected = [r for r in rows if r["update_before"] <= stop]
        axes[0].plot(
            [r["update_before"] for r in selected],
            [r["metrics"]["current_advantage_abs/mean"] for r in selected],
            label=label,
            color=color,
        )
        rows = [r for r in full_rows(identifier) if r["update"] <= stop]
        axes[1].plot(
            [r["update"] for r in rows],
            [r["metrics"]["student_teacher_full_kl"] for r in rows],
            label=label,
            color=color,
        )
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Completed optimizer updates")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Mean absolute gap (all fixed actions)")
    axes[1].set_ylabel("Student-to-teacher KL (4096 fixed prefixes)")
    fig.suptitle("Matched lr=5e-7: native training precision, common observed interval")
    figure(fig, "precision-comparison")
    tasks = ("MATH-500", "AMC23", "AIME25")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for arm, color in zip(bf, colors, strict=True):
        config = arm["config"]
        ids = ["initial-s42"] + [config["id"] + f"-update{t}" for t in config["checkpoint_steps"]]
        dose = [0] + [config["lr"] / 1e-6 * t for t in config["checkpoint_steps"]]
        for col, task in enumerate(tasks):
            for row, key in enumerate(("accuracy", "budget_exhausted_fraction")):
                axes[row, col].plot(
                    dose,
                    [100 * evaluations[i][task][key] for i in ids],
                    marker="o",
                    color=color,
                    label=f"lr={config['lr']:g}",
                )
    for col, task in enumerate(tasks):
        axes[0, col].axhline(
            100 * evaluations["teacher"][task]["accuracy"],
            color="gray",
            linestyle="--",
            label="Teacher",
        )
        axes[0, col].set_title(task)
        for row in range(2):
            axes[row, col].grid(alpha=0.2)
            axes[row, col].set_xlabel("Effective time u")
            axes[row, col].set_ylim(0, 100)
    axes[0, 0].set_ylabel("Greedy accuracy (%)")
    axes[1, 0].set_ylabel("7168-token budget exhausted (%)")
    axes[0, 0].legend(fontsize=7)
    figure(fig, "frozen-benchmarks")
    with (ROOT / "results/oneshot-inner-audit/inner-history.csv").open() as handle:
        inner = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for row, method in enumerate(("mdpo", "bpo")):
        for outer, color in zip((0, 4, 49), colors, strict=True):
            rows = [r for r in inner if r["method"] == method and int(r["outer_call"]) == outer]
            x = [int(r["inner_step"]) for r in rows]
            for col, field in enumerate(("anchor_student_tail_kl", "grad_norm")):
                axes[row, col].plot(
                    x,
                    [float(r[field]) for r in rows],
                    color=color,
                    marker=".",
                    label=f"outer={outer}",
                )
                axes[row, col].grid(alpha=0.2)
                axes[row, col].set_xlabel("Completed inner updates at measurement (pre-update)")
                axes[row, col].set_title(method.upper())
                axes[row, col].legend(fontsize=8)
        axes[row, 0].set_ylabel("Tail-aware anchor-to-student KL")
        axes[row, 1].set_ylabel("Gradient norm")
    figure(fig, "formal-inner-windows")


def main():
    """Write one report; the archived source notes remain data, not active reports."""
    OUT.mkdir(parents=True, exist_ok=True)
    plan = selected_experiment_plan(SUITE)
    bf = [
        audit_completed_arm(SUITE, c)
        for c in plan
        if c["method"] == "opd" and c["rollout"] == "frozen" and c["lr"] > 0
    ]
    assert len(bf) == 3
    assert all(arm is not None for arm in bf)
    fp_config = next(
        c
        for c in json.loads((SUITE / "plan.json").read_text())
        if c["id"] == "frozen-opd-fp32-lr5em7-s42"
    )
    fp = observations(fp_config)
    evaluations = {
        p.parent.name: audit_evaluation(p.parent)
        for p in (SUITE / "evaluation").glob("*/completion.json")
    }
    q = np.array([0.5, 0.5])
    teacher = np.array([0.9, 0.1])
    target = np.exp(0.7 * np.log(q) + 0.3 * np.log(teacher))
    target /= target.sum()
    assert np.allclose(target, [0.659073, 0.340927], atol=1e-6)
    assert abs(np.sum(target * np.log(target / teacher)) - 0.212805) < 1e-6
    draw(bf, fp_config, evaluations)
    collapse = collapse_statistics(bf)
    current_completed = [a for c in plan if (a := audit_completed_arm(SUITE, c)) is not None]
    current_updates = sum(a["config"]["updates"] for a in current_completed)
    expected_evaluations = {"teacher", "initial-s42"} | {
        c["id"] + f"-update{t}" for c in plan for t in c["checkpoint_steps"]
    }
    complete_evaluations = len(expected_evaluations & evaluations.keys())
    state = json.loads((SUITE / "status.json").read_text())
    online_id = "online-opd-bf16-lr5em7-s42"
    online_config = next(c for c in plan if c["id"] == online_id)
    online = observations(online_config)
    online_progress = json.loads((SUITE / online_id / "progress.json").read_text())
    audit = source("2026-10-12-OneShot与内层KL收敛核验.md")
    formal = source("2026-10-12-四卡数学蒸馏完整比较.md")
    progress = source("2026-10-12-OneShot长期吸收实验进展.md")
    stability = source("2026-10-13-OneShot学习率上限的历史数据验证.md")
    parts = []
    add = parts.append
    add("# OneShot与镜像蒸馏实验报告")
    add(
        """术语沿用现有实现：同策略蒸馏（On-Policy Distillation，OPD）、镜像下降策略优化（Mirror Descent Policy Optimization，MDPO）；BPO是仓库中基于中心化对数比的方差代理，目标公式在正文定义。KL指Kullback–Leibler散度，BF16指bfloat16浮点格式，FP32指单精度浮点数。外层迭代outer建立新rollout及锚点，内层更新inner在该缓存上执行一次全batch optimizer更新。"""
    )

    add(f"""本报告是这组实验的唯一维护入口，合并原10份设计、进展和结果文档。正文按实验设置、结果与分析组织；更新结果直接修改本报告。配置、原始日志、校验记录和可导出图保留在data、results与runs，原文快照保存在[data附件清单](../../data/oneshot-report-sources-20261013/manifest.json)。

## 结论与当前进度

固定监督的长期吸收已经在seed42上重现。标准BF16学习率条件从100步到200、300步的动作绝对差比值为0.556和0.420。三个学习率按有效时间对齐后，对数距离离散度降低{100 * collapse["dispersion_reduction"]:.1f}%，支持近似时间缩放，曲线仍有可见偏差。

原四卡MDPO/BPO的小student center度量本轮锚点到学生的距离，不能代表学生到教师的距离。首轮内层仍有非零梯度和持续变化；原日志不足以判定内层目标已经达到最优。FP32改变拟合速度，却没有把固定监督吸收压缩到首次更新。固定单题拟合改善也没有带来这组数学测试的准确率提升。

当前只执行seed42：12个完整条件中{len(current_completed)}个完成，完成{current_updates}/4308次计划更新；另保留{len(fp)}条已停止FP32实验的更新记录。在线半倍学习率条件在数据快照中完成{online_progress["updates_completed"]}/600步。正式评测已完成{complete_evaluations}/44组，每组570题，预检的额外评测另列。后续FP32学习率扫描及seed142、242均已取消。

| 实验组 | 已执行条件与状态 | 主要证据 |
| --- | --- | --- |
| 历史t22对照 | online、frozen各300步 | 固定响应仍在100–300步继续拟合 |
| 历史多题学习率 | 六个学习率各279条历史记录，CPU复算完成 | 低学习率近似时间缩放；1e-4末段反弹 |
| 原四卡数学训练 | 在线OPD 50步；MDPO/BPO各800步，均完成 | 三方法MATH/AMC终点改善；内层小KL不证明目标已解完 |
| 数值与链路验证 | 短响应探针、长响应预检、零更新、CPU读出均完成 | 精度影响成立，短响应结论不能直接外推大batch |
| 新固定监督BF16 | 600、300、150步三条件均完成 | 长程拟合、近似有效时间缩放和完整数学评测 |
| 新固定监督FP32 | 半倍学习率停止，保留330条更新；其余取消 | 拟合更快，仍持续数百步学习 |
| 新在线OPD | 半倍学习率运行中；另外两个待运行 | 逐步刷新响应时的有效时间关系尚未判分 |
| 新镜像生命周期 | 固定锚点和16步重设锚点各两方法，四条件待运行 | 独立测量当前教师距离和镜像目标误差 |

这里的“完成”按原始记录、实际退出与相应评测核验；停止的FP32条件没有原定600步终点，不计为完整条件。""")
    add("## 比较单位与论文观察")
    add(
        section(audit, "## 论文的固定状态实验", "## MD-OPD 的历史实验与代码").replace(
            "## 论文的固定状态实验", "### 论文设置与测量"
        )
    )
    add(r"""### 本报告统一的距离与时间口径

记当前学生为$p_t$、冻结教师为$T$、某轮锚点为$q$。N个固定动作上的平均绝对log-prob差为

$$
d_t=\frac1N\sum_{i=1}^N|\log T(a_i\mid s_i)-\log p_t(a_i\mid s_i)|.
$$

当前学生到教师的完整词表KL在M个固定前缀上独立计算：

$$
D_t=\frac1M\sum_{j=1}^M\sum_a p_t(a\mid s_j)\log\frac{p_t(a\mid s_j)}{T(a\mid s_j)}.
$$

有符号动作差在正负动作之间抵消，而且冻结动作池来自初始$q_0$。其后续期望是$\mathbb E_{q_0}\log(p_t/T)$，不能直接当作$\mathbb E_{p_t}\log(p_t/T)$。全文把动作绝对差、有符号采样量、粗粒化锚点KL、完整词表KL与任务准确率分别报告。

$t$只计真实optimizer更新。新实验的update_before$t$表示已经完成$t$次更新；正式数学日志的inner_step也是测量前已完成的本轮更新数，因此inner_step15没有包含第16次更新后的同批独立读出。历史M4的step写在更新后，reward在更新前评分，其主时钟使用$t-1$。""")
    add("## 历史t22固定与在线监督对照")
    add(
        section(audit, "## MD-OPD 的历史实验与代码", "## 四卡日志的指标语义").replace(
            "## MD-OPD 的历史实验与代码", "### 设置、结果与分析"
        )
    )
    add(
        """历史代码检查确认冻结的是首次rollout的tokens与前缀，当前学生log-prob及advantage每步更新。它不是重复同一个不变梯度，也不是对整个词表执行教师KL的精确梯度。当前checkout与历史配置的对应低于逐文件启动快照证据，原始CSV和metadata是数值结论的来源。"""
    )
    add("""两条t22历史metadata还给出以下共同条件。数据随机种子字段为空，不假定它们与新seed42的响应池逐条一致。

| 历史t22设置 | 存档值 |
| --- | --- |
| AdamW | betas=(0.9,0.999)，weight decay=0.01，clip=1，lr=1e-6 |
| 参数与前向 | FP32 master、BF16前向 |
| 训练采样 | temperature=1、top-p=1、top-k=-1、n=1 |
| prompt / response预算 | 8192 / 7168 token |
| 原运行硬件 | metadata记8 GPU；本报告仅CPU复算这些历史，不重启8卡实验 |

这些条件与新实验的实际初始响应、并行布局和实现快照不同。历史与新实验用于核验持续拟合现象，不要求逐步完全重合。""")
    add("## 四卡多题数学蒸馏比较")
    add(
        section(formal, "## 配置与比较单位", "## 运行修复与交付物")
        .replace("## 配置与比较单位", "### 共同设置")
        .replace("## 全量测试结果", "### 三方法结果")
        .replace("## 教师对齐、reward 与更新稳定性", "### 对齐与稳定性分析")
    )
    add("""### 完整在线OPD

该条件每轮刷新64条响应，50轮共50次更新。MATH-500正确数347→373、AMC23为21→26、AIME25为5→3。前后十轮训练回答预算耗尽率为62.34375%→49.06250%。在各自访问前缀上，teacher center下降46.15%；该比例不是固定前缀完整词表KL的降幅。AIME25曾在outer35达到8/30，终点为3/30，报告采用预定outer50，最终27/30条回答耗尽预算。

### MDPO

该条件每轮刷新响应、冻结当轮锚点，再执行16次更新，50轮共800次。MATH-500正确数353→387、AMC23为19→21、AIME25为5→6。它是三方法中终点MATH和AIME分数最高的运行，但更新量是在线OPD的16倍，且初始化成绩不同。outer10出现分数回落，后续恢复；生产公式、EOS与预算末动作、缓存冻结、token归约和权重同步的核验支持已测试的实现契约。方法结论限于这一组运行，不作为等更新预算的算法优劣排序。

### BPO

该条件与MDPO相同为50轮、800次更新。MATH-500正确数346→376、AMC23为19→24、AIME25为5→4。重要性比率与ESS只用于诊断，不乘入BPO损失。训练回答预算耗尽率49.0625%→39.6875%；唯一裁剪发生在outer_call11、inner_step3，尺度0.480077，下一次梯度范数0.15468。AIME25曾在outer20达到8/30，终点4/30；MATH曾达到77.8%，终点75.2%。周期峰值与预定终点均保留，结论采用outer50。

### 证据与完整轨迹

三组各有初始化及每五轮一次的11次完整570题评测。运行和模型数据、启动源码、日志计数、GPU范围、本地/在线W&B及压缩恢复校验见[完成审计](../../results/training/math-4gpu/final-completion-audit.json)。on-policy启动版本为d89e875f97738d1154fc2449720f9f5fd552ae27，MDPO为87ec5155f4ee4f7904aa275cf580261542350c82，BPO为ac54a612fb9ef0692b5b65d6c7dae620bdbb39a9。在线记录分别为wass1wqb、3o5o72du、gp7pyuj6[^wandb-wass1wqb][^wandb-3o5o72du][^wandb-gp7pyuj6]。

![三方法完整训练和全量测试轨迹](../../results/training/math-4gpu/comparison/comparison.png)

原DAPO有17917行、17255个唯一prompt，最大训练prompt为1493 token，最大测试prompt为796 token。它与MATH-500、AMC23、AIME25的精确题目重叠为2、5、0条；AMC23有12.5%的题在训练语料中出现。字符串审计不能排除其他语义重叠。三组原实验没有逐题配对输出，因此不计算配对显著性。新固定监督评测则保存逐题输出，两类证据不混用。""")
    add("## 指标定义与原内层日志")
    metrics = section(audit, "## 四卡日志的指标语义", "## 镜像内层的目标与收敛").replace(
        "## 四卡日志的指标语义", "### 设置与指标"
    )
    metrics = metrics.replace("mean}(u)", "mean}(v)").replace("u=\\log p", "v=\\log p")
    add(metrics)
    add(r"""尾部分解在$q_R,p_R>0$时可逐步写为

$$
\begin{aligned}
\sum_{a\in R}q_a\log\frac{q_a}{p_a}
&=\sum_{a\in R}q_Rq(a\mid R)
 \left[\log\frac{q_R}{p_R}+\log\frac{q(a\mid R)}{p(a\mid R)}\right]\\
&=q_R\log\frac{q_R}{p_R}\sum_{a\in R}q(a\mid R)
 +q_R\sum_{a\in R}q(a\mid R)\log\frac{q(a\mid R)}{p(a\mid R)}\\
&=q_R\log\frac{q_R}{p_R}
 +q_R\operatorname{KL}(q(\cdot\mid R)\Vert p(\cdot\mid R)).
\end{aligned}
$$

加回H内各项就得到完整KL与粗粒化KL之差。尾部桶只约束总质量，桶内条件分布的差异仍可存在。""")
    add(
        "![原正式MDPO/BPO的单轮内层KL与梯度](../../results/oneshot-effective-time/consolidated/formal-inner-windows.png)"
    )
    target = section(audit, "## 镜像内层的目标与收敛", "## 代码与数据验证")
    target = target.replace("## 镜像内层的目标与收敛", "### 分析：内层目标与教师距离")
    import re

    target = re.sub(r"\bu\b", "v", target)
    add(target)
    add(r"""MDPO的score梯度对应也依赖完整动作期望。q、r在内层固定，且$\nabla p=p\nabla\log p$，所以

$$
\begin{aligned}
\nabla_\theta J
&=\sum_a p_a\left[\frac{1+\log(p_a/q_a)}{\eta}-r_a\right]\nabla_\theta\log p_a\\
&=\mathbb E_p[(v/\eta-r)\nabla_\theta\log p]
 +\frac1\eta\sum_a\nabla_\theta p_a\\
&=\mathbb E_p[(v/\eta-r)\nabla_\theta\log p]\\
&=\mathbb E_q[(p/q)(v/\eta-r)\nabla_\theta\log p].
\end{aligned}
$$

第二行的附加项为$\nabla_\theta\sum_ap_a=0$。停止score系数的梯度得到这条目标梯度，而不是对系数再反向传播。固定动作池的经验梯度与完整q期望的差异另行保留。
""")
    add(r"""BPO的方差梯度可直接验证。令$b=v-\eta r$、$m=\mathbb E_qb$，q和r在内层固定，则

$$
\begin{aligned}
\nabla_\theta\operatorname{Var}_q(b)
&=\nabla_\theta[\mathbb E_qb^2-(\mathbb E_qb)^2]\\
&=2\mathbb E_q[b\nabla_\theta\log p]-2m\mathbb E_q\nabla_\theta\log p\\
&=2\mathbb E_q[(b-m)\nabla_\theta\log p].
\end{aligned}
$$

乘以$1/(2\eta)$即得到正文中的精确BPO系数。生产Tail-aware中心与完整中心的差异是声明的近似；代码公式验证和训练已达到该目标最优，是两个不同命题。""")
    add(
        """逐文件启动指纹核验覆盖losses、cache、metrics、tracking和verl/FSDP适配。两种方法各有800条唯一pre-update记录；每轮teacher center、sampled reward及anchor head mass保持冻结，student center随学生变化。MDPO的800条、BPO存档前416条在线student center与本地逐条相等。完整运行的finished和历史计数由完成审计证明，逐条parity的范围按实际保存的history给出。原日志没有每个inner的当前完整词表教师KL或镜像目标误差，因而不能从小student center推断最佳inner次数。"""
    )
    add("## 数值与训练链路验证")
    add("""### 设置与验证边界

所有新GPU验证使用物理GPU0–3并串行执行。CPU实验不分配GPU。预检检查更新计数、零步对齐、全局有效token归约、教师冻结、缓存生命周期和checkpoint恢复；它们用于验证实验链路，不代替正式训练效果或570题数学评测。

| 验证 | 设置 | 已核验结果 | 分析边界 |
| --- | --- | --- | --- |
| 原在线canary04 | B64、2轮/2更新、初始与最终各570题、BF16 | 正常退出；显存采样峰值35161 MiB | 全量评分链路验证 |
| 原整批canary05 | B64、1更新、整批570题输入 | 正常退出；初始与最终570题完整，峰值33493 MiB | 改变排队批量，保持评分协议 |
| 原MDPO canary | 2轮×16更新；两次16题评测 | 32次更新、同步与缓存冻结通过；峰值35267 MiB | 小样本不做算法优劣判断 |
| 原BPO canary | 同上 | 32次更新、有限诊断与冻结通过；峰值35229 MiB | ESS是诊断 |
| 词表支持检查 | 单卡、12条短响应、额外padding词表列 | student支持外质量约3.04e-9–4.47e-9，teacher约1.01e-10–2.33e-10 | 支持统一在正式训练前完成；小质量不能解释整体学习曲线 |
| tokenizer对照 | 原始模型与保存checkpoint、实际输入token比较 | 已保存的parity记录通过 | 日志warning不代替实际token比较 |

原canary的运行核验文件保存在results/training/math-4gpu，各正式运行从原始学生权重开始，不继承canary已训练参数。数学答案复用boxed核验及最后2000字符的Math-Verify fallback；验证器用于评分，不作为蒸馏梯度中的任务奖励。""")
    add("""### 短响应首步精度探针

设置为同一原始1.5B师生、单卡FSDP、SDPA、单条固定128-token响应、Top-16、eta=0.3、FP32 master及AdamW。两臂各先执行一次lr=0，再执行三次lr=1e-6。教师前向精度随BF16/FP32条件改变，因此它不是固定BF16教师标签下的严格学生精度对照。

零学习率后两臂student center保持零。首次实际更新后，BF16为0.00040795，FP32为0.0050819。该短响应探针确认精度可改变首步，但它的“FP32首步更大”不能外推正式长响应B64条件。数值见[短响应探针](../../results/oneshot-inner-audit/precision-probe.json)。

### 长响应B64预检与零更新控制

设置为seed42的同一64条固定响应、395862个有效动作，每卡micro-batch1；两种学生前向各执行4次全batch更新，教师标签统一为同一个BF16分布。完整词表测量使用同一4096个前缀。update_before3表示已完成三次更新。

| 学生精度 | 初始动作绝对差 | 三次更新后绝对差 | 初始完整KL | 三次更新后完整KL |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 0.239874 | 0.239209 | 0.106206 | 0.105518 |
| FP32 | 0.238875 | 0.217701 | 0.105226 | 0.085883 |

首次lr=1e-6更新后，学生锚点KL分别为BF16的0.00040504和FP32的0.00019788。FP32首步锚点KL更小，教师拟合却更快。这纠正了短响应探针的首步方向外推。BF16 update4 checkpoint的570题结果为MATH345/500、AMC18/40、AIME5/30；它作为链路预检结果保存，不替代正式比较。

两个正式零学习率控制各执行4次相同全batch更新。BF16最大绝对student center均值1.0879e-10、最大RMS5.9314e-9；FP32对应1.1897e-10和5.8961e-9。被检查参数的变化比例为零，动作距离不变。两臂都通过原均值小于1e-6、RMS小于1e-5的判据。

### 参数舍入与梯度位移检查

对原始BF16 checkpoint的第0层Q projection和第12层MLP down projection，将每个元素加减1e-6后再转回BF16，改变表示的比例仅约1.31%和1.10%。这是两块矩阵、给定扰动幅度的舍入界检查，原始数值见[舍入界](../../results/oneshot-inner-audit/bf16-rounding-bound.json)。

B64首次实际更新中，被检查的FP32 master元素有99.986%改变，转换回BF16后有1.024%改变表示。四次更新后，两种训练精度的位移余弦相似度分别为Q projection的0.63925、MLP down projection的0.98545。参数表示与梯度方向都受数值实现影响，全部差异不能单独归因于权重舍入；结果见[位移检查](../../results/oneshot-effective-time/canary-master-delta.json)。

### 独立CPU checkpoint读出

使用半倍学习率BF16条件的update400 checkpoint，分别做CPU BF16与FP32前向。先检查最早8个高度确定的开头位置，教师距离都接近零；再从既定reservoir中按初始化anchor–teacher KL至少0.02、响应位置不超过512，选择最早8个位置。选择依据初始化标签，不依据训练后的精度差异。

响应位置12的BF16/FP32学生到教师KL为0.009583/0.273004，位置27为0.001249/0.979104。同一权重跨精度概率差异在独立HF模型、无FSDP、无GPU环境中出现。8个位置只验证差异存在，总体幅度以4096个固定前缀的读出为准。两组完整记录分别见[开头位置](../../results/oneshot-effective-time/cpu-readout-update400.json)和[初始化差异位置](../../results/oneshot-effective-time/cpu-readout-informative-update400.json)。""")
    add("## 当前实验范围与预先判据")
    design = source("2026-10-12-OneShot长期吸收与有效时间消融设计.md")
    add(
        section(design, "## 目标与预测", "## 与论文的对应").replace(
            "## 目标与预测", "### 2026-10-12预注册预测"
        )
    )
    add("""### 用户调整后的执行范围

原计划45个条件、16074次更新。2026-10-13按用户要求取消seed142和242；停止当前FP32长跑，并取消其余FP32学习率扫描。当前保留12个完整条件、4308次更新，另保留330条已停止FP32记录。原“至少两个种子通过”及FP32跨学习率20%一致性不再执行，不宣称这两个判据已经验证；各已执行条件的数值阈值保持原样。

| 条件 | 学生前向 | 学习率 | 更新预算 | 状态 |
| --- | --- | --- | --- | --- |
| 零学习率控制 | BF16、FP32 | 0 | 各4 | 均完成 |
| 固定监督OPD | BF16 | 5e-7、1e-6、2e-6 | 600、300、150 | 均完成 |
| 固定监督OPD | FP32 | 5e-7 | 原定600，保留330条记录 | 停止，不作完整条件 |
| 在线OPD | BF16 | 5e-7、1e-6、2e-6 | 600、300、150 | 半倍运行中，其余待运行 |
| 固定初始锚点MDPO/BPO | BF16 | 1e-6 | 各300 | 待运行 |
| 每16步重设锚点MDPO/BPO | BF16 | 1e-6 | 各800 | 待运行 |

执行范围文件为[data配置](../../data/oneshot-effective-time-20261012/execution-selection.json)，原注册快照在本文附件中。条件ID始终保留，取消的条件不会重新列为pending。""")
    add("## 固定监督的三学习率实验")
    add("""### 设置

学生DeepSeek-R1-Distill-Qwen-1.5B，教师JustRL-DeepSeek-1.5B。t22.parquet的64行核对为同一个prompt，原始文本与原生模板保留。学生一次采样64条响应，temperature=1、top-p=1、top-k关闭、上限7168 token，自然EOS及预算末动作保留。三条件复用完全相同tokens、mask和BF16教师标签。

| 项目 | 实际设置 |
| --- | --- |
| 训练数据 | seed42固定64条响应、395862有效动作 |
| 概率诊断 | 全部位置均匀无放回抽取4096前缀，跨条件复用 |
| 全局batch / mini-batch / 每卡micro-batch | 64 / 64 / 1，四卡FSDP，16条/卡累积后一次更新 |
| 学生参数与前向 | FP32 master，BF16前向，FP32梯度归约 |
| 优化器 | AdamW，betas=(0.9,0.999)，epsilon=1e-8，weight decay=0.01，clip=1 |
| 学习率 | 恒定，无warmup；5e-7 / 1e-6 / 2e-6 |
| 更新预算 | 600 / 300 / 150，终点有效时间均为300 |
| 损失 | 当前teacher–student动作advantage停止梯度，opd_noratio score surrogate |
| 完整测试 | 相同570题，BF16 greedy，上限7168，保存逐题输出与评分来源SHA-256 |
| checkpoint | 有效时间50、100、200、300；初始化和教师分别全量评测 |

这组weight decay和clip与原四卡多题训练不同。冻结的是动作与状态，学生评分每步更新，教师标签保持不变。""")
    add("### 结果：固定动作长期吸收")
    add(
        "| 学习率 | 更新预算 | 初始绝对差 | 最终绝对差 | 最终完整词表KL |\n| --- | ---: | ---: | ---: | ---: |"
    )
    for arm in bf:
        config = arm["config"]
        last = full_rows(config["id"])[-1]
        add(
            f"| {config['lr']:g} | {config['updates']} | {arm['distance'][0]:.6f} | {arm['distance'][-1]:.8f} | {last['metrics']['student_teacher_full_kl']:.8f} |"
        )
    add(
        "| 有效时间u | 半倍学习率动作差 | 标准学习率动作差 | 两倍学习率动作差 |\n| --- | ---: | ---: | ---: |"
    )
    for u in (50, 100, 200, 300):
        values = [distance(a, int(u * 1e-6 / a["config"]["lr"])) for a in bf]
        add(f"| {u} | " + " | ".join(f"{v:.7f}" for v in values) + " |")
    add(f"""标准学习率的d100=0.0525956、d200=0.0292306、d300=0.0220962，两个比值为0.555761和0.420115，均通过原单种子阈值。半倍学习率从100步的0.113953持续降至600步的0.0207191。结果重现固定状态数百步持续拟合，终点误差仍为正。

### 结果：有效时间缩放

定义u=(lr/1e-6)t。在相同数值网格30–150上对log距离插值，先求三个学习率间的方差，再对网格平均并开平方。按原始更新数计算的离散度为{collapse["raw_log_dispersion"]:.6f}，按有效时间计算为{collapse["effective_log_dispersion"]:.6f}，降低{100 * collapse["dispersion_reduction"]:.2f}%，通过降低至少30%的预注册预测。

完整u=30–300区间，三学习率最大/最小距离相对差的中位数为{100 * collapse["effective_pair_relative_spread_median"]:.2f}%，最大为{100 * collapse["effective_pair_relative_spread_max"]:.2f}%。该结果是近似曲线缩放，不是精确不变性；原20%判据针对FP32扫描，不能把BF16的这个数值解释成FP32预测失败。

![三个已完成BF16条件的原始更新轴和有效时间轴](../../results/oneshot-effective-time/consolidated/frozen-effective-time.png)

### 结果：完整数学评测

下表每行均为完整570题结果。括号不是独立实验：更新数按相同有效时间安排。三个条件共用同一初始化基线；表中为正确题数。

| 模型/学习率 | 更新数 | 有效时间 | MATH-500 /500 | AMC23 /40 | AIME25 /30 |
| --- | ---: | ---: | ---: | ---: | ---: |""")
    for identifier, label in (("initial-s42", "初始化学生"), ("teacher", "冻结教师")):
        e = evaluations[identifier]
        add(
            f"| {label} | — | — | {e['MATH-500']['correct']} | {e['AMC23']['correct']} | {e['AIME25']['correct']} |"
        )
    for arm in bf:
        config = arm["config"]
        for t in config["checkpoint_steps"]:
            e = evaluations[config["id"] + f"-update{t}"]
            add(
                f"| {config['lr']:g} | {t} | {config['lr'] / 1e-6 * t:g} | {e['MATH-500']['correct']} | {e['AMC23']['correct']} | {e['AIME25']['correct']} |"
            )
    add(
        "| 学习率终点 | MATH准确率 / 预算耗尽 | AMC准确率 / 预算耗尽 | AIME准确率 / 预算耗尽 |\n| --- | --- | --- | --- |"
    )
    for arm in bf:
        c = arm["config"]
        e = evaluations[c["id"] + f"-update{c['updates']}"]
        cells = [
            f"{100 * e[t]['accuracy']:.2f}% / {100 * e[t]['budget_exhausted_fraction']:.2f}%"
            for t in ("MATH-500", "AMC23", "AIME25")
        ]
        add(f"| {c['lr']:g} | " + " | ".join(cells) + " |")
    add("""![相同有效时间checkpoint的准确率与预算耗尽率](../../results/oneshot-effective-time/consolidated/frozen-benchmarks.png)

### 分析

三种学习率终点的MATH正确数为360、351、343，均低于初始化364；AMC为18、20、21，低于或等于初始化21；AIME为3、5、4，均低于初始化7。固定训练分布对齐改善与跨题数学准确率没有同时改善。教师自身的AIME为4/30，低于学生7/30，因此教师优势也不能按三个任务统一假定。

半倍学习率终点相对初始化，MATH有53题由错变对、57题由对变错，净减少4题；AMC为5题与8题，净减少3题；AIME为0题与4题，净减少4题。该逐题结果描述确定的这组评测，不从单次小测试集推断跨训练种子的稳定性。所有checkpoint输出完整保留，不以曲线中最好的一点替代预定终点。""")
    fixed = section(progress, "固定动作的", "## 资源与复现入口")
    add("### 固定动作loss的真实优化含义\n\n" + fixed)
    add(
        section(
            progress, "## AdamW 与有效时间的关系", "## 标准学习率条件的长期吸收首个判分点"
        ).replace("## AdamW 与有效时间的关系", "### 有效时间的优化器条件")
    )
    add("## FP32固定监督对照")
    add("""### 设置与停止状态

与半倍BF16条件复用相同64条固定响应、教师BF16标签和AdamW参数，学习率5e-7，仅改变学生计算精度。FP32已提交330条更新记录，保留update100、200的模型与优化器checkpoint；原预算600没有执行完，后续两个FP32学习率未启动。教师监督不随此条件改变。

### 结果

| 更新数 | BF16动作绝对差 | FP32动作绝对差 | BF16完整KL | FP32完整KL |
| --- | ---: | ---: | ---: | ---: |""")
    fp_by_step = {r["update_before"]: r for r in fp}
    fp_full = {r["update"]: r for r in full_rows(fp_config["id"])}
    bf_full = {r["update"]: r for r in full_rows(bf[0]["config"]["id"])}
    for t in (0, 20, 100, 200, 300):
        add(
            f"| {t} | {distance(bf[0], t):.7f} | {fp_by_step[t]['metrics']['current_advantage_abs/mean']:.7f} | {bf_full[t]['metrics']['student_teacher_full_kl']:.7f} | {fp_full[t]['metrics']['student_teacher_full_kl']:.7f} |"
        )
    add("""![相同半倍学习率下两种训练精度的完整共同观测区间](../../results/oneshot-effective-time/consolidated/precision-comparison.png)

### 分析

100步时FP32动作距离约为BF16的一半，300步仍约低38.8%。计算精度明显改变拟合轨迹；单点距离比不能直接当成相同倍数的加速。FP32从100到200、300步的距离比为0.5354和0.3681，仍在后续数百步持续学习。长期吸收不能全部归因于BF16舍入。

首次半倍学习率更新后，BF16学生锚点KL为0.000406099，FP32为0.0000491368；FP32教师动作差仍为0.235307。FP32也可同时具有小锚点KL和明显教师距离。300步时，两种学生锚点KL为0.0885683和0.0892986，而教师完整KL分别只有0.00505154和0.00345255。学生接近教师的同时远离初始锚点。

把BF16训练权重直接改用FP32前向，是另一项读出检查。标准学习率BF16训练300步后的FP32读出KL为0.138567，不能拿它代表从初始化开始FP32训练的效果。FP32训练在半倍学习率300步的KL为0.00345255，属于不同优化轨迹。该FP32条件没有数学基准评测，因此不报告其数学能力变化。

FP32的330条更新记录累计约19361秒（5.38小时）；半倍BF16条件完成600更新约6135秒（1.70小时）。这些时间包含相应诊断和checkpoint，不是纯前向内核benchmark。在这条实现中，FP32每步拟合更快，但实际运行成本更高。现有精度对照足以确认速度影响，更多FP32扫描不能替代锚点重置和镜像目标检验，故按用户要求和证据优先级停止。部分日志、启动配置和24个源码签名保存在[data附件](../../data/oneshot-effective-time-20261012/frozen-opd-fp32-lr5em7-s42/partial-manifest.json)。""")
    add("## 历史多题学习率扫描")
    add(
        below(section(stability, "## 原始数据与 setting", "## 实验判断与下一组扫描位置"))
        .replace("### 原始数据与 setting", "### 设置")
        .replace("### 高学习率的实际行为", "### 结果：实际高学习率分支")
    )
    add(r"""确定性二次梯度下降的谱换算需要额外假设。若Hessian特征值为$\lambda_i$、误差沿该特征方向为$e_i$，一次GD有

$$
e_{i,t+1}=(1-\alpha\lambda_i)e_{i,t},\qquad L_{i,t+1}=(1-\alpha\lambda_i)^2L_{i,t}.
$$

按$u=\alpha t/\alpha_0$写成单位有效时间的对数下降率，得到

$$
k_i=-\frac{2\alpha_0}{\alpha}\log|1-\alpha\lambda_i|
\simeq2\alpha_0\lambda_i\quad(\alpha\lambda_i\ll1).
$$

从该近似倒推曲率还需固定二次目标、GD方向和最大曲率被观测激发。历史在线AdamW及其有符号距离没有这些条件，因此谱拟合只能做低学习率路径参考。
""")
    add(
        "### 分析与已验证边界\n\n"
        + section(stability, "## 结论", "## 原始数据与 setting").removeprefix("## 结论").strip()
    )
    add(
        """1.5e-5、2e-5、3e-5和必要时5e-5是当前历史拟合给出的候选扫描点，没有这些训练记录，亦不加入当前有限执行队列。该历史M4设置是多题、每步新rollout；其数值边界不迁移为FP32、固定锚点、eta=0.3的BPO可用学习率。"""
    )
    add("## 在线与锚点生命周期的剩余消融")
    add(f"""### 在线OPD：设置与部分结果

BF16、t22单题、每步重新从当前学生生成64条响应，再重新打教师标签；其余优化器和完整概率测量遵循当前单题配方。半倍、标准、两倍学习率分别预算600、300、150步。动作前缀和有效token数逐步变化，固定4096初始前缀的独立读出继续保留。

数据快照中半倍条件已完成{online_progress["updates_completed"]}/600步，另外两个条件尚未启动。最新已提交pre-update记录对应更新前{online[-1]["update_before"]}步，动作绝对差为{online[-1]["metrics"]["current_advantage_abs/mean"]:.7f}，student center均值为{online[-1]["metrics"]["student_center/mean"]:.3g}。这是新rollout测度下的即时量，不与固定动作池的同一步距离混成一个严格因果比较。三个在线条件和checkpoint评测未完成，当前不给在线时间缩放或数学准确率结论。

### 固定与重设锚点MDPO/BPO：设置

四个条件使用相同固定64响应、BF16、lr=1e-6、eta=0.3及Top-16加尾部桶。固定初始锚点条件各300步；每16更新重设锚点条件各800步。独立记录当前教师完整KL、全局初始锚点距离、当前窗口的镜像目标误差和每16更新后的读出，区分缓存起点与实际目标改善。四条件均未启动。

### 方法边界与待判定问题

原四卡正式训练每轮刷新响应，动作来自新锚点q；上述生命周期探针继续复用初始q0动作池，重设的是锚点及相应奖励、中心与目标。因此它是固定pool下的缓存与目标生命周期对照，不是原fresh-rollout算法的等价复现。""")
    add(r"""重设锚点为$q_k$后，生产MDPO系数仍包含$p/q_k$，而固定动作测度是$q_0$。其动作期望为

$$
\mathbb E_{q_0}\!\left[\frac p{q_k}c(a)\nabla\log p(a)\right]
=\sum_a p(a)\frac{q_0(a)}{q_k(a)}c(a)\nabla\log p(a),
$$

一般不同于$\mathbb E_p[c(a)\nabla\log p(a)]$。BPO的固定pool期望也一般不同于完整$q_k$下的方差梯度。当前设置包含动作测度与锚点改变的共同作用，结果按这一配方解释；目标误差是独立诊断，不声称生产代理已经精确优化理想分布目标。

需要判定的是：student center的重置与小幅度能否同非零教师距离及目标误差同时出现；固定锚点和窗口锚点如何改变所见曲线；完整更新预算下的数学评测是否改善。现阶段不为这些待运行条件填写结果。""")
    add("""## 证据、复算与维护

本报告采用单一维护位置。原10份人读报告由本报告替代，逐字节快照、文件名和SHA-256保存在data/oneshot-report-sources-20261013；原预注册文本也在该附件中。过程记录继续写入runs和现有operations历史，不另建进展报告。

| 证据 | 入口 |
| --- | --- |
| 历史t22 CSV、metadata | ../MD-OPD/plots/runs/ |
| 原正式数学记录与恢复签名 | data/training/math-4gpu-final-20261012/、data-manifest.json |
| 原inner复算及精度探针 | results/oneshot-inner-audit/ |
| 历史学习率输入与统计 | data/oneshot-lr-stability/、results/oneshot-lr-stability/ |
| 当前配置、运行源码快照、日志、checkpoint | runs/oneshot-effective-time-20261012/ |
| 用户修改后的范围及部分FP32附件 | data/oneshot-effective-time-20261012/ |
| 当前正式统计 | results/oneshot-effective-time/summary.json |
| 本报告表图的数据快照 | results/oneshot-effective-time/consolidated/report-snapshot.json |
| 可分享矢量图 | 本报告consolidated图目录下同名PDF |

当前训练严格只用GPU0–3，一个GPU任务结束且四卡重新空闲后启动下一个。运行期间监督driver进程归属，发现冲突则停止自己的任务并等待；不终止其他任务。CPU分析可并行。当前结论覆盖已执行的seed42响应集合与协议，取消的多种子重复不产生跨seed稳健性结论。

训练入口为training/oneshot_suite.py；当前统计为experiments/analysis/oneshot_effective_time.py；本报告与新图的更新入口为experiments/analysis/oneshot_experiment_report.py。历史内层复算、历史学习率复算和正式数学汇总分别使用oneshot_inner_audit.py、oneshot_lr_stability.py和math_training_report.py。更新本报告可在仓库根执行以下CPU命令。""")
    add(
        "~~~bash\nsource ../.bashrc\nfht-mirror\nCUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m experiments.analysis.oneshot_experiment_report\n~~~"
    )
    add("""### 参考文献

[^oneshot]: Zixuan Fu, Bingxiang He, Yuxin Zuo, et al. Rethinking On-Policy Distillation of Large Language Models II: One Training Example. arXiv 2026. [arXiv:2609.04172](https://arxiv.org/html/2609.04172).
[^online]: WncFht. One-Shot-OPD online one-shot run 5aub1k4h. W&B 2026. [Run 5aub1k4h](https://wandb.ai/wncfht/One-Shot-OPD/runs/5aub1k4h).
[^frozen]: WncFht. One-Shot-OPD frozen rollout run ceejzu26. W&B 2026. [Run ceejzu26](https://wandb.ai/wncfht/One-Shot-OPD/runs/ceejzu26).
[^full]: WncFht. One-Shot-OPD full-data run mdxt04lm. W&B 2026. [Run mdxt04lm](https://wandb.ai/wncfht/One-Shot-OPD/runs/mdxt04lm).
[^wandb-wass1wqb]: MirrorOPD. on-policy formal math training run. W&B 2026. [Run wass1wqb](https://wandb.ai/wncfht/mirror-opd-dapo-math-4gpu/runs/wass1wqb).
[^wandb-3o5o72du]: MirrorOPD. MDPO formal math training run. W&B 2026. [Run 3o5o72du](https://wandb.ai/wncfht/mirror-opd-dapo-math-4gpu/runs/3o5o72du).
[^wandb-gp7pyuj6]: MirrorOPD. BPO formal math training run. W&B 2026. [Run gp7pyuj6](https://wandb.ai/wncfht/mirror-opd-dapo-math-4gpu/runs/gp7pyuj6).
""")
    text = "\n\n".join(parts) + "\n"
    anchors = {
        "## 四卡多题数学蒸馏比较": "math-full",
        "### 完整在线OPD": "math-onpolicy",
        "### MDPO": "math-mdpo",
        "### BPO": "math-bpo",
        "## 指标定义与原内层日志": "kl-semantics",
        "## 当前实验范围与预先判据": "registered-scope",
        "## 固定监督的三学习率实验": "frozen-lr",
        "## FP32固定监督对照": "fp32-fixed",
        "## 历史多题学习率扫描": "historical-lr",
    }
    for heading, identifier in anchors.items():
        text = text.replace(heading + "\n", f'<a id="{identifier}"></a>\n\n' + heading + "\n", 1)
    for old, anchor in ALIASES.items():
        text = text.replace(old, REPORT.name + "#" + anchor)
    assert len(text.splitlines()) < 1000
    REPORT.write_text(text)
    snapshot = {
        "scope": json.loads((SUITE / "execution-selection.json").read_text()),
        "bf16_time_collapse": collapse,
        "formal_benchmarks": {
            k: v for k, v in evaluations.items() if k != "canary-bf16-s42-update4"
        },
        "fp32_partial_logged_updates": len(fp),
        "online_progress": online_progress,
        "controller_state": state,
        "completed_bf16_arms": [
            {
                "id": a["config"]["id"],
                "updates": a["config"]["updates"],
                "final_distance": a["distance"][-1],
            }
            for a in bf
        ],
    }
    (OUT / "report-snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(REPORT),
                "lines": len(text.splitlines()),
                "online_updates": online_progress["updates_completed"],
            }
        )
    )


if __name__ == "__main__":
    main()
