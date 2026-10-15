#!/usr/bin/env python3
"""Flatten the current book and integrate audited old-only material in place."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "2026-10-14-MirrorOPD新旧书稿单文件合并版"
DEST = ROOT / "theory/book-merged" / f"{NAME}.tex"


def flatten(book, relative):
    path = book / relative
    source = path.read_text()
    return re.sub(
        r"\\input\{([^}]+)\}",
        lambda match: (
            f"\n% BEGIN SOURCE: {path.parent.parent.name}/{match[1]}.tex\n"
            + flatten(book, match[1] + ".tex")
            + f"\n% END SOURCE: {match[1]}.tex\n"
        ),
        source,
    )


def main():
    entry = "2026-10-08-固定测度Token级MirrorOPD.tex"
    current = flatten(ROOT / "theory/book", entry)
    preface = r"""
\chapter*{单文件合并版说明}
\addcontentsline{toc}{chapter}{单文件合并版说明}
本书将当前稿与重构前旧稿整合为一套正文。已迁移的定理、证明、算法和实验表只保留当前修订版本；旧稿独有的有效内容补入对应主题。旧稿中已修正的结论不再作为另一版本附录收录。九章全部直接写入本文件，不通过章节级输入命令拆分；完整推导保留。

这是便于集中阅读与编辑的独立合并导出版，原有当前主稿及历史归档均保留。正文内容截至 2026 年 10 月 14 日。源文件只合并正文；ElegantBook 样式与结果图片继续复用仓库文件。文件行数按单文件交付要求保留，不按原章节拆分。

旧版修订核验记录中的二动作 Mean 数值例子补入内层投影章，原论文标签对应关系补在乘子换算处，48 组随机问题的配置补入实验证据章。旧稿中重复的 OPD 单轮 Lambert-$W$ 归一化证明统一使用一般冻结信号的证明，OPD 实例保留代入与乘子核对。
"""
    current = current.replace(r"\input{sections/reading_guide}", "")
    current = current.replace(r"\frontmatter", r"\frontmatter" + "\n" + preface, 1)
    supplement = r"""
\paragraph{二动作数值例子}
固定 $q=(0.5,0.5)$、$t=(0.75,0.25)$、$\eta=1$，旧记录的 Mean 最优分布与乘子为
\begin{align*}
p^{\mathrm M}&=(0.7291869550174757,0.2708130449825245),\\
\lambda&=0.117928665874.
\end{align*}
Gibbs 解为教师 $t$。该例直接区分 Mean 的 Lambert-$W$ 投影与 Gibbs 投影，排除了 Mean 在该条件下一步精确恢复教师的说法。

"""
    anchor = "本节损失的乘子 $\\lambda_k$ 对应原论文乘子除以"
    assert current.count(anchor) == 1
    current = current.replace(anchor, supplement + anchor, 1)
    mapping = r"""
原论文源文件中的总体解、混合正则化解释与精细损失恒等式分别使用标签 \path{thm:pmdmean_solution}、\path{eq:mixed_subproblem} 和 \path{lem:pmdmean_quad_growth}。这些是来源定位信息；本书的相应结果为定理~\ref{thm:ch2-mean-lambert}、式~\eqref{eq:ch2-mean-mixed-objective} 和式~\eqref{dyn:eq:mean-opd-excess-identity}。

"""
    anchor = "实际样本只需估计冻结的 reward center。"
    assert current.count(anchor) == 1
    current = current.replace(anchor, mapping + anchor, 1)
    old_sentence = r"\path{tests/test_mean_bpo.py} 进一步核对 Mean 与 BPO 的解析梯度、有限差分、最优点和完整词表内层几何；它与 toy 实验分别检验推导正确性和有限步优化行为。"
    new_sentence = r"\path{tests/test_mean_bpo.py} 的旧版修订核验记录覆盖 48 组随机问题：词表维数为 2、4、8，镜像系数为 0.1、0.5、1、2。核验对象包括解析梯度、逐坐标有限差分、Mean/BPO 损失及梯度恒等式、Mean 归一化与 KKT 条件、最优点驻点、Part/BPO 的 Gibbs 零损失、初始化方向尺度、恒定信号及完整词表内层几何。该记录与 toy 实验分别检验推导正确性和有限步优化行为；这里保留历史配置，不将其写成合并时重新运行的测试。"
    assert current.count(old_sentence) == 1
    current = current.replace(
        old_sentence,
        new_sentence,
        1,
    )
    current = current.replace(
        r"\title{固定测度 Token 级 MirrorOPD：目标、投影、估计与动力学}",
        r"\title{MirrorOPD 新旧书稿单文件合并版}",
    )
    current = current.replace("目标、投影、估计与动力学\\par", "新旧书稿单文件合并版\\par")
    # Resolve the class through the existing maintained copy, without duplicating it.
    current = current.replace("{elegantbook}", "{../book/elegantbook}", 1)
    current = current.replace(
        r"\begin{titlepage}", r"\hypersetup{pageanchor=false}" + "\n" + r"\begin{titlepage}", 1
    )
    current = current.replace(
        r"\frontmatter", r"\frontmatter" + "\n" + r"\hypersetup{pageanchor=true}", 1
    )
    assert not re.search(r"\\(?:input|include)\{", current)
    labels = re.findall(r"\\label\{([^}]+)\}", current)
    assert len(labels) == len(set(labels)), "Duplicate label"
    references = re.findall(r"\\(?:ref|eqref|cref|Cref|pageref|autoref)\{([^}]+)\}", current)
    assert not set(references) - set(labels), set(references) - set(labels)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(current)
    print(
        f"{DEST.relative_to(ROOT)}: {len(current.splitlines())} lines, {len(labels)} labels; no external chapter inputs"
    )


if __name__ == "__main__":
    main()
