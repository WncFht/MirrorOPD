"""Check source estimator gradients and the unified pseudocode's interior identities."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch


def load_kernel(path, name):
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"torch": torch}
    module = ast.Module(body=[function], type_ignores=[])
    # Execute only the reviewed pure tensor function from the local pinned source.
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=repo / "upstream")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "results/analysis/2026-10-12-opd-pseudocode-identities.json",
    )
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    res_path = args.upstream_root / "ResOPD/verl/trainer/distillation/resopd.py"
    sc_path = args.upstream_root / "TA-OPD/verl/verl/trainer/distillation/fsdp/losses.py"
    res_kernel = load_kernel(res_path, "resopd_loss")
    sc_kernel = load_kernel(sc_path, "sample_corrected_ta_opd")
    rows = []
    g = torch.Generator().manual_seed(712)
    for vocab in [3, 7]:
        for case_index in range(16):
            z = torch.randn(vocab, generator=g).requires_grad_()
            logp = z.log_softmax(0)
            p = logp.exp()
            lt = torch.randn(vocab, generator=g).log_softmax(0)
            t = lt.exp()
            lb = torch.randn(vocab, generator=g).log_softmax(0)
            b = lb.exp()
            true = (p * (logp - lt)).sum()
            trueg = torch.autograd.grad(true, z, retain_graph=True)[0]
            for k in [1, vocab - 1]:
                ids = t.topk(k).indices
                ps = p[ids].sum()
                ts = t[ids].sum()
                pt = 1 - ps
                tt = 1 - ts
                delta = pt.log() - tt.log()
                coarse = (p[ids] * (logp[ids] - lt[ids])).sum() + pt * delta
                stopcoarse = (p[ids] * (logp[ids] - lt[ids]).detach()).sum() + pt * delta.detach()
                cg = torch.autograd.grad(coarse, z, retain_graph=True)[0]
                sg = torch.autograd.grad(stopcoarse, z, retain_graph=True)[0]
                resgr = []
                scgr = []
                for a in range(vocab):
                    res = res_kernel(z, lt[ids], ids, lt[a : a + 1], torch.tensor([a]))[
                        "distillation_losses"
                    ]
                    resgr.append(torch.autograd.grad(res, z, retain_graph=True)[0])
                    sc = sc_kernel(
                        stopcoarse,
                        logp[a],
                        lt[a],
                        pt.log(),
                        tt.log(),
                        torch.tensor(a in ids.tolist()),
                        lb[a],
                    )
                    scgr.append(torch.autograd.grad(sc, z, retain_graph=True)[0])
                resgr = torch.stack(resgr)
                scgr = torch.stack(scgr)
                re = (p.detach()[:, None] * resgr).sum(0)
                sce = (b[:, None] * scgr).sum(0)
                row = {
                    "vocab": vocab,
                    "case_index": case_index,
                    "k": k,
                    "res_onpolicy_gradient_error": float((re - trueg).abs().max()),
                    "sc_stale_proposal_gradient_error": float((sce - trueg).abs().max()),
                    "coarse_vs_stopped_coarse_gradient_error": float((cg - sg).abs().max()),
                    "res_stale_proposal_gradient_bias": float(
                        ((b[:, None] * resgr).sum(0) - trueg).abs().max()
                    ),
                }
                assert row["res_onpolicy_gradient_error"] < 1e-12, row
                assert row["sc_stale_proposal_gradient_error"] < 3e-6, row
                assert row["coarse_vs_stopped_coarse_gradient_error"] < 1e-12, row
                rows.append(row)
    q = torch.tensor([0.4, 0.35, 0.25])
    xi = torch.tensor([0.2, 0.5, 0.3])
    e = torch.tensor([0.5, 0.3, 0.2])
    f = torch.tensor([0.25, 0.25, 0.5])
    lam = 0.7
    support = torch.tensor([0, 2])
    qc = q[support] / q[support].sum()
    d = e[support].log() - f[support].log()
    u = d - lam * (q[support].log() - xi[support].log())
    advantage = qc * (u - (qc * u).sum())
    tv = (xi[support].log() + d / lam).softmax(0)
    r = tv.log() - qc.log()
    virtual_advantage = lam * qc * (r - (qc * r).sum())
    virtual_error = float((advantage - virtual_advantage).abs().max())
    assert virtual_error < 1e-12
    p = torch.tensor([0.2, 0.3, 0.5])
    t = torch.tensor([0.6, 0.1, 0.3])
    full = float((p * (p.log() - t.log())).sum())
    coarse = float(
        p[0] * (p[0].log() - t[0].log()) + (1 - p[0]) * ((1 - p[0]).log() - (1 - t[0]).log())
    )
    result = {
        "scope": "Finite-prefix CPU verification of source kernels and exact interior algebra; no GPU pipeline or training-quality evidence.",
        "cases": len(rows),
        "random_seed": 712,
        "max_res_onpolicy_error": max(r["res_onpolicy_gradient_error"] for r in rows),
        "max_sc_stale_proposal_error": max(r["sc_stale_proposal_gradient_error"] for r in rows),
        "max_coarse_score_alias_error": max(
            r["coarse_vs_stopped_coarse_gradient_error"] for r in rows
        ),
        "max_res_unweighted_stale_bias": max(r["res_stale_proposal_gradient_bias"] for r in rows),
        "a2_virtual_teacher_identity_error": virtual_error,
        "one_token_support_example": {
            "p": p.tolist(),
            "teacher": t.tolist(),
            "support": [0],
            "conditional_kl": 0,
            "coarse_kl": coarse,
            "full_kl": full,
        },
        "rows": rows,
    }

    result["source_sha256"] = {
        "ResOPD/resopd.py": hashlib.sha256(res_path.read_bytes()).hexdigest(),
        "TA-OPD/fsdp/losses.py": hashlib.sha256(sc_path.read_bytes()).hexdigest(),
    }
    result["torch_version"] = torch.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: value for name, value in result.items() if name != "rows"}, indent=2))


if __name__ == "__main__":
    main()
