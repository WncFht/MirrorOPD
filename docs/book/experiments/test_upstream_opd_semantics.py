"""Execute upstream pure tensor functions without loading GPU/Ray services.

AST extraction retains original function bodies. It is a mathematical audit,
not a substitute for importing and testing the complete production worker.
"""
import ast
import math
from pathlib import Path

import torch


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in found} == set(names)
    exec(compile(ast.Module(body=found, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def main():
    root = Path(__file__).resolve().parents[2] / 'pmd-theory/MirrorOPD/source_audit'
    ns = load_functions(root/'One-Shot-OPD/verl/verl/workers/fsdp_teacher_worker.py',
                        ['compute_reward_weights', 'compute_top_k_reward'], {'torch': torch})
    p = torch.tensor([[.6, .3]], dtype=torch.float64)
    q = torch.tensor([[.3, .2]], dtype=torch.float64)
    overlap = torch.ones_like(p, dtype=torch.bool)
    reward = ns['compute_top_k_reward'](p.log(),q.log(),overlap,overlap,'only_stu','student_p').sum()
    norm = ((p/p.sum())*((p/p.sum()).log()-(q/q.sum()).log())).sum()
    # Reward uses normalized weights but raw full-vocabulary log-ratios.
    torch.testing.assert_close(reward, -norm-torch.log(p.sum()/q.sum()))
    assert not torch.isclose(reward, -norm)
    print(f'One-Shot sum reward={reward.item():.9f}; negative conditional KL={-norm.item():.9f}')
    ns = load_functions(root/'TA-OPD/verl/verl/trainer/distillation/fsdp/losses.py',
                        ['log1mexp','sample_corrected_ta_opd'], {'torch': torch, '_LOG2': math.log(2)})
    tails = torch.tensor([1e-3,1e-6,1e-8,1e-12],dtype=torch.float64)
    actual = ns['log1mexp'](torch.log1p(-tails)).exp()
    print('TA-OPD true tails:',tails.tolist())
    print('TA-OPD returned tails:',actual.tolist())
    torch.testing.assert_close(actual[0],tails[0])
    assert (actual[2:]>99*tails[2:]).all()
    # Enumerate reused behavior actions: correction must have correct value
    # AND gradient, even when behavior differs substantially from student.
    logits=torch.tensor([.5,-.1,-.7,.2],dtype=torch.float64,requires_grad=True)
    lp=logits.log_softmax(-1); p=lp.exp()
    q=torch.tensor([.2,.4,.1,.3],dtype=torch.float64)
    behavior=torch.tensor([.5,.1,.3,.1],dtype=torch.float64)
    inside=torch.tensor([False,True,False,True])
    pt=p[~inside].sum(); qt=q[~inside].sum()
    pa=torch.cat([p[inside],pt[None]])
    qa=torch.cat([q[inside],qt[None]])
    ta=(pa*(pa.log()-qa.log()).detach()).sum()
    corrected=ns['sample_corrected_ta_opd'](
        ta.expand(4),lp,q.log(),pt.log().expand(4),qt.log().expand(4),
        inside,behavior.log())
    expected=(behavior*corrected).sum()
    full=(p*(lp-q.log())).sum()
    # Upstream casts to float32, so use tolerances reflecting that operation.
    torch.testing.assert_close(expected,full,atol=2e-7,rtol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(expected,logits,retain_graph=True)[0],
                               torch.autograd.grad(full,logits)[0],atol=2e-7,rtol=2e-6)
    print('PASS: upstream sample-corrected TA value/gradient under reused behavior')
    print('PASS: upstream normalized-weight identity and small-tail floor verified')


if __name__ == '__main__':
    main()
