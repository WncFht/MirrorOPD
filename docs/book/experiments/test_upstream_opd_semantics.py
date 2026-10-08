"""Execute upstream pure tensor functions without loading GPU/Ray services.

AST extraction retains original function bodies. It is a mathematical audit,
not a substitute for importing and testing the complete production worker.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

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
                        ['log1mexp','sample_corrected_ta_opd','compute_mirror_opd_token_loss'], {'torch': torch, '_LOG2': math.log(2)})
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
    from token_inner import pmd_mean_loss,bpo_loss,mdpo_loss
    for method, original in [('pmd_mean',pmd_mean_loss),('bpo',bpo_loss),('mdpo',mdpo_loss)]:
        for k in [0,2,4]:
            x=logits.detach().clone().requires_grad_(True)
            anchor=behavior
            ids=anchor.topk(k).indices
            outside=torch.ones(4,dtype=torch.bool); outside[ids]=False
            qt=anchor[outside].sum()
            tail_lp=qt.log()
            kt=(anchor*(anchor.log()-q.log())).sum()
            fields=ns['compute_mirror_opd_token_loss'](
                x.expand(4,-1),torch.arange(4),ids.expand(4,-1),
                anchor[ids].log().expand(4,-1),tail_lp.expand(4),
                anchor.log(),q.log(),kt.expand(4),method,.5)
            estimated=(anchor*fields['distillation_losses']).sum()
            gradient=torch.autograd.grad(estimated,x,retain_graph=True)[0]
            assert torch.isfinite(gradient).all()
            assert not fields['mirror_student_center'].requires_grad
            if k==4:
                expected_loss=original(x,anchor,q.log()-anchor.log(),.5)
                torch.testing.assert_close(gradient,torch.autograd.grad(expected_loss,x)[0],atol=3e-7,rtol=3e-6)
    print('PASS: TA-OPD fork MirrorOPD primitive: 9 support/method cases, exact gradient endpoints')
    load_functions(root/'TA-OPD/verl/verl/trainer/distillation/fsdp/losses.py',
                   ['compute_mirror_opd_from_cache'], ns)
    ns['get_ulysses_sequence_parallel_world_size'] = lambda: 1
    offsets = torch.tensor([0, 4, 7])
    valid = torch.tensor([False, True, True, False, False, True, False])
    anchor = torch.tensor([.4, .3, .2, .1])
    teacher = torch.tensor([.1, .2, .3, .4])
    kt = (anchor * (anchor.log()-teacher.log())).sum()
    cache = {
        'mirror_valid': valid,
        'mirror_actions': torch.tensor([-99, 0, 1, -99, -99, 2, -99]),
        'mirror_anchor_ids': torch.tensor([[0, 1]]).expand(7, -1).clone(),
        'mirror_anchor_head_log_probs': anchor[:2].log().expand(7, -1).clone(),
        'mirror_anchor_tail_log_prob': torch.full((7,), .3).log(),
        'mirror_anchor_action_log_prob': anchor[torch.tensor([0,0,1,0,0,2,0])].log(),
        'mirror_teacher_action_log_prob': teacher[torch.tensor([0,0,1,0,0,2,0])].log(),
        'mirror_teacher_center': kt.expand(7).clone(),
    }
    # Invalid positions deliberately contain unusable ids and NaN caches.
    cache['mirror_anchor_ids'][~valid] = -99
    for key, value in cache.items():
        if value.is_floating_point():
            value[~valid] = float('nan')
    nested = {key: torch.nested.nested_tensor_from_jagged(value, offsets)
              for key, value in cache.items()}
    for method in ('pmd_mean', 'bpo', 'mdpo'):
        x = torch.randn(1, 7, 4, requires_grad=True)
        cfg = SimpleNamespace(distillation_loss=SimpleNamespace(
            loss_mode='mirror_'+method, mirror_eta=.5))
        fields = ns['compute_mirror_opd_from_cache'](x, nested, cfg, 'thd')
        direct = ns['compute_mirror_opd_token_loss'](
            x[0, valid], cache['mirror_actions'][valid],
            cache['mirror_anchor_ids'][valid], cache['mirror_anchor_head_log_probs'][valid],
            cache['mirror_anchor_tail_log_prob'][valid], cache['mirror_anchor_action_log_prob'][valid],
            cache['mirror_teacher_action_log_prob'][valid], cache['mirror_teacher_center'][valid],
            method, .5)
        for key, values in fields.items():
            assert torch.isfinite(values).all()
            assert (values[0, ~valid] == 0).all()
            torch.testing.assert_close(values[0, valid], direct[key])
        grad = torch.autograd.grad(fields['distillation_losses'].sum(), x)[0]
        assert (grad[0, ~valid] == 0).all()
        assert grad[0, valid].abs().sum() > 0
    print('PASS: jagged MirrorOPD adapter: 3 methods, invalid-position masking and live gradients')
    print('PASS: upstream normalized-weight identity and small-tail floor verified')


if __name__ == '__main__':
    main()
