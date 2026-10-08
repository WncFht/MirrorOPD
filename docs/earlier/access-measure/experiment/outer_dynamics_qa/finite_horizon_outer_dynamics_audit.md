# MirrorOPD outer synchronization audit

This experiment freezes the anchor policy pi_k at each outer step, samples from that anchor, performs N inner updates, and hard-syncs the resulting weights as pi_{k+1}.
The finite horizon has H independent time states and two actions; rewards are deterministic and the occupancy is uniform over time states.

## Reproducibility

Command: `python3 run_finite_horizon.py --seed 20261007 --horizon 6 --outer-steps 16 --inner-steps 24 --samples 256 --lr 0.2 --eta0 0.2 --eta1 0.7`
Seed=20261007; H=6; outer=16; inner=24; samples=256; lr=0.2.

## Interpretation

For each row, `target_kl_anchor` is the exact entropy-mirror target displacement, while `kl_policy_target` measures the finite inner budget error. `value_gain` is the one-step reward improvement against the frozen anchor; it need not stay non-negative for a stochastic approximate inner solve.
The fixed schedule keeps eta constant. The linear schedule interpolates eta0 to eta1 across outer steps. Since eta changes only at synchronization boundaries, the inner dynamics remain piecewise frozen; there is no hidden interpolation of pi_k inside an outer step.
The four methods share teacher, samples, initialization, and hyperparameters. Differences therefore diagnose estimator/geometry effects rather than rollout noise.

## Final rows

| method | schedule | value | KL(policy\|teacher) | KL(policy\|target) |
|---|---|---:|---:|---:|
| bpo | fixed | 0.573786 | 0.000055 | 0.000000 |
| bpo | linear | 0.574824 | 0.000000 | 0.000000 |
| mdpo | fixed | 0.574137 | 0.000024 | 0.000000 |
| mdpo | linear | 0.574828 | 0.000000 | 0.000000 |
| pmd_mean | fixed | 0.573784 | 0.000055 | 0.000000 |
| pmd_mean | linear | 0.574824 | 0.000000 | 0.000000 |
| pmd_part | fixed | 0.573791 | 0.000054 | 0.000000 |
| pmd_part | linear | 0.574824 | 0.000000 | 0.000000 |
