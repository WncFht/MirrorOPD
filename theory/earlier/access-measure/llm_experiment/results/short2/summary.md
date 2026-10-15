# Real-model short run

Configuration: Pythia-160M teacher, Pythia-70M student, WikiText-2 validation prompts, 4 outer steps, 10 inner steps per outer, 16 prompts, 16 generated tokens, AdamW learning rate `2e-5`, linear eta from `0.25` to `0.65`, seed `20261007`.

The metrics are sampled-continuation diagnostics. `kl_student_teacher` is the empirical token log-ratio difference on the frozen rollouts, not a full-vocabulary KL. `residual_rms` measures the current inner residual against the sampled target. PMD-Part uses the exact full-vocabulary categorical log-partition for each sampled prefix; PMD-Mean uses the anchor-batch mean and therefore includes its population target bias; BPO uses a centered residual; MDPO uses an anchor-rollout policy-gradient estimator for the KL surrogate.

Final outer rows:

| method   | student NLL | teacher NLL on same tokens | sampled KL(student\|teacher) | residual RMS |
| -------- | ----------: | -------------------------: | ---------------------------: | -----------: |
| PMD-Part |      3.1120 |                     3.5190 |                       0.4070 |       0.2459 |
| PMD-Mean |      2.6458 |                     3.4465 |                       0.8007 |       0.5051 |
| BPO      |      3.6251 |                     3.7619 |                       0.1368 |       0.2483 |
| MDPO     |      3.5624 |                     3.7974 |                       0.2350 |       0.3148 |

These four steps are an implementation audit, not a claim of convergence or superiority. MDPO was first found to diverge under an incorrectly signed sampled surrogate; the corrected estimator is the one reported above.
