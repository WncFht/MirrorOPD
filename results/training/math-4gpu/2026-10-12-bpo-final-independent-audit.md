# BPO final independent audit

Audited 2026-10-12. Run: runs/dapo-deepseek-bpo-o50-j16. Raw logs, source startup provenance, resolved configuration, model/data audit, GPU/host records, exit code and independent live W&B history were inspected. No training modifications, GPU workloads or commits.

Formal BPO is complete: exit-code.txt=0;800 unique inner rows exactly enumerate(outer_call,inner_step)=0..49×0..15;51 unique outer rows enumerate step0..50; rollout versions0..49 occur in order. All inner epochs16 and mini-batch64. Eleven evaluations occur at0,5,...,50 and each generates570 examples. All800 inner metric values are finite. Independent repository-NETRC W&B API confirms gp7pyuj6 statefinished,851 history rows, unique inner_update0..799 and eleven evaluation axes. Remote final accuracies/truncation equal local raw logs; evidence is saved in bpo-final-remote-audit.json.

Requested DeepSeek-R1-Distill-Qwen-1.5B student and JustRL-DeepSeek-1.5B teacher use checked shared tokenization/native chat and the original DAPO/MATH-500/AMC23/AIME25 datasets. All four current parquet hashes match launch model-data-audit.json. Resolved mirror_bpo, eta0.3/K16, batch64/mini64/micro1, n1/epochs16, fixed lr1e-6, clip2, FP32 parameter/AdamW storage with BF16 forward and FP32 reductions, gradient checkpointing and parameter/optimizer offload match the common configuration. Training sampling uses temperature1/top-p1/top-k−1; greedy evaluation uses temperature0. Prompt2048/response7168/context9216. Task reward and auxiliary PPO policy-gradient updates are disabled.

Startup source commit is ac54a612fb9ef0692b5b65d6c7dae620bdbb39a9. All47 first-party hashes in launch source-provenance.json match their versions at that Git commit. Current source has subsequent edits across thirteen recorded paths, including core/metrics/tracking/config/launchers. This audit verifies the startup version rather than asserting current files are identical. At that version the stopped-gradient BPO coefficient is (U+student_center)/eta−anchor_reward−teacher_center, agreeing with the existing book-old algorithm audit's Tail-aware analytic BPO update.

All50 outers start at the fresh anchor: first-inner student-center RMS≤1.07826e−7, sampled-log-ratio RMS≤1.12999e−7, ESS fraction≥0.9999999999999862 and overlap1. Teacher-center mean, sampled frozen-anchor reward mean, anchor head/tail mass means and valid-token counts have zero variation within every outer. Thus16 inner updates reuse frozen cache followed by a fresh outer sync. These metrics verify observable cache/sync invariants.

One of800 updates activates gradient clip2: outer_call11,inner_step3 has preclip grad norm4.166000366 and scale0.480076770. All other clip scales are1; global grad norm minimum0.03025498. Importance ESS fraction minimum0.99209253, mean0.99869156; maximum observed individual-token ratio29.95295334. Support-overlap mean minimum0.96483021. High aggregate ESS supports distributed weights despite the larger individual ratio maximum. No ratio clipping is silently applied; the one reported clip is global gradient clipping.

## Outer trends and benchmark results

KL/reward windows use inner_step0 for the first ten OUTERS versus the last ten OUTERS; frozen per-inner copies are not treated as independent outer observations. Verifier/truncation windows use outer step1..10 versus41..50.

| Measurement                                     | First10 outers | Last10 outers |
| ----------------------------------------------- | -------------: | ------------: |
| Fresh-anchor tail-aggregated student→teacher KL |   0.0443187578 |  0.0028074352 |
| Sampled teacher-minus-anchor log reward         |  -0.0473004427 | -0.0025550521 |
| Mathematical verifier reward                    |       0.478125 |     0.5640625 |
| Training response-budget clipping               |       0.490625 |      0.396875 |

The center is a16-token-plus-tail lower bound measured on changing prefix distributions, not exact complete-vocabulary KL on fixed prefixes. All inner metrics precede their optimizer update. A decreasing center supports alignment on encountered prefixes; the scalar stopped-gradient surrogate value cannot prove convergence.

| Benchmark |         Initial |           Final | Correct-answer change | Initial/final budget exhaustion |
| --------- | --------------: | --------------: | --------------------: | ------------------------------: |
| MATH-500  | 346/500 (69.2%) | 376/500 (75.2%) |          +30 (+6.0pp) |                    28.4% /27.0% |
| AMC23     |   19/40 (47.5%) |   24/40 (60.0%) |          +5 (+12.5pp) |                    52.5% /52.5% |
| AIME25    |   5/30 (16.67%) |   4/30 (13.33%) |          −1 (−3.33pp) |                   80.0% /83.33% |

The single completed run demonstrates learning and MATH/AMC improvement under7168-token budget. It does not improve every benchmark: AIME loses one correct answer and25/30 final responses exhaust budget; AMC21/40 responses exhaust budget. Small test sets, truncation, one seed and differing initial scores across formal runs limit general conclusions. On-policy50 updates versus MDPO/BPO800 updates are unequal optimizer budgets and cannot establish a general method winner.

GPU samples cover indices0–3 only, maximum36765MiB (35.90GiB); launcher separately restricts CUDA_VISIBLE_DEVICES=0,1,2,3. Across5008 host records, individual-process RSS peaks10.786GiB and available system memory stays above2575.317GiB; these snapshots are not aggregate job RSS. Sum of50 recorded step and periodic-evaluation times is13196.797seconds, excluding initial evaluation/startup/cleanup.

All three formal training stages now have independently audited terminal success and complete remote histories. Overall goal completion additionally requires the leader's requirement-by-requirement audit of requested hyperparameter records, maintained operations document and final comparison deliverables.
