# OPD inner 实现代码摘录

本文件只复制工作区中现有实现的核心片段，便于小型实验复用。原始仓库代码不在本目录修改；行号以摘录日期（2026-10-07）为准。

## MDPO sampled-token KL

来源：`/Users/fanghaotian/src/MD-OPD/verl/verl/workers/utils/losses.py:193-221`。

```python
if config.mdpo_kl_loss:
    mdpo_log_ratio = log_prob - old_log_prob.detach()
    mdpo_k1 = mdpo_log_ratio
    mdpo_k3 = kl_penalty(logprob=log_prob, ref_logprob=old_log_prob.detach(), kl_penalty="k3")
    mdpo_kl_loss = agg_loss(
        loss_mat=kl_penalty(
            logprob=log_prob, ref_logprob=old_log_prob.detach(), kl_penalty=config.kl_loss_type
        ),
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        **config.global_batch_info,
    )
    mdpo_kl_coef = 1.0 / mdpo_eta
    policy_loss += mdpo_kl_coef * mdpo_kl_loss
```

这里 `old_log_prob` 是 rollout 行为策略在实际采样 token 上的 log-prob。因此这是 sampled-token KL surrogate；它没有对完整词表求和，也没有 prefix importance ratio。它也没有显式的 action-level `rho=pi/old`：KL 项是在行为样本上直接用 k1/k3 或配置的 sampled estimator 计算的。`mdpo_eta` 由 `opd_trainer_mixin.py:89-112` 的 constant/linear schedule 产生。

这和理论上对 `E_pi[f] - KL(pi||b)/eta` 的无偏行为测度估计不同。后者需要在行为样本上乘 `rho=pi/b`；本目录 toy 中的 `mdpo_is` 分支才是这个 action-IS oracle，`mdpo` 分支才对应仓库的 sampled-KL 形态。两者必须分开报告。

## Plain OPD policy gradient

来源：`/Users/fanghaotian/src/MD-OPD/verl/verl/trainer/ppo/opd_core_algos.py:260-294`。

```python
if rollout_is_weights is not None:
    raise ValueError("opd_noratio does not support rollout importance-sampling weights")
pg_losses = -advantages.detach() * log_prob
pg_loss = agg_loss(
    loss_mat=pg_losses,
    loss_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
    **config.global_batch_info,
)
```

此项对应逐 token 的 `-A_t log pi_theta(y_t|s_t)`。其中 `advantages` 由教师 forward 预先计算，进入 actor loss 前停止梯度。

## PMD-Part-like residual regression

工作区没有一个单独命名为 PMD-Part 的官方实现；现有 MOPD residual 是其最接近的工程形式。来源：`opd_core_algos.py:336-411`。

```python
resid = anchor_eb_logp + (gap - eblogb) - (log_prob - teacher_lp)
if eta != 1.0:
    resid = resid + (eta - 1.0) * (teacher_lp - anchor_lp + gap)
loss_mat = resid.square() + floor_mat
pg_loss = agg_loss(
    loss_mat=loss_mat,
    loss_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
    **config.global_batch_info,
)
```

自动微分时 `anchor_eb_logp=E_b[log pi_theta]` 与 sampled `log_prob` 同时保留梯度，因而 logits 梯度为 `2*resid*(b-e_y)`。`teacher_lp`、`anchor_lp`、`gap`、`eblogb` 都是冻结量。该实现不支持 rollout IS。

## BPO-shaped weighted SFT

原始 BPO 论文（arXiv:2609.15987）没有公开独立代码仓库，因此不能把某个本地实现称为“BPO 官方代码”。`mopd_grpo` 是 MD-OPD 作者实现的 BPO-shaped OPD 适配，来源：`opd_core_algos.py:414-499`。它保留了 BPO 的 complement-probability 权重，但把终局 advantage 换成 OPD 的 `log teacher - log anchor + KL(anchor||teacher)`。以下是该适配的核心：

```python
adv = teacher_lp - anchor_lp + gap
loss_mat = -2.0 * eta * adv.detach() * log_prob
pg_loss = agg_loss(
    loss_mat=loss_mat,
    loss_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
    **config.global_batch_info,
)
```

完整实现还定义

```python
omega = ((1 + eps - exp(anchor_lp)) / (1 + eps - exp(log_prob).detach()).clamp_min(1e-8)).detach()
weight = omega.clamp_max(cap)
```

并将 `loss_mat` 改为 `-2*eta*adv*keep*weight*log_prob`。sequence experiment 使用这一 practical 形式；independent-row toy 才使用 `weight=1` 的回归对照。

## PMD-Mean 的实现状态

工作区没有 PMD-Mean 的独立训练函数。其可复用实现应在每个 prefix 上计算 full-vocabulary 行为均值，例如 `mean_r = (b * r).sum(-1)`，然后使用

```python
centered = reward - mean_r.detach()
loss = -(centered.detach() * log_prob).masked_select(response_mask).mean()
```

如果只有 sampled token，则 `mean_r` 需要用同 prefix 的多 action 样本、teacher top-k 补全或额外的 KL estimator 近似；单个 rollout token 不能无偏地恢复该条件均值。

## Teacher dense reward 与 top-k

教师 dense reward 的入口是 `opd_trainer_mixin.py:487-530`：

```python
teacher_output = self.opd_teacher_wg.compute_rm_score(batch)
teacher_log_probs = teacher_output.batch["teacher_log_probs"]
student_log_probs = batch.batch["old_log_probs"]
batch.batch["token_level_scores"] = teacher_log_probs - student_log_probs
```

top-k 的纯 tensor 实现位于 `fsdp_teacher_worker.py:105-205` 和 `235-272`。`compute_top_k_reward` 支持 `only_stu`、`only_tch`、`intersection`、`union`、`union-intersection`，并可用 `student_p`、`teacher_p` 或 `none` 作为候选权重。核心计算为：

```python
kl = student_log_probs - teacher_log_probs
weights = compute_reward_weights(...)
return -kl * weights
```

这只是候选集上的 weighted top-k reward；若需要完整 reverse KL，仍需 full-vocabulary logit reduction 或无偏 tail estimator。

## 四种 inner 在 toy experiment 中的统一接口

实验脚本可以让每个 inner 返回一个标量 loss，并由同一个 optimizer 做若干 inner updates：

```python
loss_pmd_part = 0.5 * ((log_pi - log_ref - h.detach()) ** 2).mean()
loss_pmd_mean = -(f - (b * f).sum(-1, keepdim=True)).detach() * log_pi
loss_bpo = -eta * delta.detach() * log_pi
loss_mdpo_sampled = -(f * log_pi).mean() + eta_inv * kl_sampled
loss_mdpo_is = -(rho * f * log_pi).mean() + eta_inv * (rho * kl_sampled)
```

所有 outer steps 都应重新冻结当前 student 作为 reference，重新生成 teacher target，并记录每个 inner update 的 target KL、teacher KL、loss、gradient norm 和 policy entropy。需要把 `mdpo_sampled` 与 `mdpo_is` 分开，不能用前者的结果声称 MDPO 使用了 IS。
