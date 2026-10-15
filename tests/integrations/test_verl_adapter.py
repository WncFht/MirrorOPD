"""Prediction alignment, ragged transport and real model parameter gradients."""

import copy

import pytest
import torch

from mirroropd.core.cache import build_cache
from mirroropd.core.config import CacheContext, Estimator, Objective
from mirroropd.core.vocabulary import align_logits, validate_tokenizers
from mirroropd.integrations.verl.adapter import (
    cache_outputs,
    loss_outputs,
    prediction_fields,
    restore_predictions,
    unpad_cache_fields,
)


@pytest.mark.parametrize(
    ("center", "integration"), [("topj_tail", "sampled"), ("full", "sampled"), ("full", "full")]
)
@pytest.mark.parametrize("algorithm", ["on_policy", "mdpo", "bpo"])
def test_padded_and_ragged_adapters_preserve_last_budget_action_and_gradients(
    center, integration, algorithm
):
    torch.manual_seed(17)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 6, 7]])
    attention = ids.ne(0)
    response = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    initial = prediction_fields(ids, attention, response)
    assert initial["mirror_cache_valid"].sum() == 7
    assert initial["mirror_cache_actions"][1, 5] == 7
    objective = Objective(algorithm, estimator=Estimator(center, integration, 2))
    context = CacheContext(
        0, "teacher", "vocabulary", 11, 1.0, objective.signal, objective.estimator
    )
    q, teacher = (
        torch.randn(2, 7, 11, dtype=torch.float64),
        torch.randn(2, 7, 11, dtype=torch.float64),
    )
    indices = attention.reshape(-1).nonzero().squeeze(-1)
    offsets = torch.tensor([0, 5, 12])
    flat_q = q.reshape(-1, 11).index_select(0, indices)
    flat_teacher = teacher.reshape(-1, 11).index_select(0, indices)
    prediction = unpad_cache_fields(initial, indices, offsets)
    anchor = cache_outputs(
        flat_q, prediction, "anchor", context, diagnostics="full", sequence_parallel_size=1
    )
    padded = initial | {
        key: restore_predictions(value, attention.shape, indices) for key, value in anchor.items()
    }
    teacher_input = unpad_cache_fields(padded, indices, offsets)
    completed = cache_outputs(
        flat_teacher,
        teacher_input,
        "teacher",
        context,
        diagnostics="full",
        sequence_parallel_size=1,
    )
    padded.update(
        {
            key: restore_predictions(value, attention.shape, indices)
            for key, value in completed.items()
        }
    )
    z = q.clone().requires_grad_()
    normal = loss_outputs(
        z,
        padded,
        objective,
        context,
        anchor_version=0,
        data_format="bshd",
        sequence_parallel_size=1,
    )
    ragged = unpad_cache_fields(padded, indices, offsets)
    flat_z = z.reshape(-1, 11).index_select(0, indices).unsqueeze(0)
    packed = loss_outputs(
        flat_z,
        ragged,
        objective,
        context,
        anchor_version=0,
        data_format="thd",
        sequence_parallel_size=1,
    )
    torch.testing.assert_close(
        normal["distillation_losses"][attention], packed["distillation_losses"].squeeze(0)
    )
    normal_grad = torch.autograd.grad(normal["distillation_losses"].sum(), z, retain_graph=True)[0]
    packed_grad = torch.autograd.grad(packed["distillation_losses"].sum(), z)[0]
    torch.testing.assert_close(normal_grad, packed_grad)
    assert not normal_grad[~initial["mirror_cache_valid"]].any()
    with pytest.raises(ValueError, match="Stale"):
        loss_outputs(
            z,
            padded,
            objective,
            context,
            anchor_version=1,
            data_format="bshd",
            sequence_parallel_size=1,
        )


@pytest.mark.parametrize("algorithm", ["on_policy", "mdpo", "bpo"])
def test_distinct_model_widths_share_vocabulary_and_match_independent_parameter_gradient(algorithm):
    torch.manual_seed(27)
    anchor = torch.nn.Sequential(torch.nn.Embedding(11, 7), torch.nn.Linear(7, 11)).double()
    teacher = torch.nn.Sequential(
        torch.nn.Embedding(11, 13), torch.nn.Tanh(), torch.nn.Linear(13, 11)
    ).double()
    student = copy.deepcopy(anchor)
    inputs = torch.tensor([[1, 2, 3]])
    qlog = anchor(inputs).log_softmax(-1).detach()
    tlog = teacher(inputs).log_softmax(-1).detach()
    objective = Objective(algorithm, estimator=Estimator("full", "sampled", 2))
    context = CacheContext(
        0, "different-model", "shared-vocabulary", 11, 1.0, objective.signal, objective.estimator
    )
    actions = torch.arange(11).unsqueeze(-1).expand(-1, 3)
    cache = build_cache(
        qlog.expand(11, -1, -1),
        tlog.expand(11, -1, -1),
        actions,
        torch.ones_like(actions, dtype=torch.bool),
        context,
    )
    z = student(inputs).expand(11, -1, -1)
    result = loss_outputs(
        z,
        cache.to_wire(),
        objective,
        context,
        anchor_version=0,
        data_format="bshd",
        sequence_parallel_size=1,
    )
    weights = qlog.exp().squeeze(0).T
    surrogate = (weights * result["distillation_losses"]).sum() / 3
    actual = torch.autograd.grad(surrogate, tuple(student.parameters()), retain_graph=True)
    raw = (0.8 * qlog.exp().sqrt() + 0.2 * tlog.exp().sqrt()).square()
    target = (raw / raw.sum(-1, keepdim=True)).log()
    logp = student(inputs).log_softmax(-1)
    if algorithm == "bpo":
        residual = logp - target
        residual -= (qlog.exp() * residual).sum(-1, keepdim=True)
        reference = (qlog.exp() * residual.square()).sum(-1).mean() / 0.4
    else:
        reference = (logp.exp() * (logp - target)).sum(-1).mean() / 0.2
    expected = torch.autograd.grad(reference, tuple(student.parameters()))
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=2e-13, rtol=2e-12)
    assert all(
        parameter.grad is None for parameter in (*anchor.parameters(), *teacher.parameters())
    )


def test_padded_output_rows_are_not_legal_actions_or_gradient_receivers():
    student = torch.randn(2, 13, requires_grad=True)
    teacher = torch.randn(2, 17)
    objective = Objective(estimator=Estimator("full", "full", 2))
    context = CacheContext(
        0, "teacher", "legal-vocabulary", 11, 1.0, objective.signal, objective.estimator
    )
    legal = align_logits(student, 11)
    cache = build_cache(
        legal.detach(),
        align_logits(teacher, 11),
        torch.tensor([1, 2]),
        torch.ones(2, dtype=torch.bool),
        context,
    )
    from mirroropd.core.objectives import compute_loss

    compute_loss(legal, cache, objective, anchor_version=0).loss.sum().backward()
    assert not student.grad[:, 11:].any()


def test_prompt_attention_and_tokenizer_contracts_fail_early():
    with pytest.raises(ValueError, match="attended predictors"):
        prediction_fields(
            torch.ones(1, 3, dtype=torch.long),
            torch.tensor([[0, 1, 1]]),
            torch.ones(1, 2, dtype=torch.bool),
        )

    class Tokenizer:
        bos_token_id = 0
        eos_token_id = 1
        pad_token_id = 0
        chat_template = "template"

        def get_vocab(self):
            return {"a": 0, "b": 1}

    assert validate_tokenizers(Tokenizer(), Tokenizer()).startswith("sha256:")
    other = Tokenizer()
    other.chat_template = "other"
    with pytest.raises(ValueError, match="shared chat template"):
        validate_tokenizers(Tokenizer(), other)
