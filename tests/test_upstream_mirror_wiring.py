"""Execute upstream trainer cache routing with CPU worker stand-ins.

Original trainer/padding bodies are executed. Ray transport and flash-attn
unpadding are replaced, so this test does not establish distributed readiness.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from test_upstream_opd_semantics import load_functions


class Batch:
    def __init__(self, values):
        self.batch = values

    def to_tensordict(self):
        return dict(self.batch)

    @classmethod
    def from_tensordict(cls, values):
        return cls(values)


class Config(dict):
    __getattr__ = dict.__getitem__


def load_method(path, classname, name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method.decorator_list = []
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


def main():
    root = Path(__file__).resolve().parents[1] / "upstream/TA-OPD/verl/verl"
    check_mode_routing(root)
    check_rollout_guards(root)
    check_rollout_versions(root)
    tu = SimpleNamespace(
        assign_non_tensor_data=lambda data, key, value: data.__setitem__(key, value),
        get_non_tensor_data=lambda data, key, default: data.get(key, default),
        assign_non_tensor=lambda data, **kwargs: data.update(kwargs),
        get=lambda data, key: data.get(key),
        get_tensordict=dict,
    )

    def unpad(value, mask):
        indices = mask.flatten().nonzero().flatten()
        lengths = mask.sum(-1)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
        return value.flatten(0, 1)[indices], indices, offsets

    def index_first_axis(value, indices):
        # Match the actual FlashAttention interface, including scalar-cache shape.
        assert value.ndim >= 2
        return value[indices]

    ns = {
        "torch": torch,
        "TensorDict": dict,
        "DataProto": Batch,
        "F": torch.nn.functional,
        "tu": tu,
        "index_first_axis": index_first_axis,
        "unpad_input": unpad,
        "get_ulysses_sequence_parallel_world_size": lambda: 1,
    }
    load_functions(
        root / "workers/utils/padding.py",
        [
            "build_mirror_prediction_fields",
            "left_right_2_no_padding",
            "no_padding_2_full_padding",
            "no_padding_2_padding",
        ],
        ns,
    )
    load_functions(
        root / "trainer/distillation/fsdp/losses.py",
        [
            "build_mirror_anchor_cache",
            "complete_mirror_teacher_cache",
            "compute_mirror_cache_from_logits",
        ],
        ns,
    )
    old = load_method(
        root / "trainer/ppo/ray_trainer.py",
        "RayPPOTrainer",
        "_compute_old_log_prob",
        ns,
    )
    teacher = load_method(
        root / "trainer/ppo/ray_trainer.py",
        "RayPPOTrainer",
        "_compute_mirror_teacher_cache",
        ns,
    )
    ids = torch.tensor([[0, 1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10, 11]])
    attention = torch.tensor([[0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    batch = Batch(
        {
            "input_ids": ids,
            "attention_mask": attention,
            "response_mask": mask,
            "position_ids": attention.cumsum(-1) - 1,
            "prompts": ids[:, :3],
            "responses": ids[:, 3:],
        }
    )
    torch.manual_seed(51)
    logits = {"anchor": torch.randn(11, 16), "teacher": torch.randn(11, 16)}
    requests = []

    def score(data):
        stage = data["mirror_cache_stage"]
        requests.append(stage)
        assert data["compute_loss"] is False
        assert data["temperature"] == 0.7
        output = ns["compute_mirror_cache_from_logits"](
            logits[stage], data, stage, data.get("mirror_cache_topk")
        )
        offsets = data["input_ids"].offsets()
        output = {
            key: torch.nested.nested_tensor_from_jagged(value, offsets)
            for key, value in output.items()
        }
        output.update(
            log_probs=torch.nested.nested_tensor_from_jagged(torch.zeros(11), offsets),
            entropy=torch.nested.nested_tensor_from_jagged(torch.ones(11), offsets),
            metrics={"mfu": 0.0},
        )
        return output

    actor = {"calculate_sum_pi_squared": False}
    trainer = SimpleNamespace(
        mirror_opd=True,
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(actor=actor, rollout=SimpleNamespace(temperature=0.7))
        ),
        distillation_config=SimpleNamespace(distillation_loss=SimpleNamespace(topk=3)),
        actor_rollout_wg=SimpleNamespace(compute_log_prob=score),
        ref_policy_wg=SimpleNamespace(compute_ref_log_prob=score),
    )
    anchor, _ = old(trainer, batch)
    batch.batch.update(anchor.batch)
    completed = teacher(trainer, batch)
    assert requests == ["anchor", "teacher"]
    assert completed.batch["mirror_teacher_center"].shape == ids.shape
    assert completed.batch["mirror_teacher_topk_ids"].shape == (*ids.shape, 3)
    assert completed.batch["mirror_teacher_topk_ids"].dtype == torch.long
    assert not completed.batch["mirror_teacher_topk_ids"][~batch.batch["mirror_valid"]].any()
    assert batch.batch["mirror_anchor_ids"].shape == (*ids.shape, 3)
    assert batch.batch["mirror_valid"].sum().item() == mask.sum().item()
    assert "teacher_ids" not in batch.batch
    print(
        "PASS upstream trainer routing: actor anchor -> frozen teacher, full prediction positions, temperature metadata"
    )


def check_mode_routing(root):
    """Exercise actual mode/config branches without importing Ray or CUDA."""
    ns = {
        "Optional": Optional,
        "DistillationConfig": object,
        "ActorConfig": object,
        "DictConfig": object,
        "TensorDict": dict,
        "torch": torch,
        "Metric": lambda **kwargs: kwargs,
        "AggregationType": SimpleNamespace(SUM="sum"),
    }
    load_functions(
        root / "trainer/distillation/losses.py",
        [
            "is_distillation_enabled",
            "is_mirror_opd_enabled",
            "populate_global_batch_info",
            "distillation_ppo_loss",
        ],
        ns,
    )
    load_functions(
        root / "trainer/ppo/utils.py",
        ["need_reference_policy", "need_teacher_policy"],
        ns,
    )
    prepare = load_method(
        root / "workers/config/distillation.py",
        "DistillationConfig",
        "__post_init__",
        ns,
    )
    expected = torch.tensor(0.75, requires_grad=True)
    ns["distillation_loss"] = lambda *_args: (expected, {})

    def forbidden_ppo(*_args):
        raise AssertionError("MirrorOPD must not evaluate a PPO objective")

    ns["ppo_loss"] = forbidden_ppo
    for mode in ("mirror_pmd_mean", "mirror_bpo", "mirror_mdpo"):
        distill = SimpleNamespace(
            enabled=True,
            mirror_teacher_model_path="Qwen/Qwen3-1.7B",
            distillation_loss=SimpleNamespace(loss_mode=mode),
            n_gpus_per_node=0,
            nnodes=0,
            teacher_models={"teacher_model": {"model_path": None, "num_replicas": 0}},
        )
        prepare(distill)

        config = Config(
            distillation=distill,
            algorithm={"use_kl_in_reward": False},
            actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(use_kl_loss=False)),
            global_batch_info={},
            loss_scale_factor=None,
        )
        assert ns["need_reference_policy"](config)
        assert not ns["need_teacher_policy"](config)
        data = {
            "dp_size": 1,
            "batch_num_tokens": 37,
            "global_batch_size": 8,
        }
        loss, metrics = ns["distillation_ppo_loss"](config, distill, data=data)
        assert loss is expected
        assert "distillation/loss" in metrics
        assert config.global_batch_info == {
            "dp_size": 1,
            "batch_num_tokens": 37,
            "global_batch_size": 8,
            "loss_scale_factor": None,
        }
        for field, bad in (
            ("mirror_teacher_model_path", ""),
            ("n_gpus_per_node", 1),
            ("nnodes", 1),
        ):
            original = getattr(distill, field)
            setattr(distill, field, bad)
            try:
                prepare(distill)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted invalid MirrorOPD {field}")
            setattr(distill, field, original)
        distill.teacher_models["teacher_model"]["model_path"] = "native-teacher"
        try:
            prepare(distill)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted simultaneous native and MirrorOPD teachers")
        distill.distillation_loss.loss_mode = "k1"
        assert ns["need_teacher_policy"](config)
        assert not ns["need_reference_policy"](config)
    print(
        "PASS three-mode config gates, reference/teacher selection, PPO bypass, native mode routing"
    )


def check_rollout_guards(root):
    tree = ast.parse((root / "trainer/ppo/ray_trainer.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    guard = next(
        n
        for n in init.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Attribute)
        and n.test.attr == "mirror_opd"
    )
    function = ast.parse("def validate(config):\n    pass").body[0]
    function.body = guard.body
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "upstream-rollout-guards",
            "exec",
        ),
        namespace,
    )
    rollout = Config(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        do_sample=True,
        skip=Config(enable=False),
        quantization=None,
        multi_turn=Config(enable=False),
        agent=Config(
            default_agent_loop="single_turn_agent",
            agent_loop_config_path=None,
            custom_async_server=Config(path=None),
        ),
    )
    config = Config(
        actor_rollout_ref=Config(rollout=rollout, actor=Config(use_kl_loss=False, entropy_coeff=0)),
        algorithm=Config(adv_estimator="reinforce_plus_plus", use_kl_in_reward=False),
    )
    # These optional keys are absent from the actual upstream Hydra schema.
    assert "repetition_penalty" not in rollout
    assert "agent_loop_manager_class" not in rollout.agent
    namespace["validate"](config)
    for field, bad in (
        ("top_p", 0.9),
        ("top_k", 32),
        ("temperature", 0),
        ("temperature", float("nan")),
        ("do_sample", False),
    ):
        original = rollout[field]
        rollout[field] = bad
        try:
            namespace["validate"](config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid rollout {field}={bad}")
        rollout[field] = original
    print("PASS upstream rollout guard defaults and invalid sampling rejection")


def check_rollout_versions(root):
    namespace = {}
    load_functions(
        root / "trainer/distillation/losses.py",
        ["validate_mirror_rollout_version"],
        namespace,
    )
    validate = namespace["validate_mirror_rollout_version"]
    for version in (0, 1, 2):
        assert validate({"global_steps": [version, version]}, version, 2) == {
            "mirror/rollout_weight_version": version
        }
    for fields in (
        {},
        {"global_steps": [None, 2]},
        {"global_steps": [1, 2]},
        {"global_steps": [2]},
        {"global_steps": [3, 3]},
    ):
        try:
            validate(fields, 2, 2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"stale/missing version accepted: {fields}")
    print("PASS rollout server versions: fresh, stale, absent, mixed, resumed")


if __name__ == "__main__":
    main()
