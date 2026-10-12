"""Complete 570-by-16 validation evidence and real backend response preservation."""

# ruff: noqa: PT009, PT027

import ast
import asyncio
import copy
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mirroropd.training.evaluation import (
    audit_evaluation_records,
    audit_generation_rows,
    effective_prompt_length,
    effective_response_length,
    evaluation_protocol,
    generation_metadata,
    phase_data_config,
    stable_question_uids,
)

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(os.environ.get("MIRROR_EVAL_BACKEND_ROOT", ROOT / "upstream/TA-OPD")) / "verl/verl"


def config(repeats=16):
    return {
        "data": {"max_prompt_length": 2048, "val_max_prompt_length": 4096},
        "trainer": {"validation_data_dir": None},
        "actor_rollout_ref": {
            "rollout": {
                "response_length": 7168,
                "max_model_len": 35840,
                "val_kwargs": {
                    "n": repeats,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": -1,
                    "do_sample": True,
                    "max_tokens": 31744,
                },
            }
        },
    }


def real_method(path, name, namespace):
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    function.decorator_list = []
    function.returns = None
    for argument in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]:
        argument.annotation = None
    if function.args.vararg:
        function.args.vararg.annotation = None
    if function.args.kwarg:
        function.args.kwarg.annotation = None
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class MathEvaluationTest(unittest.TestCase):
    def test_training_and_validation_budgets_are_independent(self):
        cfg = config()
        rollout = cfg["actor_rollout_ref"]["rollout"]
        self.assertEqual(effective_response_length(rollout, False), 7168)
        self.assertEqual(effective_response_length(rollout, True), 31744)
        self.assertEqual(effective_prompt_length(cfg["data"], False), 2048)
        self.assertEqual(effective_prompt_length(cfg["data"], True), 4096)
        validation_data = phase_data_config(cfg["data"], True)
        self.assertEqual(validation_data["max_prompt_length"], 4096)
        self.assertEqual(cfg["data"]["max_prompt_length"], 2048)
        rollout["max_model_len"] = 9216
        with self.assertRaisesRegex(ValueError, "context"):
            evaluation_protocol(cfg)

    def test_future_math_config_matches_mdopd_sampling(self):
        values = dict(
            argument.split("=", 1)
            for argument in json.loads(
                (ROOT / "training/configs/dapo-deepseek-4gpu.json").read_text()
            )["hydra_overrides"]
        )
        self.assertEqual(values["data.max_response_length"], "7168")
        self.assertEqual(values["data.max_prompt_length"], "2048")
        self.assertEqual(values["+data.val_max_prompt_length"], "4096")
        for key, value in {
            "n": "16",
            "temperature": "0.7",
            "top_p": "0.9",
            "top_k": "-1",
            "max_tokens": "31744",
            "do_sample": "true",
        }.items():
            self.assertEqual(values[f"actor_rollout_ref.rollout.val_kwargs.{key}"], value)
        self.assertEqual(values["trainer.validation_data_dir"], "{run_dir}/validation")
        self.assertEqual(values["actor_rollout_ref.rollout.max_model_len"], "35840")

    def test_stable_uid_and_namespace(self):
        sources = ["MATH-500", "AMC23"]
        prompts = [[{"role": "user", "content": "same question"}]] * 2
        a = stable_question_uids(sources, prompts)
        self.assertEqual(a, stable_question_uids(sources, prompts))
        self.assertNotEqual(a[0], a[1])
        with self.assertRaisesRegex(ValueError, "unique"):
            stable_question_uids(["MATH-500"] * 2, prompts)

    def make_rows(self, repeats=16):
        cfg = config(repeats)
        protocol = evaluation_protocol(cfg)
        rows = []
        for source, questions in (("MATH-500", 500), ("AMC23", 40), ("AIME25", 30)):
            for question in range(questions):
                for sample in range(repeats):
                    rows.append(  # noqa: PERF401 - nested suite/UID/sample order is part of this fixture
                        {
                            "uid": f"{source}:{question}",
                            "data_source": source,
                            "sample_index": sample,
                            "score": int(question % 2 == 0 and sample == 0),
                            "finish_reason": "stop",
                            "response_tokens": 8000,
                            "protocol_id": protocol["protocol_id"],
                            "step": 0,
                        }
                    )
        return cfg, protocol, rows

    def test_full_suite_9120_samples_and_mean_pass_are_distinct(self):
        _, protocol, rows = self.make_rows()
        metrics = audit_generation_rows(rows, protocol, 570)
        self.assertEqual(metrics["evaluation/generated_examples"], 9120)
        self.assertEqual(metrics["evaluation/unique_questions"], 570)
        self.assertEqual(metrics["val-core/MATH-500/acc/mean@16"], 1 / 32)
        self.assertEqual(metrics["val-core/MATH-500/acc/pass@16"], 1 / 2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            audit_generation_rows(rows[:-1], protocol, 570)
        broken = copy.deepcopy(rows)
        broken[1]["sample_index"] = 0
        with self.assertRaisesRegex(ValueError, "sample index"):
            audit_generation_rows(broken, protocol, 570)

    def test_count_preserving_uid_corruption_fails(self):
        _, protocol, rows = self.make_rows()
        rows[16]["uid"] = rows[0]["uid"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            audit_generation_rows(rows, protocol, 570)

    def test_saved_evidence_is_required_and_matches_summary(self):
        cfg, protocol, rows = self.make_rows()
        metrics = audit_generation_rows(rows, protocol, 570)
        metrics["evaluation/protocol_id"] = protocol["protocol_id"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "saved"):
                audit_evaluation_records(directory, cfg, [{"step": 0, "data": metrics}], 570)
            cfg["trainer"]["validation_data_dir"] = directory
            path = Path(directory) / "0.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(
                audit_evaluation_records(directory, cfg, [{"step": 0, "data": metrics}], 570)[
                    "evaluation_examples"
                ],
                9120,
            )
            metrics["val-core/MATH-500/acc/mean@16"] = 0.5
            with self.assertRaisesRegex(ValueError, "aggregate"):
                audit_evaluation_records(directory, cfg, [{"step": 0, "data": metrics}], 570)

    def test_legacy_greedy_n1_without_uid_artifacts_is_supported(self):
        cfg = config(1)
        kwargs = cfg["actor_rollout_ref"]["rollout"]["val_kwargs"]
        kwargs.pop("max_tokens")
        kwargs.update(do_sample=False, temperature=0, top_p=1)
        cfg["data"].pop("val_max_prompt_length")
        cfg["actor_rollout_ref"]["rollout"]["max_model_len"] = 9216
        self.assertEqual(
            audit_evaluation_records(
                "unused", cfg, [{"step": 0, "data": {"evaluation/generated_examples": 570}}], 570
            )["evaluation_examples"],
            570,
        )

    def test_generation_metadata_exposes_actual_abort(self):
        metadata = generation_metadata(["q", "q"], ["M", "M"], ["stop", "length"])
        self.assertEqual(metadata["sample_index"], [0, 1])
        with self.assertRaisesRegex(ValueError, "completed"):
            generation_metadata(["q"], ["M"], ["abort"])

    def test_real_validation_metric_reduction_excludes_identity_strings(self):
        def scoring_only(_sources, _uids, infos):
            self.assertNotIn("uid", infos)
            self.assertNotIn("finish_reason", infos)
            self.assertNotIn("protocol_id", infos)
            return {"M": {"acc": {"mean@16": 0.03125}}}

        namespace = {
            "process_validation_metrics": scoring_only,
            "np": np,
        }
        metrics_update = real_method(
            BACKEND / "trainer/ppo/ray_trainer.py", "_val_metrics_update", namespace
        )
        cfg = config()
        cfg["trainer"]["mirror_math_evaluation"] = True
        worker = SimpleNamespace(config=SimpleNamespace(**cfg), val_dataset=[0, 1])
        uids = ["q0"] * 16 + ["q1"] * 16
        sources = ["M"] * 32
        infos = generation_metadata(uids, sources, ["stop"] * 32)
        infos["reward"] = [1] + [0] * 31
        infos["protocol_id"] = [evaluation_protocol(cfg)["protocol_id"]] * 32
        result = metrics_update(worker, sources, uids, infos, [])
        self.assertEqual(result["val-core/M/acc/mean@16"], 1 / 32)
        self.assertEqual(result["val-core/M/acc/pass@16"], 1 / 2)
        self.assertEqual(result["evaluation/generated_examples"], 32)

    def test_real_single_turn_preserves_long_validation_after_server_pops_max_tokens(self):
        namespace = {
            "AgentLoopOutput": SimpleNamespace,
            "simple_timer": lambda *_args: nullcontext(),
            "uuid4": lambda: SimpleNamespace(hex="test"),
        }
        run = real_method(
            BACKEND / "experimental/agent_loop/single_turn_agent_loop.py", "run", namespace
        )

        class Agent:
            response_length = 7168

            async def process_multi_modal_info(self, _messages):
                return {}

            def _get_mm_processor_kwargs(self, _audios):
                return {}

            async def apply_chat_template(self, _messages, **_kwargs):
                return [1, 2]

            async def generate(self, **kwargs):
                kwargs["sampling_params"].pop("max_tokens", None)
                return SimpleNamespace(
                    token_ids=[3] * 8000,
                    log_probs=[-0.5] * 8000,
                    routed_experts=None,
                    num_preempted=0,
                    extra_fields={"generation_finish_reason": "stop"},
                )

        agent = Agent()
        agent.server_manager = agent
        validate = asyncio.run(run(agent, {"max_tokens": 31744}, raw_prompt=[]))
        train = asyncio.run(run(agent, {}, raw_prompt=[]))
        self.assertEqual(len(validate.response_ids), 8000)
        self.assertEqual(len(validate.response_logprobs), 8000)
        self.assertEqual(len(train.response_ids), 7168)

    def test_real_postprocess_uses_validation_prompt_and_response_padding(self):
        namespace = {
            "torch": torch,
            "np": np,
            "_InternalAgentLoopOutput": SimpleNamespace,
            "effective_response_length": effective_response_length,
            "effective_prompt_length": effective_prompt_length,
        }
        postprocess = real_method(
            BACKEND / "experimental/agent_loop/agent_loop.py", "_agent_loop_postprocess", namespace
        )

        worker_config = config()

        class Worker:
            config = SimpleNamespace(data=worker_config["data"])
            rollout_config = worker_config["actor_rollout_ref"]["rollout"]

            def _pad_token_ids(self, tokens, *, max_length, padding_side, return_attention_mask):
                self.assert_length = len(tokens) <= max_length
                if not self.assert_length:
                    raise ValueError("Silent truncation would be required")
                padding = [0] * (max_length - len(tokens))
                values = padding + tokens if padding_side == "left" else tokens + padding
                result = {"input_ids": torch.tensor([values])}
                if return_attention_mask:
                    mask = [1] * len(tokens)
                    result["attention_mask"] = torch.tensor(
                        [padding + mask if padding_side == "left" else mask + padding]
                    )
                return result

            def _compute_multi_modal_inputs(self, _output, _input_ids):
                return {}

            def _compute_position_ids(self, _input_ids, attention_mask, *_args):
                return attention_mask.cumsum(-1)

            async def _compute_score(self, _outputs, **_kwargs):
                return None

            async def _compute_teacher_logprobs(self, _output, **kwargs):
                self.teacher_validate = kwargs["validate"]

        output = SimpleNamespace(
            prompt_ids=[1] * 3000,
            response_ids=[2] * 8000,
            response_mask=[1] * 8000,
            response_logprobs=[-0.2] * 8000,
            routed_experts=None,
            mm_processor_kwargs={},
            multi_modal_data={},
            extra_fields={},
            reward_score=1,
            num_turns=2,
            metrics={},
        )
        worker = Worker()
        result = asyncio.run(postprocess(worker, output, True, raw_prompt=[]))
        self.assertEqual(tuple(result.prompt_ids.shape), (1, 4096))
        self.assertEqual(tuple(result.response_ids.shape), (1, 31744))
        self.assertEqual(tuple(result.input_ids.shape), (1, 35840))
        self.assertEqual(result.response_mask.sum().item(), 8000)
        self.assertEqual(result.response_logprobs[0, 7999].item(), np.float32(-0.2))
        self.assertTrue(worker.teacher_validate)


if __name__ == "__main__":
    unittest.main()
