"""Execute pinned native control-flow boundaries without CUDA or a Ray cluster."""

import __future__

import ast
import copy
import os
import textwrap
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mirroropd.runtime.evaluation import question_uids


@pytest.fixture(scope="module")
def backend():
    path = Path(os.environ.get("MIRROR_TEST_BACKEND", "upstream/TA-OPD-v2")) / "verl/verl"
    if not path.is_dir():
        pytest.fail(
            "Prepare the pinned backend and set MIRROR_TEST_BACKEND before integration tests"
        )
    return path


def native_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = copy.deepcopy(
        next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)  # noqa: S102 -- execute only the pinned local function under test
    return namespace[name]


def test_direct_distillation_never_calls_a_discarded_ppo_objective(backend):
    calls = []

    def populate(config, data):
        config.global_batch_info.update(data)
        calls.append("global")

    def distill(config, *_args):
        assert config.global_batch_info["batch_num_tokens"] == 19
        calls.append("loss")
        return 0.25, {}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("PPO objective must not execute")

    namespace = {
        "mirror": SimpleNamespace(is_enabled=lambda _cfg: True, prepare_global_batch=populate),
        "distillation_loss": distill,
        "ppo_loss": forbidden,
        "Metric": lambda **kwargs: kwargs,
        "AggregationType": SimpleNamespace(SUM="sum"),
    }
    function = native_function(
        backend / "trainer/distillation/losses.py", "distillation_ppo_loss", namespace
    )
    config = SimpleNamespace(global_batch_info={})
    result = function(
        config, SimpleNamespace(distillation_loss=object()), {}, {"batch_num_tokens": 19}, None
    )
    assert result[0] == 0.25
    assert calls == ["global", "loss"]


def test_real_training_generation_keeps_weight_version_and_sample_coordinates(backend):
    source = (backend / "trainer/ppo/ray_trainer.py").read_text()
    begin = source.index(
        "                batch: DataProto = DataProto.from_single_dict(batch_dict)"
    )
    end = source.index(
        "                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:", begin
    )
    snippet = textwrap.dedent(source[begin:end])

    class Batch:
        def __init__(self, n=2):
            self.batch = [None] * n
            self.meta_info = {}
            self.non_tensor_batch = {
                "data_source": np.array(["source"] * n),
                "raw_prompt": np.array(["q1", "q2"][:n]),
                "uid": np.array(["row1", "row2"][:n]),
            }

        def __len__(self):
            return len(self.batch)

        def repeat(self, repeat_times, interleave):
            assert interleave
            result = Batch(0)
            result.batch = [None] * (len(self) * repeat_times)
            result.meta_info = dict(self.meta_info)
            result.non_tensor_batch = {
                key: np.repeat(value, repeat_times) for key, value in self.non_tensor_batch.items()
            }
            return result

    def get_gen(batch):
        result = copy.deepcopy(batch)
        result.meta_info = {}
        return result

    trainer = SimpleNamespace(
        mirror_enabled=True,
        global_steps=3,
        _get_gen_batch=get_gen,
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=1.0, n=2))
        ),
    )
    namespace = {
        "self": trainer,
        "DataProto": SimpleNamespace(from_single_dict=lambda _value: Batch()),
        "batch_dict": {},
        "np": np,
        "uuid": uuid,
        "question_uids": question_uids,
    }
    exec(compile(snippet, "native-generation-coordinates", "exec"), namespace)  # noqa: S102 -- exact pinned native generation boundary
    generated = namespace["gen_batch_output"]
    assert generated.meta_info["mirror_generation_version"] == 2
    assert generated.meta_info["global_steps"] == 3
    assert generated.non_tensor_batch["sample_index"].tolist() == [0, 1, 0, 1]


def test_native_integration_has_one_current_mode_and_no_legacy_imports(backend):
    source = (backend / "trainer/distillation/losses.py").read_text()
    module = ast.parse(source)
    marker = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MIRROR_API_VERSION"
            for target in node.targets
        )
    )
    assert marker.value.value == 2
    for path in backend.rglob("*.py"):
        text = path.read_text()
        assert "mirroropd.training" not in text, path
        assert '"mirror_opd_onpolicy"' not in text, path
        assert '"mirror_mdpo"' not in text, path
