"""Reject an undeployed validation integration before starting a new math run."""

import argparse
import ast
import json
import shlex
from pathlib import Path


def preparation_commands():
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / "upstream-manifest.json").read_text())
    item = next(checkout for checkout in manifest["checkouts"] if checkout["name"] == "TA-OPD")
    destination = repo / "runs/math-evaluation-backend"
    q = shlex.quote
    return (
        "Prepare a fresh idle checkout from upstream-manifest.json (the same base as make upstream):\n"
        f"git clone --no-checkout {q(item['url'])} {q(str(destination))}\n"
        f"git -C {q(str(destination))} checkout --detach {q(item['base_commit'])}\n"
        f"git -C {q(str(destination))} apply {q(str(repo / item['patch']))}\n"
        f"MIRROR_UPSTREAM_ROOT={q(str(destination))} bash {q(str(repo / 'training/run_dapo_math_4gpu.sh'))} "
        "mirror_mdpo NEW_RUN_NAME\n"
        "Keep active backends unchanged; check GPU capacity with a separate canary before a full run."
    )


def check_backend(root):
    source = root / "verl/verl"
    required = {
        "workers/config/rollout.py": ("max_tokens: Optional[int] = None",),
        "experimental/agent_loop/agent_loop.py": (
            'sampling_params["max_tokens"] = effective_response_length(config, True)',
            "max_length=response_length",
            'phase_data_config(self.config.data, trajectory["validate"])',
            "effective_prompt_length(self.config.data, validate)",
            'non_tensor_batch["generation_finish_reason"]',
        ),
        "experimental/agent_loop/single_turn_agent_loop.py": (
            'sampling_params.get("max_tokens")',
            "output.token_ids[:response_length]",
        ),
        "workers/rollout/vllm_rollout/vllm_async_server.py": (
            'extra_fields["generation_finish_reason"] = finish_reason',
        ),
        "trainer/ppo/ray_trainer.py": (
            "phase_data_config(self.config.data, True)",
            "stable_question_uids(",
            "complete_sample_metrics(",
            "generation_metadata(",
        ),
    }
    for relative, fragments in required.items():
        path = source / relative
        if not path.is_file():
            raise ValueError(
                f"Validation backend source is missing: {path}.\n{preparation_commands()}"
            )
        text = path.read_text()
        ast.parse(text, filename=str(path))
        for fragment in fragments:
            if "".join(fragment.split()) not in "".join(text.split()):
                raise ValueError(
                    f"Backend lacks independent math validation: {path}: {fragment}. "
                    "\n" + preparation_commands()
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", type=Path)
    args = parser.parse_args()
    check_backend(args.backend)
    print("Math validation backend integration is present; GPU capacity remains unverified")


if __name__ == "__main__":
    main()
