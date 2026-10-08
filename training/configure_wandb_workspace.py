"""Create a manual W&B workspace with ordered, pinned research diagnostics."""

import argparse

from wandb_workspaces.reports.v2 import LinePlot
from wandb_workspaces.workspaces import Section, Workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="wncfht")
    parser.add_argument("--project", default="mirror-opd-pythia")
    parser.add_argument("--url", help="Update this saved workspace instead of creating a view")
    args = parser.parse_args()
    groups = [
        (
            "01 Quality - independent full-vocabulary evaluation",
            "00_axes/evaluation_outer",
            [
                "01_quality/student_teacher_kl",
                "01_quality/teacher_student_kl",
                "01_quality/student_continuation_nll",
            ],
        ),
        (
            "02 Off-policy - importance weighting",
            "00_axes/inner_update",
            [
                "02_offpolicy/inner_importance_ess_fraction",
                "02_offpolicy/inner_importance_ratio/max",
                "02_offpolicy/inner_sampled_log_ratio/rms",
            ],
        ),
        (
            "03 Optimization - gradients and score coefficients",
            "00_axes/inner_update",
            [
                "03_optimization/inner_grad_norm",
                "03_optimization/inner_score_coefficient/rms",
            ],
        ),
        (
            "04 KL centers - Tail-aware diagnostics",
            "00_axes/inner_update",
            [
                "04_kl_centers/inner_student_center/mean",
                "04_kl_centers/inner_teacher_center/mean",
            ],
        ),
        ("05 Entropy", "00_axes/inner_update", ["05_entropy/inner_student_entropy/mean"]),
        (
            "06 Support mass - anchor head and tail",
            "00_axes/inner_update",
            [
                "06_support_mass/inner_anchor_head_mass/mean",
                "06_support_mass/inner_student_head_mass/mean",
                "06_support_mass/inner_student_tail_mass/mean",
            ],
        ),
        (
            "07 Reward",
            "00_axes/inner_update",
            [
                "07_reward/inner_sampled_reward/mean",
                "07_reward/inner_sampled_reward/std",
            ],
        ),
        (
            "08 Rollout",
            "00_axes/outer_step",
            [
                "08_rollout/response_length/mean",
                "08_rollout/response_length/clip_ratio",
            ],
        ),
        (
            "09 Resources",
            "00_axes/outer_step",
            [
                "09_resources/timing_s/update_actor",
                "09_resources/timing_s/step",
                "09_resources/actor/perf/max_memory_allocated_gb",
            ],
        ),
        (
            "90 Detail - extrema",
            "00_axes/inner_update",
            [
                "90_detail_extrema/inner_score_coefficient/max",
                "90_detail_extrema/inner_sampled_reward/min",
            ],
        ),
    ]
    workspace = (
        Workspace.from_url(args.url)
        if args.url
        else Workspace(
            entity=args.entity,
            project=args.project,
            name="MirrorOPD - grouped diagnostics",
        )
    )
    workspace.sections = [
        Section(
            name=name,
            is_open=i < 4,
            pinned=i < 2,
            panels=[LinePlot(title=key.split("/", 1)[1], x=axis, y=[key]) for key in keys],
        )
        for i, (name, axis, keys) in enumerate(groups)
    ]
    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
