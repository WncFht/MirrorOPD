# MirrorOPD small language-model experiment

This directory contains a real-model OPD experiment. The teacher is
`EleutherAI/pythia-160m-deduped` and the student is
`EleutherAI/pythia-70m-deduped`. Both use the same GPT-NeoX tokenizer family.

The experiment keeps one frozen student anchor per outer step. It samples
continuations from the anchor, computes teacher and anchor token log-probs,
and runs a finite number of inner updates before hard-syncing the new student.
The four inner objectives are `pmd_part`, `pmd_mean`, `bpo`, and `mdpo`.

The first version is deliberately a small, auditable token-level experiment:

- prompts come from WikiText-2 validation text;
- each outer step uses a fixed prompt batch and frozen rollouts;
- the target is evaluated on sampled continuation tokens;
- metrics include student NLL, teacher NLL, KL(student || teacher) on sampled tokens,
  OPD reward `log teacher - log anchor`, and inner residual.

Run on archbox:

```bash
cd ~/src/mirroropd-llm
uv run --with torch --with transformers --with datasets --with accelerate \
  python train_opd.py --method pmd_part --outer-steps 8 --inner-steps 50
```

The script writes checkpoints, JSONL metrics, and plots below `runs/`, never `/tmp`.
