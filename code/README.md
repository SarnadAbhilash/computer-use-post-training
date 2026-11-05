# Code

## harness/

Runs the model against a live desktop and scores each attempt with automatic checkers.

- `agent.py` / `heldout.py` — the computer-use loop (screenshot → model → click/type → repeat)
- `env.py` — display and sandbox helpers
- `checkers.py` / `checkers4.py` — check real files on disk; do not trust the model's "done" claim
- `eval4.py` — eval runner for the two locked test suites

Scripts that talk to a model server read API keys from environment variables (for example `VLLM_API_KEY`). Nothing in this repo stores those keys.

## train/

Training scripts used for the final stage:

- `train_lora.py` — LoRA supervised fine-tuning (and the same loss used for GRPO rows with signed weights)
- `grpo_select.py` — turn rollout results into GRPO training rows
- `merge_adapter.py` — merge a LoRA adapter into full weights
- `requirements-train.txt` / `requirements-serve.txt` — package lists for training and serving

Paths inside these scripts are local defaults from the original training machine. Point them at your own data and checkpoints before running.
