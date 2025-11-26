# Official Repository for Output Supervision Can Obfuscate the Chain of Thought

This repository supports:
1. Post-training language models to demonstrate feedback spillover
2. Training language models with the mitigations introduced in our paper

## 1. Root-Level Layout

| Path                       | Purpose |
| -------------------------- | ------- |
| `src/`                     | Main python package with all library code.  Import root is `simple_obfuscation`. |
| `scripts/`                 | Human / CI entry-point scripts (e.g. `train.py`). |
| `tests/`                   | Unit tests covering critical utilities and pipelines. |
| `rollouts/`                | (git-ignored)  Serialized rollout data produced during training. |
| `wandb/`                   | (git-ignored)  Weights & Biases logs. |
| `requirements.txt`         | Locked dependency list used by `setup.sh`. |
| `README.md`                | You are here. |

---
## 2. Source Package – `src/`

```
src/
├── __init__.py              # Exposes top-level public API
├── generation/              # Prompt building and logit manipulation
├── reward/                  # Modular reward functions & registry
├── trainer/                 # REINFORCE trainer and rollout buffer
├── config/                  # Hydra-style configuration objects / YAML
└── utils/                   # Small stateless helpers (token ops, etc.)
```

### 2.1 `generation/`
File | Responsibility
---- | --------------
`prompt_builder.py` | Converts raw task inputs (e.g. MMLU questions) into prompts. Also houses configurable templates.
`logit_processors.py` | Custom `transformers.LogitsProcessor` subclasses (e.g. masking, biasing) applied during generation.

### 2.2 `reward/`
File | Responsibility
---- | --------------
`base.py` | Abstract `Reward` base class defining the scoring interface.
`registry.py` | Dynamic factory mapping `reward_name ➜ class`.
`boxed_answer_reward.py` | Rewards exact matches inside a bounding box.
`regex_reward.py` | Regex-based matcher for flexible string scoring.
`judge_reward.py` | Delegates scoring to an external LLM “judge”.

### 2.3 `trainer/`
File | Responsibility
---- | --------------
`reinforce_trainer.py` | Core REINFORCE training loop.
`rollout_store.py` | Simple in-memory buffer → persisted into `rollouts/`.

### 2.4 `config/`
Component | Description
--------- | -----------
`__init__.py` | Pydantic/Hydra config objects with sensible defaults.
`[task].yaml` | YAML override for [task].

### 2.5 `utils/`
File | Responsibility
---- | --------------
`token_utils.py` | Miscellaneous helpers for token counting, truncation, etc.

---
## 3. Training Entrypoint

`scripts/train.py`
• Wraps the `reinforce_trainer.ReinforceTrainer` with CLI argument parsing.
• Loads config via `src.config`, initializes reward + generation pipeline, then launches training.

---
## 4. Test Suite

`tests/` contains pytest-based validation for registry loading, prompt generation, masking logic, and utility functions.
