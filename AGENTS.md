# Rosetta Transformer Agent Notes

Rosetta-Transformer-for-Antiquity is a research portfolio repo for ancient and
low-resource language NLP: translation, tagging, corpus search, annotation, and
expert-in-the-loop feedback.

## Architecture

- `rosetta/api/`: FastAPI serving, inference, feedback, and search.
- `rosetta/models/`: base model interfaces, pretraining, seq2seq, token tasks.
- `rosetta/data/`: corpus loading, cleaning, alignment, annotation schemas.
- `rosetta/evaluation/`: metrics and output validators.
- `rosetta/utils/`: config, logging, reproducibility.
- `rosetta/cli.py`: training/evaluation/data-processing CLI.

The project bridges ML engineering and classical scholarship. Keep outputs
traceable and reproducible; do not treat ancient-language examples as generic
NLP toy text.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.in
pip install -r requirements-dev.in
python -c "from rosetta.utils import Config, setup_logging, set_seed; print('ok')"
python examples/examples_utils_usage.py
uvicorn rosetta.api.main:app --reload --host 0.0.0.0 --port 8000
```

Run the repo's configured ruff/mypy/pytest commands from `pyproject.toml` before
shipping changes.

## Editing Rules

- Keep configuration YAML-driven and environment-variable friendly.
- Use Pydantic schemas at API/data boundaries.
- Do not commit corpora, checkpoints, generated annotations, or local
  experiment outputs.
- Evaluation should include statistical confidence where the existing metrics
  framework supports it.
