# CLAUDE.md — Rosetta Transformer for Antiquity

## Research Problem

Ancient and low-resource languages (Ancient Greek, Latin, Classical Arabic) are underserved by modern NLP — training data is scarce, annotation is expert-only, and standard tokenizers break on ancient morphology. **Rosetta** adapts modern Transformer architectures for these languages: translation, NER, POS tagging, morphological analysis, and relation extraction.

## Supported Languages

| Code | Language |
|---|---|
| `grc` | Ancient Greek |
| `la` | Latin |
| `ar` | Classical Arabic |

## Architecture

```
rosetta/
  models/
    pretraining.py    Language model pretraining on ancient corpora
    seq2seq.py        Neural translation (ancient ↔ modern languages)
    token_tasks.py    NER, POS tagging, morphological analysis, relation extraction
  data/               Corpus loaders, tokenization, augmentation
  api/                FastAPI REST API for interactive use
  evaluation/
    metrics.py        BLEU, character F1, bootstrap confidence intervals
    validators.py     Annotation validators
  utils/
  cli.py              CLI entrypoint
```

## Setup & Commands

```bash
python3.9 -m venv .venv && source .venv/bin/activate   # Python 3.9+ (3.11 preferred)
pip install -r requirements.in                          # production deps
pip install -r requirements-dev.in                     # dev deps

make install-dev     # installs both + dev tools
make format          # ruff format
make lint            # ruff check + mypy
make test            # pytest tests/
make ci              # format + lint + test (full CI sequence)

# CLI
rosetta --help
rosetta train --config configs/<task>.yaml
rosetta evaluate --config configs/<task>.yaml --checkpoint outputs/best.ckpt
rosetta serve                                           # start FastAPI
```

## Configuration

YAML-based configs in `configs/` — one per task (translation, pretraining, token tasks). Uses environment variable interpolation for data paths and model checkpoints.

## Docker

CPU and GPU variants in `docker/`. Use Docker for experiments requiring GPU:

```bash
docker compose -f docker/docker-compose.gpu.yml up
```

## Evaluation

Metrics depend on task:
- **Translation**: BLEU, chrF, character F1
- **Token tasks** (NER, POS, morph): span-level F1, accuracy
- **All tasks**: bootstrap confidence intervals (95% CI) via `evaluation/metrics.py`

Statistical significance is required for paper-worthy claims — use bootstrap CI, not just point estimates.

## Research Notes

Ancient morphology is the main challenge — languages like Ancient Greek and Latin are highly inflected, and modern BPE tokenizers over-segment uncommon forms. When pretraining or fine-tuning, watch subword vocabulary coverage on ancient text specifically. The `data/` module handles corpus-specific preprocessing; do not bypass it even for quick experiments. Expert-in-the-loop annotation (human feedback integration) is planned — `examples/` shows how to wire it in.
