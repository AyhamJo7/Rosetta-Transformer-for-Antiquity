# Security Policy

## Supported versions

Security fixes are applied to the latest `main` branch. There is no long-term-support branch.

## Reporting a vulnerability

If you discover a security vulnerability, please report it **privately** — do not open a public issue.

- Email: **ayhamjumran7@gmail.com** with the subject line `SECURITY: Rosetta-Transformer`.
- Include a description, reproduction steps, and the potential impact.

You can expect an acknowledgement within a few business days. This is a research/portfolio project maintained on a best-effort basis; there is no formal disclosure SLA, but credible reports are investigated and fixed as quickly as is practical.

## Scope

Rosetta Transformer loads models, tokenizers, and multilingual corpora that may download remote artifacts or execute code. Only load checkpoints and datasets from sources you trust. Reports about untrusted-model or untrusted-corpus execution paths are in scope.
