# Exact Prefix Caching for Bidirectional Encoders: Serving Many Questions over One Document

A block attention mask makes a bidirectional encoder's document keys and values independent of the questions asked of it, so one encode serves every closed-set question about a document exactly and can be cached. This repository holds the mask, the engine that runs it as FlashAttention with no mask, the measurements of its accuracy price and speed, and an escalation threshold set on the scores the served engine actually produces.

- `onepass/` — package.
- `scripts/` — one runner per experiment, writing to `results/wNN_<exp>/`.
- `results/` — committed summaries (JSON) and per-row outputs (JSONL).
- `RUNBOOK.md` — environment setup and reproduction commands.

Hardware target: one Apple M3 Ultra (96 GB unified memory, 28 cores) for development, fine-tunes and ARM measurements; one rented bare-metal x86-64 Linux host for the second ISA; up to about 50 rented H100 hours for the GPU column.
