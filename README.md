# One-Pass Decisions: Serving Typed Decisions with Exact State Reuse and Certified Escalation

A serving engine for typed-decision workloads: many small questions over one shared state, packed into a single encoder pass with exact reuse of the state encoding, and gated by an escalation threshold calibrated on the scores the served engine actually produces.

- `onepass/` — package.
- `scripts/` — one runner per experiment, writing to `results/wNN_<exp>/`.
- `results/` — committed summaries (JSON) and per-row outputs (JSONL).
- `RUNBOOK.md` — environment setup and reproduction commands.

Hardware target: one Apple M3 Ultra (96 GB unified memory, 28 cores) for development, fine-tunes and ARM measurements; one rented bare-metal x86-64 Linux host for the second ISA; up to about 50 rented H100 hours for the GPU column.
