# Qwen3.8-Flash-Next on 6 GB VRAM — serving ceilings, stability fixes, 3-bit QAT

Serving a 176B-class MoE (`qwen4exp`: 125B MoE + 51B PLE + 4B MTP) on a GTX 1660 Ti 6 GB for real agentic-coding work: measured ceilings, the stability bugs found along the way (two filed/fixed upstream), and a parallel 3-bit quantization project whose format work is done and whose training is honestly short of its bar.

- Hardware: GTX 1660 Ti 6 GB (TU116, sm_75, no tensor cores) · i5-9400F 6C6T AVX2 · 32 GB DDR4 · NVMe · Void Linux (runit); inference in a `cuda-box` distrobox (Debian, CUDA 12.4, driver 580.178.04 LTS).
- Server: `ik_llama.cpp` `llama-server`, CPU/RAM expert offload, build `+GGML_CUDA_FORCE_MMQ=ON` (+48% PP on TU116, tg parity), `GGML_CUDA_NO_PINNED=1` (without it the loader demands 64 GiB of pinned host memory on a 32 GB box).

## Key results

**Serving (finished, reproduced):**

| setup | ctx proven | PP | gen | note |
|---|---|---|---|---|
| UD-IQ1_M, q8 KV, ub1024 | 32k daily driver | 47.7 t/s | ~7.8 | quality probe PASS |
| UD-IQ1_M ladder (KV shedding) | **128k** (192k fails fit) | 23.9 | 4.0 | probes PASS at every rung |
| AD-4.27, q8 KV, ub1024 | 16k quality tier | 40.6 | 7.3 | 89.5% top-1 |
| AD-4.27, q4 KV, ub384 | **30.8k** | ~17 | — | max-context winner |

The headline negative result: **no config does PP-30 and 32k simultaneously on this card** — [the ladder and the mechanism](POST-why-no-30-32k.md).

**QAT (in progress, bar unmet):** three custom 3-bit GGML types (CPU+CUDA kernels, token-identical generation) that beat IQ3_K at fewer bits *before training*; self-distillation training with lossless grid transfer (dg +0.0002) but weights at ppl 21.9 vs the 16.5 bar. Full interim record in [QAT.md](QAT.md).

**Upstream:** 2 issues, 1 PR (crash-wedge fix, verified live 7x), 1 RFC discussion, 2 HF data posts — [record](UPSTREAM.md).

## Contents

- [SERVING.md](SERVING.md) — quant tiers, optimization journey, freeze saga, OOM series, full ladder with per-rung data, production setup.
- [QAT.md](QAT.md) — 3-bit types, grid sweep, training series (1028 → 21.9), lessons from failed runs.
- [POST-why-no-30-32k.md](POST-why-no-30-32k.md) — why the (30 t/s, 32k) point does not exist.
- [UPSTREAM.md](UPSTREAM.md) — issues / PR / discussions with outcomes.
- [REPRODUCE.md](REPRODUCE.md) — exact build flags, server flags, ladder + PPL commands.
- [scripts/](scripts/) — `bench-ladder.sh` (prefill ladder harness), `server-guard.sh` (singleton + kill-discipline guard).

## Honest gaps

- QAT training has not reached its pre-registered bar (21.9 vs ≤16.5); the 0.5B floor and the scale-up math are documented, not hand-waved.
- ub432 (PP ~22) is a 2-rung probe, not a ladder — no ceiling measured.
- ad16k-s died on rung 6 at the 16k ctx wall rather than clean-refusing — reported as-is.
