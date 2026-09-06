# Qwen3.8-Flash-Next on 6 GB VRAM — serving ceilings and stability fixes

Serving a 176B-class MoE (`qwen4exp`: 125B MoE + 51B PLE + 4B MTP) on a GTX 1660 Ti 6 GB for real agentic-coding work: measured ceilings and the stability bugs found along the way (one filed/fixed upstream).

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

**Upstream:** 1 PR (crash-wedge fix, verified live 7x) + 2 HF data posts — [record](UPSTREAM.md).

Separate project: [qwen25-q3-qat](https://github.com/TitanMamo/qwen25-q3-qat) (3-bit quantization + QAT on Qwen2.5 0.5B/1.5B). Different model, different question — kept apart deliberately.

## Contents

- [SERVING.md](SERVING.md) — quant tiers, optimization journey, freeze saga, OOM series, full ladder with per-rung data, production setup.
- [POST-why-no-30-32k.md](POST-why-no-30-32k.md) — why the (30 t/s, 32k) point does not exist.
- [UPSTREAM.md](UPSTREAM.md) — issues / PR / discussions with outcomes.
- [REPRODUCE.md](REPRODUCE.md) — exact build flags, server flags, ladder + PPL commands.
- [scripts/](scripts/) — `bench-ladder.sh` (prefill ladder harness), `server-guard.sh` (singleton + kill-discipline guard).

## Honest gaps

- ub432 (PP ~22) is a 2-rung probe, not a ladder — no ceiling measured.
- ad16k-s died on rung 6 at the 16k ctx wall rather than clean-refusing — reported as-is.

## How this was built (human + AI)

Ideas, direction, and verification standards are mine; implementation is AI-assisted. I set the questions (ladder the ceilings, pre-register the bars, ablate the failures), the AI writes the code, harnesses, and drafts — I review, catch mistakes, and redirect. The ladder method that found every ceiling, the export-bug hunt, the batch-scaling postmortem, and the #2414 concession all came out of that loop. Nothing here was accepted on the AI's say-so: every number is a measured artifact, every "rejected" verdict is documented, and the honest-gaps section above is deliberate — falsifiable claims over optimism.
