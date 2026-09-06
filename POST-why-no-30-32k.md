# Why (30 t/s, 32k context) does not exist on a 6 GB card

*Measured on GTX 1660 Ti 6 GB / i5-9400F / 32 GB RAM, Qwen3.8-Flash-Next (qwen4exp MoE) via ik_llama.cpp, driver 580. Every number below is a server-reported measurement, not an estimate.*

The goal was a daily driver for agentic coding: 32k context window at ~30 tokens/s prompt processing. It does not exist on this card. Not "we haven't found it" — the ladder shows the point is off the curve, and the mechanism explains why.

## The ladder

Standard benchmarks use ~2k prompts. They never OOM, so they never see the wall. I ran accumulating-history ladders instead — pi-style, the client resends full history every turn, server `prompt_n` as ground truth, decode held at 16 tokens so the measurement is pure prefill:

| ubatch | KV | PP (flat) | ceiling |
|---|---|---|---|
| 1024 | q8 | ~41 t/s | ~8–11k, OOM |
| 512 | q8 | ~25 | ≥16k (died rung 6 at the 16k ctx wall, no clean refusal) |
| 512 | q6 | ~25 | ~15–19k, OOM |
| 512 | q4 | ~25 | ~19–23k, OOM |
| 384 | q6 | ~17 | ~27–31k, OOM |
| 384 | q4 | ~17 | **30.8k proven** (died rung 10) |
| 432 | q4 | ~22 | 2-rung probe, no ceiling measured |

Read it as two independent axes. **PP is set by ubatch alone** — KV quantization does not move it a single t/s (q8/q6/q4 all sit at ~25 under ub512). **Ceiling marches ~+4k per KV step.** The target needs ub1024-class speed (only ub≥~700 reaches 30) with ub384-class memory (only ub≤~450 survives past 25k). Those two sets do not intersect. Closest point overall: ub384 + q4 KV at (17 t/s, 30.8k).

## Why the wall is where it is

The killer is not steady-state KV — it is the transient. This model's hybrid SWA path discards cache on mismatch and re-prefills the whole prompt while old KV is still resident: peak = old KV + full re-prefill scratch, and the scratch scales with KV length, not ubatch. A 16k re-prefill needs ~0.5–1 GB of pool growth on top of a card idling at 5.5/5.7 GB. Margins (384→768→1024 MB) bought one rung each and then stopped helping; the dense stack (~4.6 GB Q8 on the quality quant) cannot trade experts for headroom the way the 2 GB-dense quant can.

Side finding with teeth: q6 KV costs ~40% decode on TU116 (4.1 vs 7.3 t/s, prefill unaffected) — dequant ALU per KV byte on every step. Quantize KV only when VRAM forces it.

## What I actually run

The IQ1_M quant (slim dense stack) does what AD cannot: 32k at PP 47.7, and a validated ladder out to 128k (PP 23.9, quality probes pass) on the same 6 GB. The production setup is AD-4.27/ub384/q4 (17 t/s, 30.8k proven) with client-side compaction firing at 24.5k — keeping every prefill inside ladder-proven ground instead of pushing the wall.

## Reproduce it

Launch script, ladder harness, and full tables: link. One command per rung, OOMs die clean (backtrace patch, PR #2416) with GPU self-freeing. If your card has more VRAM, the same ladder tells you exactly which (PP, context) points exist on it — the method transfers, the numbers won't.
