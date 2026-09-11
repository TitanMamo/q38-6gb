# Experiment: MoE expert cache on 6 GB VRAM (Codacus `perf` branch)

## Hypothesis

Our decode bottleneck is expert traffic (RAM → GPU per token). Pinning
hot-routed experts GPU-resident should lift tg the way it did on their
measured setups (+21% Qwen3.6, +47% Flash-Next, bit-identical output).
Expected Ladder: IQ1_M + 20–35 slots beats IQ1_M baseline tg (~7.8)
without losing the 32k window.

## Why IQ1_M, not AD (decided 2026-09-11)

Expert slots need VRAM headroom past dense+KV+compute, plus ~900 MB
transient headroom (their rule, matches our own OOM forensics):
- AD-4.27: dense 4.6 + KV 0.4 + compute 0.5 ≈ 5.5/5.7 GB → ~4 slots.
  Possible per their Nemotron result, but headroom-free = OOM roulette.
- IQ1_M: dense ~2.0 GB → ~2.5–3 GB free → 20–35 slots with headroom.
Different codebase note: their branch is mainline-based (`perf`,
qwen4exp wired Sept 3); our stack is ik. Flags don't transfer 1:1;
this experiment runs their stack standalone first, port decision after.

## Method (section discipline)

1. Build `perf` tip (CUDA sm_75) → verify `--moe-cache-slots` in help.
2. Capture routing profile (`llama-moe-trace`, code+chat prompts merged).
3. Serve IQ1_M + slots ladder (12 → 24 → 36 or fit wall), 16k ctx:
   record tg/PP, VRAM peak, `init_moe_expert_cache` line.
4. Baseline first: same model/prompts on ik stack (have: tg 7.82,
   PP 47.67 @32k; re-bench at 16k for apples-to-apples).
5. Quality probe per rung (17×23 both decompositions + reasoning).
   Bit-identical claim is theirs; our probes verify independently.
6. Verdict: adopt (switch stack or port), reject (record why), or
   scope (e.g. helps PP not tg).

## Verdict: REJECTED for 6 GB (2026-09-11)

| Stack (IQ1_M, c8192) | PP t/s | tg t/s | VRAM |
| --- | --- | --- | --- |
| mainline `perf` baseline | 5.88 | 6.79 | 4358 MB |
| mainline `perf` + cache 8 slots | 7.10 (+21%) | 5.93 (−13%) | 5456 MB |
| ik production (c32k, reference) | 47.67 | 7.82 | ~2 GB dense |

Cache engaged (+1098 MB pack, profile valid, 48/48 layers matched).
Prefill win matches their claims; decode loses: 8/512 experts/layer =
1.6% coverage, far below their ~25% breakeven — dual-path overhead wins.
VRAM caps us at ~8 slots, so the breakeven is unreachable on 6 GB.
Stack gap is decisive anyway (ik PP 47 vs 7).

Genuine bycatch, kept: their residency gate (`buft_is_host`) silently
no-ops for mmap'd models (all big models on small RAM). One-line fix —
Upstream note: thecodacus/llama.cpp has issues disabled; patch kept
locally (this file) — offered via discussion/video comment if suitable.

## Status log

- 2026-09-11: branch cloned at 27c54b4b, CUDA build fired; IQ1_M
  staging NVMe←archive (70 GB, SHA-verified set). AD-4.27 archived
  (copy verified by size/count). Production AD server untouched
  (mmap holds deleted inode safely).
