# Upstream record (serving)

## PR #2416 — ggml backtrace fork fix (MERGED 2026-09-10)

`ikawrakow/ik_llama.cpp` PR, single file `ggml.c`: the debugger invocation is gone entirely — straight to `backtrace_symbols_fd`, no fork/exec/wait anywhere on the fatal path. Converts every CUDA fatal from a permanent wedge (port+VRAM held, HTTP dead) into a clean death with backtrace. v1 verified live 8x (clean deaths, zero orphans); maintainer asked for the simpler shape (delete instead of detect-and-skip), v2 merged as suggested. Full mechanism in [SERVING.md §5](SERVING.md#5-the-abort-wedge-fixed-upstream-pr-2416).

## HF discussions (data posts)

- unsloth/Qwen3.8-Flash-Next-GGUF #66 (UD-IQ1_M: 32k numbers, 128k ladder, hiccups + fixes).
- AtomicChat/Qwen3.8-Flash-Next-GGUF #20 (AD-4.27: 16k/48k numbers, 32k ladder to 30.8k, dense-headroom analysis).

## Issue #2433 — qwen4exp cache-miss full-reprocessing (OPEN, direction asked)

General-first framing: hybrid non-partially-removable state (SSM + indexer + compressed KV, *not* SWA — verified zero SWA keys in 70-key GGUF metadata) forces full re-prefill on every miss; checkpoint store always empty under ckpt-0 (erase crash survival). Prior art surveyed (#22940, missing `--swa-full`, SGLang non-transfer); checkpoint-free vs checkpoint-restore tradeoff laid out with the 3.6 GB budget figure. Explicitly asked direction before implementing. Full research in `/home/titan/swa-fix-HANDOFF.md` (local).

**Update (Sept 2026):** production moved to ckpt 8/1024 (ladder-proven 30.8k, restores 16–95 ms, 0 fatals); spill-to-NVMe branch live-tested (70 ms from-disk restores). But logprob parity falsifies exactness: cold full-prefill 48/48 @ 0.0000, restore diverges 5/48 @ 0.38 from token 1 — base state corrupt on restore, prime suspect recurrent-state tail mapping. Posted to the thread with spill data; awaiting maintainer direction. No exact-parity claim until the tail is fixed.

Quantization upstream work (issues #2414/#2415, discussion #2417) lives with the [qwen25-q3-qat](https://github.com/TitanMamo/qwen25-q3-qat) project.
