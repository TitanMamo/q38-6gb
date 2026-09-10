# Upstream record (serving)

## PR #2416 — ggml backtrace fork fix (MERGED 2026-09-10)

`ikawrakow/ik_llama.cpp` PR, single file `ggml.c`: the debugger invocation is gone entirely — straight to `backtrace_symbols_fd`, no fork/exec/wait anywhere on the fatal path. Converts every CUDA fatal from a permanent wedge (port+VRAM held, HTTP dead) into a clean death with backtrace. v1 verified live 8x (clean deaths, zero orphans); maintainer asked for the simpler shape (delete instead of detect-and-skip), v2 merged as suggested. Full mechanism in [SERVING.md §5](SERVING.md#5-the-abort-wedge-fixed-upstream-pr-2416).

## HF discussions (data posts)

- unsloth/Qwen3.8-Flash-Next-GGUF #66 (UD-IQ1_M: 32k numbers, 128k ladder, hiccups + fixes).
- AtomicChat/Qwen3.8-Flash-Next-GGUF #20 (AD-4.27: 16k/48k numbers, 32k ladder to 30.8k, dense-headroom analysis).

Quantization upstream work (issues #2414/#2415, discussion #2417) lives with the [qwen25-q3-qat](https://github.com/TitanMamo/qwen25-q3-qat) project.
