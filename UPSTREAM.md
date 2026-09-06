# Upstream record

All as TitanMamo (code) / TitanMamo10 (HF). Hardware and recipes above are the evidence base.

## PR #2416 — ggml backtrace fork fix (OPEN, awaiting review)

`ikawrakow/ik_llama.cpp` PR from `origin/main`, single file `ggml.c` (+74/−5): skip the `fork()` when no debugger is installed, timed waitpid + kill, `_exit` in the child. Converts every CUDA fatal from a permanent wedge (port+VRAM held, HTTP dead) into a clean death with backtrace. Verified live 7x, zero orphans. Full mechanism in [SERVING.md §5](SERVING.md#5-the-abort-wedge-fixed-upstream-pr-2416).

## Issue #2414 — IQ3_KT collapse (resolved as non-bug)

Filed with diagnosis (shipped L=16/N=8 codebook ~22% error), repro, and workarounds. Maintainer showed 1.5B holds; I ran his implicit test and confirmed (F16 3.15 / iq3_k 3.33 / iq3_kt 3.49, no collapse), conceded with numbers, clarified the PR-#113 spec question (never merged — 16/8 was the deliberate improvement), left measurements on record for him to close. Full story in [QAT.md §4](QAT.md#4-side-result-iq3_kt-upstream-2414).

## Issue #2415 — `--custom-q` silently ignored (declined by design, warning offered)

`llama_tensor_get_type` is only consulted for quantized default ftypes, so custom rules with an unquantized default do nothing with no message (verified in clean upstream `llama-quantize.cpp`; also noted case-sensitive type-name parsing). Maintainer declined the premise as out of scope; offered a warning-only patch (no behavior change), awaiting answer. Found via the QAT mixed-grid export bug ([QAT.md §3](QAT.md#3-failed-runs-kept-for-the-lessons)).

## Discussion #2417 — Q3 types RFC (OPEN, awaiting interest decision)

Asked whether the three Q1_G128-based types (Q3_0_G128 3.125 / Q3_1_G128 3.25 / Q3_1_G64 3.50 bpw) are wanted for mainline, with PTQ tables, kernel throughput, token-identical generation, and the QAT angle (training grid = deployment grid). Offered rebase, debug strip, per-type PRs, Trellis untouched, regression checks. Cleanup gated on the answer. Discussion thread also carries the PoC-rationale answer with artifact links.

## HF discussions (data posts)

- unsloth/Qwen3.8-Flash-Next-GGUF #66 (UD-IQ1_M: 32k numbers, 128k ladder, hiccups + fixes).
- AtomicChat/Qwen3.8-Flash-Next-GGUF #20 (AD-4.27: 16k/48k numbers, 32k ladder to 30.8k, dense-headroom analysis).
- Artifacts repo: https://huggingface.co/TitanMamo10/qwen25-qat-q3-poc
