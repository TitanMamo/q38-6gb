# Reproduce

## 1. Prerequisites

- NVIDIA GPU with ~6 GB VRAM (results: GTX 1660 Ti, TU116/sm_75, no tensor cores), 6C+ CPU, 32 GB RAM, ~100 GB free NVMe (one quant is 75–89 GB + build tree).
- Linux, NVIDIA driver **580-series LTS** (the 595 open-module branch corrupts `nvidia_uvm` under sustained UVM load — see SERVING.md §3; any 580.x should do).
- CUDA 12.x toolchain + `cmake`, `curl`, `python3`. All runs here went through a Debian distrobox; native works if the driver userspace matches.
- Box-specific, probably not yours: `pcie_aspm=off` kernel param (this board's PCH root port throws RxErr under DMA load). Check your own AER counters before copying.

## 2. Models (exact sources)

| quant | source repo / subdir | shards | size | role |
|---|---|---|---|---|
| UD-IQ1_M | `unsloth/Qwen3.8-Flash-Next-GGUF`, subdir `UD-IQ1_M` | 3 (`...-00001-of-00003` ≈ 11 KB index + 50.0 GB + 24.5 GB) | ~74.5 GB | speed tier, deep-ctx ladder |
| AD-4.27bpw-Q4_K_M-M64 | `AtomicChat/Qwen3.8-Flash-Next-GGUF` (M64 shard layout: PLE table isolated) | 33 | ~89 GB | quality tier (89.5% top-1) |

Download: single-stream HF CLI stalled repeatedly on these shards — a parallel range-request downloader (12 concurrent ranges/shard, byte-exact assembly, resumable) held ~23 MB/s. Whatever tool you use, **gate on SHA-256 per file** (HF API LFS oids): a corrupt shard presented as a hung server, not an error, and cost a full re-download to diagnose. Never trust size checks alone.

## 3. Build

`ik_llama.cpp`, CUDA build, Release:

```
CMAKE_BUILD_TYPE=Release
CMAKE_CUDA_ARCHITECTURES=75        # match your card
GGML_CUDA_FORCE_MMQ=ON             # +48% PP on TU116, tg parity (measured)
GGML_IQK_FA_ALL_QUANTS=ON          # quantized-KV flash-attn paths
```

Runtime env (both required on this box):

```
export GGML_CUDA_NO_PINNED=1       # else the loader demands ~64 GiB pinned host RAM
export LD_LIBRARY_PATH=<staged 580 libcuda/libnvidia-ml>:<build>/src:<build>/ggml/src:...
```

## 4. System tuning (measured, keep what helps)

- `taskset -c 0-5` on the server (variance reduction), `-t 4 -tb 6` (swept 2–8; 4/6 optimal on 6C6T).
- NVMe readahead 128 → 512 KB; CPU governor `performance`; swappiness left at 150 (zram default: protects file cache).
- 32 GB RAM caps quant choice: resident page-cache + GPU weights need ≳50% of body size or decode falls into a page-fault spiral (coverage model, SERVING.md §1).

## 5. Recipes (complete table)

Common flags for all: `--prefetch-experts --defer-ple --flash-attn on -wgt 1 --ctx-checkpoints-interval 1024 --ctx-checkpoints 8 -t 4 -tb 6 -np 1 --jinja` (reasoning_effort medium), `temp 1.0, top-p 0.95, top-k 20`. Checkpoints are **on, cap 8** (evolved from off — SERVING.md §4; restores 16–95 ms, 0 fatals, with a bit-exactness caveat tracked in #2433).

| recipe | quant | ctx | fit-margin | ub / b | KV cache | measured |
|---|---|---|---|---|---|---|
| iq1m-32k | IQ1_M | 32768 | 768 | 1024 / 2048 | q8/q8 | PP 47.7, tg ~7.8 |
| iq1m-32k-s | IQ1_M | 32768 | 768 | 512 / 2048 | q8/q8 | ladder variant |
| iq1m-48k | IQ1_M | 49152 | 1024 | 512 / 2048 | Kq6/Vq6 + ictk q8 | PP 30.1, gen 4.8 |
| iq1m-64k | IQ1_M | 65536 | 512 | 384 / 2048 | Kq6/Vq4 + vhad + ictk q8 | PP 25.3, gen 5.4 |
| iq1m-96k | IQ1_M | 98304 | 512 | 512 / 2048 | same | PP 29.0, gen 4.4 |
| iq1m-128k | IQ1_M | 131072 | 512 | 384 / 2048 | q4+had both + ictk q8 | PP 23.9, gen 4.0 |
| ad16k | AD | 16384 | 384 | 1024 / 2048 | q8/q8 | PP 40.6, tg 7.3 |
| ad16k-800 | AD | 16384 | 384 | 800 / 2048 | q8/q8 | PP ~31, proven 11543 |
| ad16k-s | AD | 16384 | 384 | 512 / 2048 | q8/q8 | PP ~25 |
| ad32k-q6 | AD | 32768 | 384 | 512 / 2048 | Kq6/Vq6 + ictk q8 | PP ~25, ceil ~19k |
| ad32k-q6u | AD | 32768 | 384 | 1024 / 2048 | same | NO-FIT at load |
| ad32k-q6-384 | AD | 32768 | 384 | 384 / 2048 | same | PP ~17, 26919 ok |
| ad32k-q4 | AD | 32768 | 384 | 512 / 2048 | q4+had both + ictk q8 | PP ~25, ceil ~23k |
| ad32k-q4-384 | AD | 32768 | 384 | 384 / 2048 | same | PP ~17, **30763 proven** |
| ad32k-q4-432 | AD | 32768 | 384 | 432 / 1728 | same | PP ~22, **production** (ckpt 8/1024, spill binary) |
| ad48k | AD | 49152 | 512 | 128 / 1024 | q4+had both + ictk q8 | PP 6.3, gen 2.9 |

Margins are load-bearing, not cosmetic: 32k/m384 → 768 (16k-prefill OOM), 48k/m512 → 1024 (24k-extension OOMs). On AD the margin cannot go above 512 (dense stack too fat — 40 MB slack).

## 6. Running (kill discipline or you will wedge the box)

```
source scripts/server-guard.sh
q38_stop_server || exit 1     # SIGTERM + 60 s wait for CUDA teardown
q38_require_free || exit 1    # refuse instead of duplicating
taskset -c 0-5 llama-server <flags> --host 127.0.0.1 --port 8013
curl -s localhost:8013/health # {"status":"ok",...}
tail -f logs/q38-server.log   # fatals MUST land in a file
```

Rules: one server at a time (two split 5.7 GB VRAM and share the port into a hang); always match `[l]lama-server` (bare `pkill -f llama-server` matches your own shell); **never `-9` a live CUDA server** (corrupts `nvidia_uvm`, 5.5 GB leak needing rmmod/reboot) — `-9` only via `Q38_FORCE=1` after TERM fails. Post-patch binary dies clean on fatals (backtrace in log, GPU self-frees); read the `ggml_abort` line before relaunching.

## 7. Ladder + verify

```bash
./scripts/bench-ladder.sh <label>                  # stops at first OOM, logs/ladder-<label>.log
RESUME_FROM=8 ./scripts/bench-ladder.sh <label>    # resume after restart, history kept
```

Ground truth is server-reported `prompt_n` per rung; `max_tokens=16` isolates prefill; rung targets default `2048 2048 4096×8`. 2k-prompt spot checks are **not** a substitute — they never OOM and miss the transient-pool wall entirely. Quality probe per recipe: small math + reasoning check (both decompositions of 17×23, explanation, colors) — catches garbage, not regressions.

## 8. Client setup (production)

Consumer configured to context 32768 with compaction (reserve 8192 → fires at 24576; post-compact ~22k; next trigger ~24.5k prefill — all inside ladder-proven ground), output cap at half the window, 30-min timeouts. Keep the tool/skill set stable per session — a mid-chat skill load re-renders the prompt prefix and forces a total-cache-miss re-prefill.

## 9. Thinking toggle (pi + ik server, verified)

Alibaba's levels are `xhigh` (default, what their benchmarks use), `medium`, `low`. The server fixes one via `--chat-template-kwargs` but honors **per-request** `chat_template_kwargs` (server-common.cpp merges them over the CLI default) — so pi can switch effort mid-session with no restart:

- pi model entry (`~/.pi/agent/models.json`): `"reasoning": true`, `thinkingLevelMap` {off→off, minimal/low→low, medium→medium, high/xhigh/max→xhigh}, `compat: {thinkingFormat: "chat-template", chatTemplateKwargs: {enable_thinking: {$var: "thinking.enabled"}, preserve_thinking: true, reasoning_effort: {}}}`.
- In-session: `/thinking low|medium|max` (default medium = today's behavior, zero change until touched).
- Verified live: same puzzle, low → 3.3k thinking chars, xhigh → 5.5k, both correct.
- With a near-full window prefer low/medium — xhigh's extra thinking tokens come out of remaining context.
