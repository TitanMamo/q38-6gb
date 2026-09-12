# Serving: from first load to 128k on 6 GB

Method throughout: same box, same prompts, server-reported timings. Prefill (PP) measured with `max_tokens=16`; decode (tg/gen) with short prompts or post-prefill generation. Quality probes are small math+reasoning checks (e.g. 17×23 both decompositions), not benchmarks — they catch garbage, not regressions.

## 1. Quant tiers: speed vs quality is binary, not a spectrum

Three quants measured head-to-head (16k ctx, same harness):

| metric | UD-IQ1_S (72.5 GB) | AD-3.84bpw (79.1 GB) | AD-4.27bpw (88 GB) |
|---|---|---|---|
| tg short-prompt | **7.57** | 5.53 | 5.58 |
| PP 2k (ub) | **45.4** (1024) | 28.6 (512) | 22.7 → **40.6** (1024 after `-wgt 1` fix, §2) |
| gen after 2k prefill | **4.5–5.2** | 3.74 | 2.88 |
| top-1 / KLD | 77.3% / ~0.4 | 82.7% / 0.23 | **89.5% / 0.084** |

Findings:

- **tg does not scale with expert-body bytes at this tier.** AD-3.84 (IQ1_M-class body) and AD-4.27 (heavier mix) both land at ~5.5 t/s — the bottleneck is the shared constant: the AtomicChat Q8_0 dense stack (~4.6 GB on GPU) + MoE sync + per-token fixed costs. The IQ1_S advantage (~2 t/s) = smaller GPU dense (~3.4 GB) + 2–3 expert layers fitted on GPU + ub1024.
- The coverage model held: resident page-cache + GPU weights vs body size predicts tg within ~1 t/s (60% coverage → 5.6 measured). Below ~50% coverage the box falls into a page-fault spiral — a hard floor for quant choice on 32 GB RAM.
- **No sweet spot in the middle**: AD-3.84 is dominated (same speed as AD-4.27, worse quality). Real choice is binary — IQ1_M for speed, AD-4.27 for quality. AD-3.84 was deleted.
- Audit note: AD-3.84's "IQ4_XS" name is cosmetic — its body is ffn_down MXFP4 (22.8%) + gate/up IQ1_M + IQ2_S edges + Q8_0 dense + Q5_1 table.

## 2. Optimization journey (UD-IQ1_S @ 32k): tg 3.84 → 7.57

| step | tg | note |
|---|---|---|
| first working config (margin 1024, -t 6) | 3.84 | — |
| `--fit-margin 384` (more expert layers on GPU) | 3.98 | — |
| `-no-ooae` | 3.51 | hurts, reverted |
| `--prefetch-experts` | 4.69 | background page-cache populate |
| `-t 5` / `-t 4` / `-t 3` / `-t 2` | 6.48 / **6.96–7.32** / 7.19 / 6.16 | -t 4 optimal on 6C6T |
| final: margin 384 + prefetch + `-t 4 -tb 5` + ub1024 | **7.57 best, 7.0–7.3 typical** | — |

PP is a ubatch sweep: ub256 → 22.6, **ub1024 → 45.4**, ub1536 → server spins (abandoned), ub2048 → OOM at context creation. Later: `-wgt 1` rescued ub1024 on the Q8_0 dense stack (AD-4.27 PP 22.7 → 39.3, +73%), and `-tb 6` + auto-prefetch confirmed optimal after rejecting tb8 / prefetch-threads 8 / ub1536 (all measured, all rejected). MTP speculative decoding unavailable (no MTP tail in the quant). Final defaults: `CTX=32768 MARGIN=384 UBU=1024 T=4 TB=6 wgt1`, q8 KV, flash-attn, checkpoints off.
Postscript (Sept 2026): ub 432→512 re-sweep on ad32k-q4-432 (identical essay probe): PP 2.74→3.01, tg 6.79→6.81 — noise, no gain. Interactive prompts are expert-streaming-bound, not ubatch-bound; ub432 stands.
7k-prefill A/B (7810-tok prompt, fresh slot): ub432 PP 20.95, ub512 PP 21.89 (+4.5%, n=1 each) — real but marginal; repeat on the same slot collapsed to PP 5.17 (second-turn slot-reuse pathology, the #2433 class — see UPSTREAM.md), unrelated to ub. ub432 stands: +4.5% large-prefill PP is not worth re-proving the 30.8k ceiling at higher transient pressure.

## 3. The freeze saga (8 hard freezes → root-caused, fixed)

Three overturned convictions, kept in the record as a warning about premature root-causing:

1. **MXFP4 convicted, then exonerated** — freezes spanned quants with and without MXFP4 tensors.
2. **ASPM exonerated, then convicted** — a sysfs `performance` switch proved nothing (cannot override BIOS-configured ASPM); only the `pcie_aspm=off` kernel param kills it. Decisive evidence: a 5.4k-token prefill burst drove root-port RxErr **7 → 1064 in ~2.5 min**; after the kernel param, **zero errors at boot and idle** (baseline: errors within 3 s, +321/min).
3. **Driver 595 convicted** — `nvidia_uvm` HMM lazy-free list corruption (`LIST_POISON2`, unkillable spinning kthread, zero kernel traces before socklog was installed). HMM-off broke model load instead (both states broken on 595.91 + kernel 6.18), so the real fix was a driver branch change: **580.178.04 LTS**, full parity plus records (tg 7.39, **PP 47.67**).

Mechanism: correctable RxErr → retransmissions → Completion Timeout under max DMA (big prefills + expert streaming + NVMe on one PCH) → link death → hard lockup. Depth/load correlation explained.

## 4. Prompt-cache crash (fixed, validated)

Second turns of real conversations threw `CUDA error: illegal memory access` — the checkpoint apply/invalidate path on hybrid/recurrent state (upstream issue #1762 class). Fix: cherry-picked upstream PR #1976 hunks **plus `--ctx-checkpoints 0`** (cap 0 — no checkpoints ever exist, so the erase path never runs; mismatch degrades to clean full re-prefill). Validated on the exact crash sequence with 0 CUDA errors. Checkpoints are on by default (interval 512, cap 32) and ate **3.6 GB VRAM at ~10.5k tokens** before this — disabling them fixed an OOM class too.

**Evolution (Sept 2026): ckpt-0 was the safe interim, not the end.** Cap-2 proven pointless (0 restores in the wild — the window it covers never misses), then `--ctx-checkpoints 8 --ctx-checkpoints-interval 1024` ladder-proven to **30.8k** on ad32k-q4-432: restores 16–95 ms, suffix-only, 0 fatals. A `spill-checkpoints` branch (local fork, `--ctx-ckpt-spill DIR --ctx-ckpt-live N`) extends this to NVMe (345 MB/s write, ~70 ms from-disk restores, 0 fatals, live-tested, flags dormant in production). Honest caveat: logprob parity shows restores diverge from the cold-prefill baseline from token 1 (5/48 @ 0.38 vs 48/48 clean cold) — restores are fast and stable but not bit-exact; prime suspect is recurrent-state tail mapping on restore. Production runs ckpt 8/1024 on the spill binary; exact-parity is tracked as upstream issue #2433.

## 5. The abort wedge (fixed upstream, PR #2416)

Any fatal (`GGML_ABORT`, in practice always `cuMemCreate` OOM) called `ggml_print_backtrace()`, which `fork()`s — the only `fork()` in all linked code. The child wedged pre-exec (fork+threads/CUDA-fork unsafety, inherited listen socket + CUDA fds) while the parent blocked in `waitpid` forever: HTTP dead, VRAM+port held, every fatal a permanent wedge. Patch: skip the fork when no debugger is installed, timed waitpid + kill, `_exit` in the child. **Verified live 7x**: clean deaths with backtrace in the log, GPU/port self-free, zero orphans. Guard script (`scripts/server-guard.sh`) enforces the rest: singleton discipline, SIGTERM + 60 s wait, refuse duplicates, `-9` only via `Q38_FORCE=1` (a `-9`'d CUDA server corrupts `nvidia_uvm`'s lazy-free list — measured 5.5 GB leak needing rmmod/reboot).

## 6. OOM series: the transient-pool wall

2k-prompt benchmarks never OOM'd, so the wall stayed invisible until real chats hit it. Five captured fatals, all `cuMemCreate reserve_size` during pool growth:

- **#1 (~16k continuation, 32k/m384/ub1024):** attention scratch outgrew the pool at ~5.5/5.7 GB idle. Fix: margin 384 → 768.
- **#2 (same, margin-768):** the hybrid SWA path discards cache (`forcing full prompt re-processing`) and re-prefills the whole ~16–20k prompt while old KV is resident — peak = old KV + full re-prefill transient (~0.5–1 GB+). Fix: move to the 48k recipe (smaller chunks + KV shedding + headroom).
- **#3/#4 (48k/m512, 24.6k extension, warm cache):** even perfect prefix reuse dies past ~25k — the wall is total size. Fix: margin 512 → 1024 without touching ub (whole-layer granularity freed ~800 MB).
- **#5 (cold 22.5k re-prefill):** first fatal on the patched binary — clean death, no wedge, no leak. Patch confirmed live.

Proven along the way: **prefix reuse works** (41- and 153-token tasks, decode 6.2–6.4 t/s — faster than validated, warm cache + short tails); re-prefills happen only on genuine mismatch or cold cache. VRAM returns to 9 MiB clean after every OOM — no leaks anywhere in the series.

AD-4.27 caveat: its Q8_0 dense stack (~5.1 GB at 48k) leaves **40 MB slack** at m512, and no fatter margin fits — the margin treatment does not transfer. AD is the quality tier for short work; long turns need compaction or IQ1_M.

## 7. Full prefill ladder (AD-4.27 @ 32k)

Harness: `scripts/bench-ladder.sh` — accumulating pi-style history, server `prompt_n` ground truth, `max_tokens=16`. Raw logs kept per recipe; rung targets: 2k, 2k, 4k×8.

| recipe | ub | KV | PP (per rung) | last ok (prompt_n) | death |
|---|---|---|---|---|---|
| ad16k | 1024 | q8 | 42.2, 40.7, 39.7 | 7699 | rung 4 OOM |
| ad16k-800 | 800 | q8 | 31.7, 30.3, 32.1, 31.0 | 11543 | rung 5 OOM (~15k attempt) |
| ad16k-s | 512 | q8 | 27.9, 25.7, 24.9, 25.2, 24.9 | 15387 | rung 6, died at 16k ctx wall (not a clean refusal) |
| ad32k-q6 | 512 | q6 | 26.9, 24.9, 24.6, 25.2, 24.9 | 15387 | rung 6 OOM |
| ad32k-q4 | 512 | q4 | 27.1, 25.1, 24.6, 25.2, 24.9, 25.2 | 19231 | rung 7 OOM |
| ad32k-q6-384 | 384 | q6 | 21.0, 18.1, 16.9×6 | 26919 | rung 9 OOM |
| ad32k-q4-384 | 384 | q4 | 22.8, 20.3, 19.0, 17.0, 17.1, 16.9, 16.1, 17.8, 17.1 | **30763** | rung 10 OOM |
| ad32k-q4-432 | 432 | q4 | 21.9, 22.4 | probe only (2 rungs, no OOM) | not laddered to death |

Laws: PP is set by ub alone (KV quant moves nothing); ceiling gains ~+4k per KV step; ub1024 dies ≤11k while ub512 never reaches 30 — (30 t/s, 32k) is off the curve. [Full argument](POST-why-no-30-32k.md).

KV-quant speed tax (AD-4.27 @16k, only KV differs): q8 → tg 7.3, PP 40.6; q6 → tg 4.1, PP 40.1. **q6 KV costs ~40% decode on TU116**, prefill unaffected — quantize KV only when VRAM forces it.

IQ1_M deep ladder (same method family, KV shedding): 32k/q8/ub1024 → 48k/q6/ub512 (PP 30.1) → 64k/Kq6+Vq4/ub384 (PP 25.3) → 96k/Kq6+Vq4/ub512 (PP 29.0) → **128k/q4+had/ub384 (PP 23.9)**, quality probes PASS at every rung; 192k fails fit (needs 5159 > 5142 MiB). 128k is the card ceiling — a 4x window range on 6 GB.

## 8. Production setup

AD-4.27 `ad32k-q4-384` (30.8k proven, PP ~17) + client context 32768 + compaction tripwire (reserve 8192 → fires at 24576; post-compact ~22k; next trigger ~24.5k prefill — all inside proven ground). Skill/toolset kept stable per session (a mid-chat skill load re-renders the tools prefix → total cache miss).
