# QAT: 3-bit types that work, training short of the bar

Goal: a ~3–3.5 bpw quantization whose grid is learned by training (QAT) instead of fitted post-hoc (PTQ), so deployment *is* the training grid. Base models: Qwen2.5-0.5B (fast loop) and 1.5B (scale check) — the largest a 6 GB card can train through quantization with a CPU-offload AdamW trainer.

Status: **the format/runtime half is done; the training half is not at its bar.** Pre-registered bar: wikitext2 ppl of the Q3 file ≤ 16.5 (stock F16 14.74, stock IQ3_K+imatrix 15.80 — must beat PTQ at fewer bits). Best: **21.88**.

## 1. The types (done, verified)

Three custom GGML types, CPU + CUDA (dp4a) kernels, generation token-identical to F16:

| type | grid | bpw | PTQ ppl, stock 0.5B (wikitext2) |
|---|---|---|---|
| Q3_0_G128 | symmetric 3-bit, group 128 | 3.125 | 223.9 |
| Q3_1_G128 | asymmetric 8-level, group 128 | 3.25 | 74.8 |
| Q3_1_G64 | asymmetric 8-level, group 64 | 3.50 | 35.9 |

Grid sweep (100-row CE, stock 3.13): sym-gs128 6.33 → sym-gs64 5.74 → asym-gs128 5.22 → asym-gs64 4.44. Asymmetry + halved group size buy **−1.9 CE for +0.375 bpw**. All three beat IQ3_K at fewer bits before any training. Sustained kernel throughput ~205 tok/s (dp4a parity class).

## 2. Training series (mixed-export ppl)

Self-distillation only (frozen teacher, full-vocab KL, STE@0, filtered grid keeping embed/lm_head/first+last layers fp, wikitext corpus with 0.3% eval-contaminated blocks dropped — wikitext-2 test is partially inside wikitext-103 train):

| run | ppl | what changed |
|---|---|---|
| V1 | 1028.5 | top-K=32 renormalized targets — tail erosion, failed as designed |
| R3 | 60.1 | full-vocab KL + general corpus + teacher-context bugfix |
| R5 (sym) | 34.3 | mixed-grid export matching the training grid |
| R6 (asym G128) | 25.7 | better grid |
| R8 (asym G64) | **21.9** | best local; transfer gap dg +0.0002 (lossless by construction) |

**Transfer is solved** (the trained weights survive export exactly); **the weights themselves are +0.13 CE over the bar** at 0.5B/data scale. Scale math: closing needs ~100–170k blocks and/or 1.5B scale (1.5B QAT sits at 14.10 on the same grid class). On-grid drift is uniform (~23% relative on all mid-layer groups — no freezable subset, measured, anchor ablated and rejected).

## 3. Failed runs, kept for the lessons

- **Cloud batch-8 on 0.5B diverged twice** (fp 5.06 → 8.17 while batch-1 converges): the batch-1 LR recipe does not transfer; LR must scale with batch and a 500-step gate must pass before committing a full run. ~$1 of tuition.
- **Mixed-grid export silently re-gridded fp layers** (144.9 vs true 34.3): `--custom-q` only applies when the default ftype is quantized — found in `llama-quantize.cpp`, filed upstream as issue #2415.
- **Teacher-context bug**: chunked teacher scoring fed context-free slices (corrupted half the targets); a full-context pass halved ppl alone.
- **V1 top-K**: renormalized 32-top targets carry zero mass on ~20% of true tokens — guaranteed tail erosion, ~1000 ppl.
- **Constant-rate anchor**: consistently 0.03–0.05 CE worse than unanchored — rejected by A/B.
- **Checkpoint averaging**: midpoint loses on a still-descending trajectory — rejected.

## 4. Side result: IQ3_KT (upstream #2414)

Reported the shipped IQ3_KT codebook's ~22% weight error and 0.5B collapse with repro + workarounds; maintainer showed it holds at 1.5B, and I verified his claim myself (F16 3.15 / iq3_k 3.33 / iq3_kt 3.49 — no collapse, conceded with numbers). Lesson kept: 0.5B fragility does not generalize; size-match every claim to its evidence.

## 5. Artifacts

https://huggingface.co/TitanMamo10/qwen25-qat-q3-poc — stock F16 reference (14.74), 0.5B QAT (21.88), 1.5B QAT (14.10), README with verify commands. Needs a build with the Q3 kernels for the QAT files; the F16 verifies anywhere (see [REPRODUCE.md](REPRODUCE.md)).
