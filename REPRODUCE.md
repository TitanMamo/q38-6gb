# Reproduce

## Serving

Build (ik_llama.cpp tree): standard CUDA build plus `GGML_CUDA_FORCE_MMQ=ON`. Runtime env: `GGML_CUDA_NO_PINNED=1`, `LD_LIBRARY_PATH` pointed at the staged 580 driver libs + build `src`/`ggml/src`.

Server flags (AD-4.27 max-context recipe — the full flag set matters):

```
llama-server --model <AD-4.27 shards> --ctx-size 32768 --fit --fit-margin 384 \
  --prefetch-experts --defer-ple --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 -khad -vhad -ictk q8_0 \
  -wgt 1 --ctx-checkpoints-interval 0 --ctx-checkpoints 0 \
  -b 2048 -ub 384 -t 4 -tb 6 -np 1 --jinja \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 127.0.0.1 --port 8013 --metrics
```

IQ1_M daily driver differs: q8 KV, `-ub 1024`, `--fit-margin 768`, no `-khad/-vhad`.

Ladder (needs a healthy server on :8013, stops at first OOM):

```bash
./scripts/bench-ladder.sh <label>            # logs to logs/ladder-<label>.log
RESUME_FROM=8 ./scripts/bench-ladder.sh <label>   # resume after a restart, history kept
LADDER_RUNGS="2048 2048 4096 ..." ./scripts/bench-ladder.sh <label>
```

Ground truth is server-reported `prompt_n` per rung; `max_tokens=16` isolates prefill. Edit `LOGDIR`/`BASE` at the top of the script for your paths. Guard discipline for all launchers: source `scripts/server-guard.sh`, then `q38_stop_server || exit 1` + `q38_require_free || exit 1` before launch; match processes with the `[l]lama-server` bracket pattern (bare `pkill -f llama-server` matches your own shell).

## QAT / PPL verify

Reference PPL (any stock llama-perplexity build):

```bash
llama-perplexity -m stock-f16.gguf -f ppl_wikitext2.txt -ngl 99 -t 6  # expect ~14.74
```

QAT files need a build with the Q3_0_G128 / Q3_1_G128 / Q3_1_G64 kernels (see artifacts repo README). Mixed-grid files are the training grid (mid blocks quantized, embed/first/last/output fp); pure files quantize everything.

Mixed-grid export (the correct invocation — `--custom-q` requires a quantized default ftype, cf. issue #2415; type names lowercase):

```bash
llama-quantize --custom-q 'blk\.0\..*=f16,blk\.23\..*=f16' \
  --output-tensor-type f16 --token-embedding-type f16 \
  trained-f16.gguf trained-mixed.gguf Q3_1_G64
```
