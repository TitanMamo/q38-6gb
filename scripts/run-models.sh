#!/usr/bin/env bash
# Qwen3.8-Flash-Next launcher — all validated recipes in one file.
# RUN INSIDE THE CONTAINER:
#   distrobox-enter -n cuda-box -- /home/titan/Downloads/run-models.sh <recipe>
#
# Recipes (measured on GTX 1660 Ti 6GB / i5-9400F / 32GB, driver 580):
#   iq1m-32k   IQ1_M @ 32K, q8 KV, ub1024, m768      tg 7.82 / PP 47.67  (DAILY DRIVER; m768 after 16k-prompt OOM 2026-09-04)
#   iq1m-48k   IQ1_M @ 48K, q6 KV, ub512, m1024     PP 30.1 / gen 4.83 (m1024 after 24k-ext OOMs 2026-09-05, ub untouched)
#   iq1m-64k   IQ1_M @ 64K, Kq6+Vq4+had, ub384       PP 25.3 / gen 5.35
#   iq1m-96k   IQ1_M @ 96K, Kq6+Vq4+had, ub512       PP 29.0 / gen 4.35
#   iq1m-128k  IQ1_M @128K, q4+had, ub384            PP 23.9 / gen 4.00  (ceiling: 192K fails fit)
#   ad16k      AD-4.27 @ 16K, q8 KV, ub1024          tg 7.31 / PP 40.63  (quality tier, 89.5% top-1)
#   ad48k      AD-4.27 @ 48K, q4+had, ub128          PP 6.31 / gen 2.88  (max window, slow; m512 ONLY - dense too fat for more)
#   iq1m-32k-s IQ1_M @ 32K, q8 KV, ub512, m768      (ladder variant: half chunks, same margin)
#   ad16k-s    AD-4.27 @ 16K, q8 KV, ub512           (ladder variant: does halving chunks save AD?)
#   ad32k-q6   AD-4.27 @ 32K, q6 KV, ub512           (experimental: 32k needs KV quant on AD dense)
#   ad32k-q4   AD-4.27 @ 32K, q4+had KV, ub512       (fallback if q6 OOMs; decode will suffer, PP should hold)
# Swap quants NVMe<->HDD with: /mnt/Data/ik_llama.cpp-upstream/model-ctl.sh {list|archive|use|status}
# Logs: tail -f /mnt/Data/ik_llama.cpp-upstream/logs/q38-server.log
set -euo pipefail

if [ ! -f /.dockerenv ] && [ -z "${CONTAINER_ID:-}" ] && [ ! -f /run/.containerenv ]; then
  echo "Run this INSIDE the container:"
  echo "  distrobox-enter -n cuda-box -- $0 ${1:-<recipe>}"
  exit 1
fi

RECIPE="${1:-}"
# Fatals (CUDA errors, ggml_abort) must land in a file, not a lost terminal —
# the abort-path fork wedges the server holding port+VRAM, and the message is
# the only clue. Everything below appends to the canonical log.
LOG=/mnt/Data/ik_llama.cpp-upstream/logs/q38-server.log
echo "run-models.sh $RECIPE — logging to $LOG (tail -f $LOG)"
exec >>"$LOG" 2>&1
LIVE=/home/titan/models-q38
BIN_DIR=/mnt/Data/ik_llama.cpp-upstream/build-upstream-cuda-mmq/bin
export LD_LIBRARY_PATH=/mnt/Data/cuda-driver-libs-580:/mnt/Data/ik_llama.cpp-upstream/build-upstream-cuda-mmq/src:/mnt/Data/ik_llama.cpp-upstream/build-upstream-cuda-mmq/ggml/src:/mnt/Data/ik_llama.cpp-upstream/build-upstream-cuda-mmq/examples/mtmd:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GGML_CUDA_NO_PINNED=1

# defaults (overridden per recipe)
QDIR=""; CTX=32768; MARGIN=384; UBU=1024; BB=2048
CACHE="--cache-type-k q8_0 --cache-type-v q8_0"

case "$RECIPE" in
  iq1m-32k)  QDIR=UD-IQ1_M;            CTX=32768;  MARGIN=768; UBU=1024; BB=2048; CACHE="--cache-type-k q8_0 --cache-type-v q8_0" ;;
  iq1m-48k)  QDIR=UD-IQ1_M;            CTX=49152;  MARGIN=1024; UBU=512;  BB=2048; CACHE="--cache-type-k q6_0 --cache-type-v q6_0 -ictk q8_0" ;;
  iq1m-64k)  QDIR=UD-IQ1_M;            CTX=65536;  MARGIN=512; UBU=384;  BB=2048; CACHE="--cache-type-k q6_0 --cache-type-v q4_0 -vhad -ictk q8_0" ;;
  iq1m-96k)  QDIR=UD-IQ1_M;            CTX=98304;  MARGIN=512; UBU=512;  BB=2048; CACHE="--cache-type-k q6_0 --cache-type-v q4_0 -vhad -ictk q8_0" ;;
  iq1m-128k) QDIR=UD-IQ1_M;            CTX=131072; MARGIN=512; UBU=384;  BB=2048; CACHE="--cache-type-k q4_0 --cache-type-v q4_0 -khad -vhad -ictk q8_0" ;;
  iq1m-32k-s) QDIR=UD-IQ1_M;           CTX=32768;  MARGIN=768; UBU=512;  BB=2048; CACHE="--cache-type-k q8_0 --cache-type-v q8_0" ;;
   ad16k)     QDIR=AD-4.27bpw-Q4_K_M-M64; CTX=16384; MARGIN=384; UBU=1024; BB=2048; CACHE="--cache-type-k q8_0 --cache-type-v q8_0" ;;
   ad16k-800) QDIR=AD-4.27bpw-Q4_K_M-M64; CTX=16384; MARGIN=384; UBU=800;  BB=2048; CACHE="--cache-type-k q8_0 --cache-type-v q8_0" ;;  # MEASURED 2026-09-06: PP ~31, proven 11543, OOM rung 5 (~15k). Production pick: speed + 10k usable.
  ad16k-s)   QDIR=AD-4.27bpw-Q4_K_M-M64; CTX=16384; MARGIN=384; UBU=512;  BB=2048; CACHE="--cache-type-k q8_0 --cache-type-v q8_0" ;;
  ad32k-q6)  QDIR=AD-4.27bpw-Q4_K_M-M64; CTX=32768; MARGIN=384; UBU=512;  BB=2048; CACHE="--cache-type-k q6_0 --cache-type-v q6_0 -ictk q8_0" ;;
  ad32k-q6-384) QDIR=AD-4.27bpw-Q4_K_M-M64; CTX=32768; MARGIN=384; UBU=384; BB=2048; CACHE="--cache-type-k q6_0 --cache-type-v q6_0 -ictk q8_0" ;;
  ad32k-q6u) QDIR=AD-4.27bpw-Q4_K_M-M64; CTX=32768; MARGIN=384; UBU=1024; BB=2048; CACHE="--cache-type-k q6_0 --cache-type-v q6_0 -ictk q8_0" ;;
  ad32k-q4)  QDIR=AD-4.27bpw-Q4_K_M-M64; CTX=32768; MARGIN=384; UBU=512;  BB=2048; CACHE="--cache-type-k q4_0 --cache-type-v q4_0 -khad -vhad -ictk q8_0" ;;
  ad32k-q4-384) QDIR=AD-4.27bpw-Q4_K_M-M64; CTX=32768; MARGIN=384; UBU=384; BB=2048; CACHE="--cache-type-k q4_0 --cache-type-v q4_0 -khad -vhad -ictk q8_0" ;;
  ad32k-q4-432) QDIR=AD-4.27bpw-Q4_K_M-M64; CTX=32768; MARGIN=384; UBU=432;  BB=1728; CACHE="--cache-type-k q4_0 --cache-type-v q4_0 -khad -vhad -ictk q8_0" ;;
  ad48k)     QDIR=AD-4.27bpw-Q4_K_M-M64; CTX=49152; MARGIN=512; UBU=128;  BB=1024; CACHE="--cache-type-k q4_0 --cache-type-v q4_0 -khad -vhad -ictk q8_0" ;;
   *) echo "usage: $0 {iq1m-32k|iq1m-32k-s|iq1m-48k|iq1m-64k|iq1m-96k|iq1m-128k|ad16k|ad16k-800|ad16k-s|ad32k-q6|ad32k-q6-384|ad32k-q6u|ad32k-q4|ad32k-q4-384|ad32k-q4-432|ad48k}"; exit 1 ;;
esac

GUARD=/mnt/Data/ik_llama.cpp-upstream/server-guard.sh
[ -f "$GUARD" ] || { echo "missing $GUARD"; exit 1; }
# shellcheck disable=SC1090
source "$GUARD"

MODEL=$(ls "$LIVE/$QDIR"/*00001-of-*.gguf 2>/dev/null | head -1)
[ -n "${MODEL:-}" ] || { echo "quant $QDIR not live under $LIVE — install with model-ctl.sh use $QDIR"; exit 1; }
echo "recipe $RECIPE -> $MODEL (ctx $CTX, ub $UBU)"

# Singleton: one server at a time (two split 5.7GB VRAM and wedge on :8013).
# Patient SIGTERM; refuses instead of duplicating when the old server is wedged.
q38_stop_server || exit 1
q38_require_free || exit 1

# NOTE (2026-09-11): ckpt 8/1024 below — restores kill the miss-reprocess OOM
# class (30.8k proven with ~7k prefills, 0 fatals). Was 0/0 (erase-crash
# survival); crash scoped away by scale probes to 31k.
exec taskset -c 0-5 "$BIN_DIR/llama-server" \
  --model "$MODEL" --alias Qwen3.8-Flash-Next \
  --ctx-size "$CTX" --fit --fit-margin "$MARGIN" \
  --prefetch-experts --defer-ple \
  --flash-attn on $CACHE \
  -wgt 1 --ctx-checkpoints-interval 1024 --ctx-checkpoints 8 \
  -b "$BB" -ub "$UBU" -t 4 -tb 6 -np 1 \
  --jinja --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 \
  --host 127.0.0.1 --port 8013 --metrics
