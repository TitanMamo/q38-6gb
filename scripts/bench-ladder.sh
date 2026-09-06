#!/usr/bin/env bash
# bench-ladder.sh — prefill ladder: one growing conversation (pi-style: the
# client resends FULL history every turn). Measures prompt-processing speed
# per rung and finds the OOM ceiling, i.e. the largest total context this
# recipe survives. Previous validations only used 2k prompts and missed the
# transient-pool OOM class entirely.
#
# Usage: ./bench-ladder.sh LABEL
# Assumes a HEALTHY server on :8013 (launch recipe first). STOPS at the first
# OOM/server death and reports the ceiling. All output -> logs/ladder-LABEL.log
#
# Method: history accumulates (user chunk + canned "Noted." ack per rung, like
# a real chat). Ground truth = server-reported prompt_n per rung. max_tokens=16
# keeps decode trivial — this ladder measures PREFILL.
# New tokens per rung: 2k,2k,4k x8 (context ~2,4,8,...,36k).
set -u

LABEL="${1:-ladder}"
PORT="${Q38_PORT:-8013}"
BASE="http://127.0.0.1:${PORT}"
LOGDIR=/mnt/Data/ik_llama.cpp-upstream/logs
LOG="$LOGDIR/ladder-$LABEL.log"
HIST="$LOGDIR/ladder-$LABEL-history.json"

log() { echo "$@" | tee -a "$LOG"; }
START="${RESUME_FROM:-1}"
if [ "$START" -le 1 ]; then
  : > "$LOG"
  echo "[]" > "$HIST"
else
  log "resuming $LABEL at rung $START (history kept)"
fi

curl -s --max-time 10 "$BASE/health" | grep -q '"status":"ok"' \
  || { echo "no healthy server on :$PORT — launch a recipe first"; exit 1; }

log "ladder $LABEL starting $(date -u +%FT%TZ) — accumulating history"
RUNGS="${LADDER_RUNGS:-2048 2048 4096 4096 4096 4096 4096 4096 4096 4096}"
RUNG=0
for TGT in $RUNGS; do
  RUNG=$((RUNG + 1))
  if [ "$RUNG" -lt "$START" ]; then continue; fi
  RESP=$(python3 - "$TGT" "$BASE" "$HIST" <<'PYEOF'
import json, sys, urllib.request
tgt, base, histpath = int(sys.argv[1]), sys.argv[2], sys.argv[3]
para = ("Die Energiewende erfordert massive Investitionen in Netze, Speicher und Erzeugung. "
        "Photovoltaik und Windkraft liefern inzwischen die guenstigsten Kilowattstunden, "
        "doch ihre Volatilitaet verlangt flexible Lasten und Langzeitspeicher. "
        "Wasserstoff aus Elektrolyse gilt als Schluessel fuer Dunkelflauten, "
        "waehrend Batterien Kurzfristschwankungen ausgleichen. ")
text = (para * (tgt * 4 // len(para) + 2))[:tgt * 4]
msgs = json.load(open(histpath))
msgs.append({"role": "user", "content": text})
body = json.dumps({"model": "Qwen3.8-Flash-Next",
                   "messages": msgs,
                   "max_tokens": 16}).encode()
req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=3500) as r:
        print(r.read().decode())
    msgs.append({"role": "assistant", "content": "Noted."})
    json.dump(msgs, open(histpath, "w"))
except Exception as e:
    sys.stderr.write("REQUEST FAILED: %r\n" % e)
    sys.exit(3)
PYEOF
)
  RC=$?
  if [ $RC -ne 0 ] || [ -z "$RESP" ]; then
    log "rung $RUNG (~${TGT} new): REQUEST FAILED rc=$RC — waiting 15s then health-checking"
    sleep 15
    if curl -s --max-time 10 "$BASE/health" | grep -q '"status":"ok"'; then
      if pgrep -f "[l]lama-server" >/dev/null; then
        log "server ALIVE but rung failed — likely ctx-limit refusal (HTTP 500), not OOM. Check server log for send_error."
        log "RESULT: VRAM ceiling NOT reached; ctx wall or refusal at this size"
        exit 3
      fi
      log "server alive but rung failed — ABORTING ladder (investigate log)"
      exit 1
    else
      PREV=$(python3 -c "import json;print(sum(len(m['content'])//4 for m in json.load(open('$HIST')) if m['role']=='user'))" 2>/dev/null || echo "?")
      log "SERVER DEAD — OOM ceiling near ~${PREV} chars of history (rung $RUNG)"
      log "CEILING: investigate server log tail for the fatal line"
      exit 2
    fi
  fi
  NP=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['timings']['prompt_n'])" 2>/dev/null || echo "?")
  PP=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['timings']['prompt_per_second'])" 2>/dev/null || echo "?")
  FR=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['finish_reason'])" 2>/dev/null || echo "?")
  log "rung $RUNG: target_new=$TGT context_prompt_n=$NP PP=$PP finish=$FR"
done
log "LADDER COMPLETE without OOM"
