#!/usr/bin/env bash
# server-guard.sh — shared singleton + kill-discipline guard for ALL
# Qwen3.8-Flash-Next launchers (run-*.sh, bench-*.sh, probe-*.sh).
#
# Incident history (see QWEN38_FLASH_NEXT_RESULTS.md): every duplicate-server
# launch wedges the box — two servers split 5.7GB VRAM, share :8013, and HTTP
# hangs forever. bench-*.sh used `pkill; sleep 3` (CUDA teardown of 5.7GB takes
# 10-60s, so the old server was still alive when the new one bound the port)
# and probe-*.sh launched with no guard at all.
#
# Rules:
#   1. One server at a time. Stop-then-start, never start-then-hope.
#   2. SIGTERM + patient wait (default 60s). NEVER `pkill -9` a live CUDA
#      server — it corrupts nvidia_uvm's lazy-free list (oops/freeze, and the
#      5.5GB VRAM leak that needs rmmod/reboot). -9 only via Q38_FORCE=1,
#      and only after TERM failed.
#   3. Always match with the "[l]lama-server" bracket pattern. Bare
#      `pkill -f llama-server` also matches your own shell when its command
#      line contains that string (e.g. `bash -c '... llama-server ...'`).
#
# Usage: source this file, then call:
#   q38_stop_server    # TERM + wait; returns 0 when no server remains,
#                      # 1 when something refuses to die (duplicate launch blocked)
#   q38_require_free   # exit 1 if a server pid OR a healthy :PORT answers
#
# Knobs: Q38_PORT (default 8013), Q38_WAIT (default 60s), Q38_FORCE=1 (allow -9).

Q38_PORT="${Q38_PORT:-8013}"
Q38_WAIT="${Q38_WAIT:-60}"

q38_pids() { pgrep -f "[l]lama-server" 2>/dev/null || true; }

q38_healthy() {
  curl -s --max-time 2 "http://127.0.0.1:${Q38_PORT}/health" 2>/dev/null \
    | grep -q '"status":"ok"'
}

# Wait until every pid in $1 is gone (or timeout seconds in $2). Returns 0/1.
q38_wait_gone() {
  local pids="$1" timeout="$2" waited=0
  while [ "$waited" -lt "$timeout" ]; do
    local alive=""
    local pid
    for pid in $pids; do
      kill -0 "$pid" 2>/dev/null && alive="$alive $pid"
    done
    [ -z "$alive" ] && return 0
    sleep 2
    waited=$((waited + 2))
  done
  return 1
}

q38_stop_server() {
  local pids
  pids="$(q38_pids)"
  if [ -z "$pids" ] && ! q38_healthy; then
    echo "guard: no llama-server running, port ${Q38_PORT} free"
    return 0
  fi
  # A foreign process may hold the port without a matching pid; still refuse.
  if [ -z "$pids" ]; then
    echo "guard: port ${Q38_PORT} answers but no llama-server pid found — refusing (stop it first)"
    return 1
  fi
  echo "guard: stopping [$pids] (SIGTERM, up to ${Q38_WAIT}s for CUDA teardown)..."
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  if q38_wait_gone "$pids" "$Q38_WAIT"; then
    echo "guard: stopped clean"
  else
    if [ "${Q38_FORCE:-}" = "1" ]; then
      echo "guard: WARNING: TERM failed, escalating to -9 (UVM corruption risk — watch dmesg, check nvidia-smi for leaked VRAM)"
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
      sleep 3
      if [ -n "$(q38_pids)" ]; then
        echo "guard: even -9 failed — wedged. Try: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm (or reboot)"
        return 1
      fi
    else
      echo "guard: old server won't die gracefully in ${Q38_WAIT}s (wedged?) — refusing duplicate."
      echo "guard: wait for it, or re-run with Q38_FORCE=1 (risks UVM corruption — last resort only)"
      return 1
    fi
  fi
  if q38_healthy; then
    echo "guard: pids gone but port ${Q38_PORT} still answers — refusing duplicate"
    return 1
  fi
  return 0
}

q38_require_free() {
  local pids
  pids="$(q38_pids)"
  if [ -n "$pids" ]; then
    echo "guard: llama-server already running [$pids] — refusing duplicate (stop it first)"
    return 1
  fi
  if q38_healthy; then
    echo "guard: a server is already healthy on :${Q38_PORT} — refusing duplicate"
    return 1
  fi
  return 0
}
