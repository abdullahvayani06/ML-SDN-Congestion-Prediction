#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-help}"
case "$ACTION" in
 evaluate) cd "$ROOT/results/recorded"; exec python3 "$ROOT/src/evaluation/evaluate_and_plot.py" ;;
 train|baseline|ml|all) ;;
 *) echo 'Usage: bash scripts/run.sh {evaluate|train|baseline|ml|all}'; exit 0 ;;
esac
mkdir -p "$ROOT/results/runs"
WORK=$(mktemp -d "$ROOT/results/runs/${ACTION}-XXXXXXXX")
cp "$ROOT"/src/*/*.py "$WORK/"
cp "$ROOT"/models/*.pkl "$WORK/"
cd "$WORK"
echo "Run directory: $WORK"
if [[ "$ACTION" == train || "$ACTION" == all ]]; then python3 train_model.py; fi
if [[ "$ACTION" == train ]]; then exit 0; fi
PID=""
cleanup() { if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
run_trial() {
 local label="$1" controller=ml_proactive_controller.py attempt
 if [[ "$label" == baseline ]]; then controller=baseline_controller.py; fi
 for attempt in 1 2 3 4; do
  sudo mn -c >/dev/null 2>&1
  SDN_MODE="$label" ryu-manager "$controller" > "ryu_${label}.log" 2>&1 & PID=$!
  sleep 3
  sudo python3 topology.py "$label"
  cleanup; PID=""; sleep 1
  if python3 - "$label" <<'PYEOF'
import sys
from evaluate_and_plot import parse_iperf_and_nic
v = parse_iperf_and_nic(sys.argv[1])
print(f"iPerf datagrams={v[4]}, NIC sent={v[6]}")
sys.exit(0 if v[4]>=44800 and v[6]>=44800 else 1)
PYEOF
  then return 0; fi
 done
 echo "Incomplete workload: $label" >&2; return 1
}
case "$ACTION" in
 baseline) run_trial baseline ;;
 ml) run_trial ml_proactive ;;
 all)
  for mode in baseline thresh_imm_agg thresh_agg ramp_agg ramp_agg_5s rf_nocooldown rf_gbr_nocooldown rf_agg_nocooldown rf_gbr_agg_nocooldown ml_proactive; do
   run_trial "$mode"
   case "$mode" in thresh_imm_agg|ramp_agg|ramp_agg_5s|rf_gbr_agg_nocooldown|ml_proactive) run_trial "${mode}_run2"; run_trial "${mode}_run3";; esac
  done;;
esac
python3 evaluate_and_plot.py
