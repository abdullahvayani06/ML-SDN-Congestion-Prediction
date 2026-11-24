#!/bin/bash
MODE=${1:-ml}

if [ "$MODE" = "baseline" ]; then
    CTRL_SCRIPT="baseline_controller.py"
    LABEL="demo_baseline"
    TITLE="STATIC SHORTEST-PATH BASELINE (No Offloading)"
else
    CTRL_SCRIPT="ml_proactive_controller.py"
    LABEL="demo_ml"
    TITLE="PROPOSED ML-PROACTIVE CONTROLLER (RF+GBR + Agg-Recov + 5s Cooldown)"
fi

for ATTEMPT in 1 2 3; do
    echo "========================================================================"
    echo " LIVE DEMO: ${TITLE} (Attempt ${ATTEMPT})"
    echo "========================================================================"
    sudo killall -9 ryu-manager iperf 2>/dev/null || true
    sudo mn -c > /dev/null 2>&1
    sleep 1

    SDN_MODE=ml_proactive ryu-manager ${CTRL_SCRIPT} > ryu_${LABEL}.log 2>&1 &
    RYU_PID=$!
    sleep 3

    sudo python3 topology.py ${LABEL}
    sudo kill -9 $RYU_PID 2>/dev/null || true
    sleep 1

    SENT=$(python3 -c "import json; print(json.load(open('nic_loss_${LABEL}.json')).get('sent_pkts', 0))" 2>/dev/null || echo 0)
    if [ "$SENT" -ge 44800 ]; then
        echo "[OK] Full workload verified (${SENT} packets >= 44,800)."
        break
    else
        echo "[WARN] VM CPU pause detected (Sent=${SENT} < 44,800). Retrying automatically..."
    fi
done
