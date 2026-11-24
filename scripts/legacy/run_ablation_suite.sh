#!/bin/bash
# Automated 6-Way Ablation Suite Runner with Sent-Packet Verification Gate
set -e

echo "=================================================================="
echo " Starting Verified 6-Way SDN Ablation Suite (Mininet + Ryu)       "
echo "=================================================================="

python3 train_model.py

run_mode() {
    MODE_NAME=$1
    CTRL_SCRIPT=$2
    STEP_TITLE=$3

    for ATTEMPT in 1 2 3; do
        echo -e "\n>>> ${STEP_TITLE} (Attempt ${ATTEMPT})..."
        sudo mn -c > /dev/null 2>&1
        sleep 1
        SDN_MODE=${MODE_NAME} ryu-manager ${CTRL_SCRIPT} > ryu_${MODE_NAME}.log 2>&1 &
        RYU_PID=$!
        sleep 2
        sudo python3 topology.py ${MODE_NAME}
        kill -9 $RYU_PID 2>/dev/null || true
        sleep 1

        # Verify that the full ~45,200 packet workload completed without a VM pause
        SENT_PKTS=$(python3 -c "import json; print(json.load(open('nic_loss_${MODE_NAME}.json')).get('sent_pkts', 0))" 2>/dev/null || echo 0)
        if [ "$SENT_PKTS" -ge 43000 ]; then
            echo "    [OK] Workload verified: ${SENT_PKTS} packets transmitted."
            break
        else
            echo "    [WARN] Short transmission (${SENT_PKTS} pkts < 43000). Retrying ${MODE_NAME}..."
        fi
    done
}

run_mode "baseline" "baseline_controller.py" "[1/6] Baseline (Static Shortest-Path)"
run_mode "rf_nocooldown" "ml_proactive_controller.py" "[2/6] RF-Only (Single-Port Recovery, 0s Cooldown)"
run_mode "rf_gbr_nocooldown" "ml_proactive_controller.py" "[3/6] RF + GBR (Single-Port Recovery, 0s Cooldown)"
run_mode "rf_agg_nocooldown" "ml_proactive_controller.py" "[4/6] RF-Only + Aggregate-Load Recovery (0s Cooldown)"
run_mode "rf_gbr_agg_nocooldown" "ml_proactive_controller.py" "[5/6] RF + GBR + Aggregate-Load Recovery (0s Cooldown)"
run_mode "ml_proactive" "ml_proactive_controller.py" "[6/6] RF + GBR + Aggregate-Load Recovery (5s Cooldown)"

sudo chown -R $USER:$USER ~/ccn_cep_sdn
python3 evaluate_and_plot.py
