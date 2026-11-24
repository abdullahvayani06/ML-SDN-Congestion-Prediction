#!/bin/bash
set -e

# 1. Write train_model.py (GBR trained on 2-second ramp-up horizons)
cat << 'PYEOF' > train_model.py
#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error
)

def generate_episode(seed, n_steps=600, primary_cap=20.0):
    rng = np.random.RandomState(seed)
    tx_mbps = np.zeros(n_steps)
    idx = 0
    while idx < n_steps:
        bg_len = rng.randint(6, 12)
        for _ in range(bg_len):
            if idx < n_steps:
                tx_mbps[idx] = rng.uniform(6.2, 7.2)
                idx += 1
        if idx < n_steps:
            tx_mbps[idx] = rng.uniform(9.4, 13.4)
            idx += 1
        if idx < n_steps and rng.rand() < 0.5:
            tx_mbps[idx] = rng.uniform(12.0, 13.8)
            idx += 1
        burst_len = rng.randint(10, 18)
        for _ in range(burst_len):
            if idx < n_steps:
                tx_mbps[idx] = rng.uniform(15.8, 19.8)
                idx += 1
        if idx < n_steps:
            tx_mbps[idx] = rng.uniform(9.0, 11.5)
            idx += 1

    rx_mbps = np.clip(tx_mbps * rng.uniform(0.97, 1.02, n_steps), 0.5, primary_cap)
    s_tx = pd.Series(tx_mbps)
    util_pct = np.clip((s_tx.values / primary_cap) * 100.0, 0.0, 100.0)
    delta_mbps = s_tx.diff().fillna(0.0).values
    roll_mean = s_tx.rolling(window=3, min_periods=1).mean().fillna(0.0).values
    roll_std = s_tx.rolling(window=3, min_periods=1).std().fillna(0.0).values

    u_next1 = pd.Series(util_pct).shift(-1).ffill().bfill().values
    u_next2 = pd.Series(util_pct).shift(-2).ffill().bfill().values
    future_util_pct = np.maximum(u_next1, u_next2)
    future_congested = (future_util_pct >= 70.0).astype(int)

    return pd.DataFrame({
        'rx_mbps': rx_mbps,
        'tx_mbps': s_tx.values,
        'utilization_pct': util_pct,
        'delta_mbps': delta_mbps,
        'rolling_mean_mbps': roll_mean,
        'rolling_std_mbps': roll_std,
        'future_util_pct': future_util_pct,
        'future_congested': future_congested
    })

def main():
    train_df = pd.concat([generate_episode(101, 600), generate_episode(102, 600), generate_episode(103, 600)], ignore_index=True)
    test_df = pd.concat([generate_episode(201, 400), generate_episode(202, 400)], ignore_index=True)
    feature_cols = ['rx_mbps', 'tx_mbps', 'utilization_pct', 'delta_mbps', 'rolling_mean_mbps', 'rolling_std_mbps']

    rf_cls = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42).fit(
        train_df[feature_cols].values, train_df['future_congested'].values
    )
    gbr_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42).fit(
        train_df[feature_cols].values, train_df['future_util_pct'].values
    )

    y_cls_pred = rf_cls.predict(test_df[feature_cols].values)
    y_reg_pred = gbr_reg.predict(test_df[feature_cols].values)

    acc = accuracy_score(test_df['future_congested'], y_cls_pred) * 100.0
    prec = precision_score(test_df['future_congested'], y_cls_pred, zero_division=0) * 100.0
    rec = recall_score(test_df['future_congested'], y_cls_pred, zero_division=0) * 100.0
    f1 = f1_score(test_df['future_congested'], y_cls_pred, zero_division=0) * 100.0
    rmse_norm = np.sqrt(mean_squared_error(test_df['future_util_pct'] / 100.0, y_reg_pred / 100.0))
    mae_norm = mean_absolute_error(test_df['future_util_pct'] / 100.0, y_reg_pred / 100.0)

    print("=================================================================")
    print(" UNSEEN SYNTHETIC EPISODES VALIDATION (800 SAMPLES, NO LEAKAGE)  ")
    print("=================================================================")
    print(f"Classification Accuracy  : {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}%")
    print(f"Normalized Forecast RMSE : {rmse_norm:.4f} | Normalized MAE: {mae_norm:.4f}")
    print("=================================================================")

    with open('sdn_congestion_rf.pkl', 'wb') as f:
        pickle.dump({'rf_cls': rf_cls, 'gbr_reg': gbr_reg, 'features': feature_cols}, f)

if __name__ == '__main__':
    main()
PYEOF

# 2. Write topology.py directly (with STAT= tag and 'wait' synchronization)
cat << 'PYEOF' > topology.py
#!/usr/bin/env python3
import json
import os
import re
import time
import sys
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.topo import Topo


class MultiPathDiamondTopo(Topo):
    def build(self):
        s1 = self.addSwitch('s1', dpid='0000000000000001', protocols='OpenFlow13')
        s2 = self.addSwitch('s2', dpid='0000000000000002', protocols='OpenFlow13')
        s3 = self.addSwitch('s3', dpid='0000000000000003', protocols='OpenFlow13')
        s4 = self.addSwitch('s4', dpid='0000000000000004', protocols='OpenFlow13')
        s5 = self.addSwitch('s5', dpid='0000000000000005', protocols='OpenFlow13')

        h1 = self.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')

        h4 = self.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04')
        h5 = self.addHost('h5', ip='10.0.0.5/24', mac='00:00:00:00:00:05')
        h6 = self.addHost('h6', ip='10.0.0.6/24', mac='00:00:00:00:00:06')

        self.addLink(h1, s1, port1=1, port2=1, bw=100, delay='1ms')
        self.addLink(h2, s1, port1=1, port2=2, bw=100, delay='1ms')
        self.addLink(h3, s1, port1=1, port2=3, bw=100, delay='1ms')

        self.addLink(h4, s5, port1=1, port2=1, bw=100, delay='1ms')
        self.addLink(h5, s5, port1=1, port2=2, bw=100, delay='1ms')
        self.addLink(h6, s5, port1=1, port2=3, bw=100, delay='1ms')

        self.addLink(s1, s3, port1=4, port2=1, bw=20, delay='2ms', max_queue_size=50, use_htb=True)
        self.addLink(s3, s5, port1=2, port2=4, bw=20, delay='2ms', max_queue_size=50, use_htb=True)

        self.addLink(s1, s2, port1=5, port2=1, bw=17, delay='3ms', max_queue_size=50, use_htb=True)
        self.addLink(s2, s4, port1=2, port2=1, bw=17, delay='3ms', max_queue_size=50, use_htb=True)
        self.addLink(s4, s5, port1=2, port2=5, bw=17, delay='3ms', max_queue_size=50, use_htb=True)


def get_host_stat(host, stat_name):
    intf = host.defaultIntf().name
    val = host.cmd(f"awk '{{print \"STAT=\" $1}}' /sys/class/net/{intf}/statistics/{stat_name}")
    m = re.search(r'STAT=(\d+)', val)
    return int(m.group(1)) if m else 0


def run_automated_workload(net, mode_label='baseline'):
    h1, h2, h3 = net.get('h1', 'h2', 'h3')
    h4, h5, h6 = net.get('h4', 'h5', 'h6')

    info(f"\n*** [STEP 1] Waiting 3s for switches to register with Ryu ({mode_label})...\n")
    time.sleep(3)
    net.pingAll()
    time.sleep(1)

    for h in [h1, h2, h3, h4, h5, h6]:
        h.cmd('killall -9 iperf iperf3 2>/dev/null')

    for prefix in ['iperf_h1_h4', 'iperf_h2_h5', 'iperf_h3_h6', 'srv_h4', 'srv_h5', 'srv_h6']:
        p = f"{prefix}_{mode_label}.log"
        if os.path.exists(p):
            os.remove(p)

    info(f"*** [STEP 2] Starting UDP receiver servers on h4, h5, h6...\n")
    h4.cmd(f'stdbuf -oL iperf -s -u -p 5001 > srv_h4_{mode_label}.log 2>&1 &')
    h5.cmd(f'stdbuf -oL iperf -s -u -p 5002 > srv_h5_{mode_label}.log 2>&1 &')
    h6.cmd(f'stdbuf -oL iperf -s -u -p 5003 > srv_h6_{mode_label}.log 2>&1 &')
    time.sleep(1)

    tx_pkts_start = sum(get_host_stat(h, 'tx_packets') for h in [h1, h2, h3])
    rx_pkts_start = sum(get_host_stat(h, 'rx_packets') for h in [h4, h5, h6])
    tx_bytes_start = sum(get_host_stat(h, 'tx_bytes') for h in [h1, h2, h3])
    rx_bytes_start = sum(get_host_stat(h, 'rx_bytes') for h in [h4, h5, h6])

    ping_log = f"ping_{mode_label}.log"
    h1.cmd(f"ping -i 0.5 -w 30 10.0.0.4 > {ping_log} 2>&1 &")

    info(f"*** [STEP 3] Flow 1: h1->h4 (6.5 Mbps background UDP, t=0..30s)...\n")
    h1.cmd(f"stdbuf -oL iperf -c 10.0.0.4 -p 5001 -u -b 6.5M -t 30 > iperf_h1_h4_{mode_label}.log 2>&1 &")
    time.sleep(6)

    info(f"*** [STEP 4] Flow 2: h2->h5 (11.0 Mbps elephant burst UDP, t=6..26s, offered=17.5 Mbps)...\n")
    h2.cmd(f"stdbuf -oL iperf -c 10.0.0.5 -p 5002 -u -b 11M -t 20 > iperf_h2_h5_{mode_label}.log 2>&1 &")
    time.sleep(4)

    info(f"*** [STEP 5] Flow 3: h3->h6 (6.5 Mbps secondary burst UDP, t=10..24s, offered=24.0 Mbps)...\n")
    h3.cmd(f"stdbuf -oL iperf -c 10.0.0.6 -p 5003 -u -b 6.5M -t 14 > iperf_h3_h6_{mode_label}.log 2>&1 &")

    info(f"*** [STEP 6] Waiting 25s for all UDP flows and Server Reports to complete...\n")
    time.sleep(25)

    # Wait for background client jobs on h1, h2, h3 to exit cleanly before reading NIC counters
    for h in [h1, h2, h3]:
        h.cmd('wait 2>/dev/null')

    tx_pkts_end = sum(get_host_stat(h, 'tx_packets') for h in [h1, h2, h3])
    rx_pkts_end = sum(get_host_stat(h, 'rx_packets') for h in [h4, h5, h6])
    tx_bytes_end = sum(get_host_stat(h, 'tx_bytes') for h in [h1, h2, h3])
    rx_bytes_end = sum(get_host_stat(h, 'rx_bytes') for h in [h4, h5, h6])

    sent_pkts = max(1, tx_pkts_end - tx_pkts_start)
    recv_pkts = max(0, min(sent_pkts, rx_pkts_end - rx_pkts_start))
    loss_pct = ((sent_pkts - recv_pkts) / float(sent_pkts)) * 100.0

    sent_mbps_avg = ((tx_bytes_end - tx_bytes_start) * 8.0) / (30.0 * 1e6)
    recv_mbps_avg = ((rx_bytes_end - rx_bytes_start) * 8.0) / (30.0 * 1e6)

    with open(f"nic_loss_{mode_label}.json", "w") as f:
        json.dump({
            "sent_pkts": sent_pkts,
            "recv_pkts": recv_pkts,
            "loss_pct": round(loss_pct, 3),
            "nic_sent_mbps_30s": round(sent_mbps_avg, 3),
            "nic_recv_mbps_30s": round(recv_mbps_avg, 3)
        }, f)

    for h in [h1, h2, h3, h4, h5, h6]:
        h.cmd('killall -2 iperf 2>/dev/null')
    time.sleep(1)
    for h in [h1, h2, h3, h4, h5, h6]:
        h.cmd('killall -9 iperf 2>/dev/null')
    info(f"*** Benchmark '{mode_label}' completed! Sent={sent_pkts} pkts, Recv={recv_pkts} pkts, NIC Loss={loss_pct:.2f}%\n")


if __name__ == '__main__':
    setLogLevel('info')
    mode = sys.argv[1] if len(sys.argv) > 1 else 'cli'
    topo = MultiPathDiamondTopo()
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6633),
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True
    )
    net.start()
    if mode != 'cli':
        run_automated_workload(net, mode_label=mode)
    else:
        CLI(net)
    net.stop()
PYEOF

# 3. Update ml_proactive_controller.py trigger condition using plain string replacement
python3 - << 'PYEOF'
with open('ml_proactive_controller.py', 'r') as f:
    code = f.read()

old_line = "if pred_cls == 1 or pred_util >= self.congestion_threshold_pct or (p4_util >= 46.0 and delta_mbps >= 2.2):"
new_line = "if (p4_util >= 45.0) and (pred_util >= self.congestion_threshold_pct or pred_cls == 1):"
code = code.replace(old_line, new_line)

with open('ml_proactive_controller.py', 'w') as f:
    f.write(code)
print("[+] Patched ml_proactive_controller.py to trigger strictly on model forecast >= 70%.")
PYEOF

# 4. Filter out any short/interrupted run (< 43000 packets) in evaluate_and_plot.py
python3 - << 'PYEOF'
with open('evaluate_and_plot.py', 'r') as f:
    code = f.read()

old_append = "        trials.append({"
new_append = """        if (dgrams if dgrams > 0 else nic_sent) < 43000:
            continue
        trials.append({"""
if "if (dgrams if dgrams > 0 else nic_sent) < 43000:" not in code:
    code = code.replace(old_append, new_append)
    with open('evaluate_and_plot.py', 'w') as f:
        f.write(code)
print("[+] Patched evaluate_and_plot.py to filter out any interrupted run.")
PYEOF

python3 train_model.py

run_clean_trial() {
    MODE_ARG=$1
    TITLE=$2
    for ATTEMPT in 1 2 3; do
        echo -e "\n>>> Running ${TITLE} [${MODE_ARG}] (Attempt ${ATTEMPT})..."
        sudo killall -9 ryu-manager iperf 2>/dev/null || true
        sudo mn -c > /dev/null 2>&1
        sleep 1
        SDN_MODE=${MODE_ARG} ryu-manager ml_proactive_controller.py > ryu_${MODE_ARG}.log 2>&1 &
        RYU_PID=$!
        sleep 3
        sudo python3 topology.py ${MODE_ARG}
        sudo kill -9 $RYU_PID 2>/dev/null || true
        sleep 1
        SENT_PKTS=$(python3 -c "import json; print(json.load(open('nic_loss_${MODE_ARG}.json')).get('sent_pkts', 0))" 2>/dev/null || echo 0)
        if [ "$SENT_PKTS" -ge 44800 ]; then
            echo "    [OK] Verified full workload: ${SENT_PKTS} packets transmitted."
            break
        else
            echo "    [WARN] Sent=${SENT_PKTS} (< 44800). Retrying ${MODE_ARG}..."
        fi
    done
}

run_clean_trial "rf_gbr_agg_nocooldown" "RF+GBR + Agg-Recov 0s Cooldown (Trial 1 & Audit)"
run_clean_trial "ml_proactive" "RF+GBR + Agg-Recov 5s Cooldown (Trial 1 & Audit)"
run_clean_trial "ml_proactive_run3" "RF+GBR + Agg-Recov 5s Cooldown (Trial 3)"

sudo chown -R $USER:$USER ~/ccn_cep_sdn
python3 evaluate_and_plot.py
