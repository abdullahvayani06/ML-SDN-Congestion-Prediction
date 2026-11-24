#!/bin/bash
set -e

echo "================================================================================"
echo " Running Final Matched-Cooldown Isolation (ramp_agg_5s + 3rd 0s-ML Trial)       "
echo "================================================================================"

# 1. Patch ml_proactive_controller.py to support 'ramp_agg_5s' (Ramp Heuristic + Agg-Recov + 5s Cooldown, No ML)
python3 - << 'PYEOF'
with open('ml_proactive_controller.py', 'r') as f:
    code = f.read()

code = code.replace(
    "self.cooldown_sec = 5.0 if self.base_mode == 'ml_proactive' else 0.0",
    "self.cooldown_sec = 5.0 if self.base_mode in ['ml_proactive', 'ramp_agg_5s'] else 0.0"
)
code = code.replace(
    "'thresh_imm_agg', 'thresh_agg', 'ramp_agg',",
    "'thresh_imm_agg', 'thresh_agg', 'ramp_agg', 'ramp_agg_5s',"
)
code = code.replace(
    "elif self.base_mode == 'ramp_agg':",
    "elif self.base_mode in ['ramp_agg', 'ramp_agg_5s']:"
)

with open('ml_proactive_controller.py', 'w') as f:
    f.write(code)
print("[+] Added 'ramp_agg_5s' (Ramp Heuristic + 5s Cooldown, No ML) to ml_proactive_controller.py")
PYEOF

# 2. Update evaluate_and_plot.py to include '2d. ramp_agg_5s' and align Trigger Times to Flow 1 start (t_bench)
cat << 'PYEOF' > evaluate_and_plot.py
#!/usr/bin/env python3
import os
import re
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_ping_log(filepath):
    if not os.path.exists(filepath):
        return []
    rtts = []
    with open(filepath, 'r') as f:
        for line in f:
            m = re.search(r'time=([\d\.]+)\s*ms', line)
            if m:
                val = float(m.group(1))
                if val <= 450.0:
                    rtts.append(val)
    return rtts


def parse_iperf_and_nic(mode_label, df=None):
    tot_lost, tot_dgrams, tot_rx_mib = 0, 0, 0.0
    for fl in ['h1_h4', 'h2_h5', 'h3_h6']:
        for fn in [f"iperf_{fl}_{mode_label}.log", f"srv_{fl.split('_')[1]}_{mode_label}.log"]:
            if os.path.exists(fn):
                try:
                    with open(fn, 'r', errors='ignore') as f:
                        txt = f.read()
                    matches = re.findall(
                        r'([\d\.]+)\s*MBytes\s+([\d\.]+)\s*Mbits/sec\s+([\d\.]+)\s*ms\s+(\d+)/\s*(\d+)\s*\(([\d\.]+)%\)',
                        txt
                    )
                    if matches:
                        mb_s, _, _, lost_s, tot_s, _ = matches[-1]
                        tot_rx_mib += float(mb_s)
                        tot_lost += int(lost_s)
                        tot_dgrams += int(tot_s)
                        break
                except Exception:
                    pass

    nic_sent, nic_loss = 0, 0.0
    nic_file = f"nic_loss_{mode_label}.json"
    if os.path.exists(nic_file):
        try:
            with open(nic_file, 'r') as f:
                nd = json.load(f)
            nic_sent = int(nd.get('sent_pkts', 0))
            nic_loss = float(nd.get('loss_pct', 0.0))
        except Exception:
            pass

    s5_recv_mbps = 0.0
    if df is not None and 'recv_delivered_mbps' in df.columns:
        valid_s5 = df[df['recv_delivered_mbps'] > 0.5]['recv_delivered_mbps']
        if len(valid_s5) > 0:
            s5_recv_mbps = float(valid_s5.mean())

    iperf_goodput_mbps = (tot_rx_mib * 1024.0 * 1024.0 * 8.0) / (30.0 * 1e6) if tot_rx_mib > 0 else 0.0
    iperf_loss_pct = (tot_lost / float(tot_dgrams)) * 100.0 if tot_dgrams > 0 else nic_loss

    return iperf_goodput_mbps, s5_recv_mbps, iperf_loss_pct, nic_loss, tot_dgrams, tot_lost, nic_sent


def load_active_window(csv_file):
    if not os.path.exists(csv_file):
        return None
    df = pd.read_csv(csv_file)
    if 'total_tx_mbps' not in df.columns:
        return None
    active = df[df['total_tx_mbps'] > 1.0].reset_index(drop=True)
    return active if len(active) >= 10 else df


def extract_aligned_trigger(suffix):
    """
    Computes trigger metrics aligned to Flow 1 start (t_bench = 0.0s, where total_tx_mbps > 1.0).
    Flow 2 (11 Mbps elephant burst) starts at t_bench = 6.0s.
    """
    csv_file = f"telemetry_{suffix}.csv"
    if not os.path.exists(csv_file):
        return None
    df_full = pd.read_csv(csv_file)
    active = df_full[df_full['total_tx_mbps'] > 1.0].reset_index(drop=True)
    if len(active) < 5:
        return None
    t0_elapsed = float(active['elapsed_sec'].iloc[0])

    chg_rows = active[active['route_changes'] >= 1]
    if len(chg_rows) == 0:
        return None
    r_trig = chg_rows.iloc[0]
    trig_idx = int(r_trig.name)
    t_raw = float(r_trig['elapsed_sec'])
    t_bench = round(t_raw - t0_elapsed, 2)
    u_meas = float(r_trig['p4_util_pct'])
    u_pred = float(r_trig['predicted_util_pct'])
    prev_tx = float(active['p4_tx_mbps'].iloc[trig_idx - 1]) if trig_idx > 0 else float(r_trig['p4_tx_mbps'])
    slope_dr = float(r_trig['p4_tx_mbps']) - prev_tx
    disp_ms = float(r_trig['flowmod_ms']) if 'flowmod_ms' in r_trig else 0.5

    reason = 'Measured>=70%'
    af_path = f"trigger_audit_{suffix}.json"
    if os.path.exists(af_path):
        try:
            ad = json.load(open(af_path))
            u_meas = float(ad.get('measured_p4_util_pct', u_meas))
            u_pred = float(ad.get('predicted_util_pct', u_pred))
            slope_dr = float(ad.get('delta_mbps', slope_dr))
            disp_ms = float(ad.get('dispatch_ms', ad.get('flowmod_ms', disp_ms)))
            reason = ad.get('trigger_reason', reason)
            # Re-align audit raw elapsed_sec using t0_elapsed from the same run's CSV
            if 'elapsed_sec' in ad:
                t_bench = round(float(ad['elapsed_sec']) - t0_elapsed, 2)
        except Exception:
            pass

    return {
        't_bench': t_bench,
        't_raw': t_raw,
        'u_meas': u_meas,
        'u_pred': u_pred,
        'slope_dr': slope_dr,
        'disp_ms': disp_ms,
        'reason': reason
    }


def collect_trials(m_key):
    candidates = [m_key] + [
        os.path.basename(p).replace("telemetry_", "").replace(".csv", "")
        for p in sorted(glob.glob(f"telemetry_{m_key}_run*.csv"))
    ]
    seen = []
    for c in candidates:
        if c not in seen and os.path.exists(f"telemetry_{c}.csv"):
            seen.append(c)

    trials = []
    for suffix in seen:
        df = load_active_window(f"telemetry_{suffix}.csv")
        if df is None:
            continue
        rtts = parse_ping_log(f"ping_{suffix}.log") or parse_ping_log(f"ping_{m_key}.log")
        iperf_gp, s5_recv, iperf_loss, nic_loss, dgrams, lost_dg, nic_sent = parse_iperf_and_nic(suffix, df)
        if max(dgrams, nic_sent) < 44800:
            continue
        avg_rtt = float(np.mean(rtts)) if rtts else 0.0
        pri_cong = int((df['p4_util_pct'] >= 70.0).sum())
        bak_cong = int((df['p5_util_pct'] >= 70.0).sum())
        changes = int(df['route_changes'].max())
        trig_info = extract_aligned_trigger(suffix)
        trials.append({
            'suffix': suffix,
            'iperf_gp': iperf_gp,
            's5_recv': s5_recv,
            'iperf_loss': iperf_loss,
            'nic_loss': nic_loss,
            'dgrams': dgrams if dgrams > 0 else nic_sent,
            'lost_dg': lost_dg,
            'avg_rtt': avg_rtt,
            'pri_cong': pri_cong,
            'bak_cong': bak_cong,
            'changes': changes,
            'trig': trig_info,
            'df': df,
            'rtts': rtts
        })
    return trials


def main():
    modes = [
        ('baseline', '1. Baseline (Static Shortest-Path)'),
        ('thresh_imm_agg', '2a. Threshold-1Tick + Agg (0s, No ML)'),
        ('thresh_agg', '2b. Threshold-2Tick + Agg (0s, No ML)'),
        ('ramp_agg', '2c. Ramp-Heuristic + Agg (0s, No ML)'),
        ('ramp_agg_5s', '2d. Ramp-Heuristic + Agg (5s, No ML)'),
        ('rf_nocooldown', '3. RF-Only (Single-Port Recov, 0s Cool)'),
        ('rf_gbr_nocooldown', '4. RF+GBR (Single-Port Recov, 0s Cool)'),
        ('rf_agg_nocooldown', '5. RF-Only + Agg-Load Recov (0s Cool)'),
        ('rf_gbr_agg_nocooldown', '6. Pure RF+GBR + Agg-Recov (0s Cool)'),
        ('ml_proactive', '7. Pure RF+GBR + Agg-Recov (5s Cool)')
    ]

    print("\n==========================================================================================================================================================")
    print("       TABLE A: 10-WAY VERIFIED ABLATION SUMMARY (ACCEPTED RUNS >= 44,800 PKTS; DECIMAL IPERF GOODPUT [Mbps] & UNCAPPED S5 RATE)                         ")
    print("==========================================================================================================================================================")
    header = f"{'Configuration':<39} | {'Trials':<6} | {'iPerf Sent':<10} | {'iPerf Lost':<10} | {'iPerf Loss':<12} | {'iPerf Goodput':<14} | {'S5 Egress':<9} | {'Avg RTT':<11} | {'Pri/Bak>=70%':<12} | {'Chg'}"
    print(header)
    print("-" * 160)

    trials_by_mode = {}
    for m_key, m_label in modes:
        trials = collect_trials(m_key)
        if not trials:
            continue
        trials_by_mode[m_key] = trials

        gp_arr = np.array([t['iperf_gp'] for t in trials])
        s5_arr = np.array([t['s5_recv'] for t in trials])
        il_arr = np.array([t['iperf_loss'] for t in trials])
        rt_arr = np.array([t['avg_rtt'] for t in trials])
        pc_arr = np.array([t['pri_cong'] for t in trials])
        bc_arr = np.array([t['bak_cong'] for t in trials])
        ch_arr = np.array([t['changes'] for t in trials])
        dg_mean = int(round(np.mean([t['dgrams'] for t in trials])))
        ld_mean = int(round(np.mean([t['lost_dg'] for t in trials])))

        if len(trials) > 1:
            gp_s = f"{gp_arr.mean():.2f}±{gp_arr.std():.2f}M"
            il_s = f"{il_arr.mean():.2f}±{il_arr.std():.2f}%"
            rt_s = f"{rt_arr.mean():.1f}±{rt_arr.std():.1f}ms"
        else:
            gp_s = f"{gp_arr.mean():>8.2f} Mbps"
            il_s = f"{il_arr.mean():>8.2f}%"
            rt_s = f"{rt_arr.mean():>7.2f} ms"

        pb_s = f"{int(round(pc_arr.mean())):>2d}s / {int(round(bc_arr.mean())):>2d}s"
        print(f"{m_label:<39} | n={len(trials):<4d} | {dg_mean:>10d} | {ld_mean:>10d} | {il_s:<12} | {gp_s:<14} | {s5_arr.mean():>6.2f} M | {rt_s:<11} | {pb_s:<12} | {int(round(ch_arr.mean())):>3d}")

    print("==========================================================================================================================================================\n")

    print("====================================================================================================================================================")
    print("       TABLE B: TRAFFIC-ALIGNED TRIGGER AUDIT LOG (ALIGNED TO FLOW 1 START t_bench=0s; FLOW 2 BURST ARRIVES AT t_bench=6.0s)                       ")
    print("====================================================================================================================================================")
    print(f"{'Configuration':<39} | {'Bench Time t_bench':<18} | {'Measured U_1,4':<15} | {'Predicted U_t+2':<15} | {'Slope dR':<10} | {'Dispatch(ms)':<12} | {'Trigger Source'}")
    print("-" * 150)
    for m_key, m_label in modes:
        if m_key == 'baseline' or m_key not in trials_by_mode:
            continue
        trigs = [t['trig'] for t in trials_by_mode[m_key] if t['trig'] is not None]
        if not trigs:
            continue
        tb_arr = np.array([tr['t_bench'] for tr in trigs])
        um_arr = np.array([tr['u_meas'] for tr in trigs])
        up_arr = np.array([tr['u_pred'] for tr in trigs])
        dr_arr = np.array([tr['slope_dr'] for tr in trigs])
        dm_arr = np.array([tr['disp_ms'] for tr in trigs])
        reason = trigs[-1]['reason']

        if len(trigs) > 1:
            tb_s = f"{tb_arr.mean():>5.2f}±{tb_arr.std():.2f} s"
            um_s = f"{um_arr.mean():>6.2f}±{um_arr.std():.2f}%"
            up_s = f"{up_arr.mean():>6.2f}±{up_arr.std():.2f}%"
        else:
            tb_s = f"t={tb_arr[0]:>6.2f} s"
            um_s = f"{um_arr[0]:>11.2f} %"
            up_s = f"{up_arr[0]:>12.2f} %"

        print(f"{m_label:<39} | {tb_s:<18} | {um_s:<15} | {up_s:<15} | {dr_arr.mean():>+6.2f} M  | {dm_arr.mean():>9.3f} ms | {reason}")
    print("====================================================================================================================================================\n")


if __name__ == '__main__':
    main()
PYEOF

run_verified_trial() {
    MODE_ARG=$1
    TITLE=$2
    for ATTEMPT in 1 2 3 4; do
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

# 1. Run 3 trials of Ramp-Rate Heuristic + Aggregate Recovery + 5s Cooldown (No ML)
run_verified_trial "ramp_agg_5s" "Ramp-Heuristic + Agg-Recov + 5s Cooldown (Trial 1)"
run_verified_trial "ramp_agg_5s_run2" "Ramp-Heuristic + Agg-Recov + 5s Cooldown (Trial 2)"
run_verified_trial "ramp_agg_5s_run3" "Ramp-Heuristic + Agg-Recov + 5s Cooldown (Trial 3)"

# 2. Re-run Trial 1 of rf_gbr_agg_nocooldown so Config 6 has n=3 complete trials (>= 44,800 pkts)
run_verified_trial "rf_gbr_agg_nocooldown" "Pure RF+GBR + Agg-Recov + 0s Cooldown (Trial 1)"

sudo chown -R $USER:$USER ~/ccn_cep_sdn
python3 evaluate_and_plot.py
