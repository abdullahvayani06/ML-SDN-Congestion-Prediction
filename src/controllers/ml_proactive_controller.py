#!/usr/bin/env python3
"""
CS-351 CCN Complex Engineering Problem (CEP)
File: ml_proactive_controller.py
Description: Uncapped telemetry logging + exact per-trial trigger attribution across all 10 modes.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import csv
import json
import os
import pickle
import time
import numpy as np
from collections import deque
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub


class MLProactiveRoutingController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(MLProactiveRoutingController, self).__init__(*args, **kwargs)
        self.mode = os.environ.get('SDN_MODE', 'ml_proactive')
        self.base_mode = self.mode.split('_run')[0]
        self.datapaths = {}
        self.prev_s1 = {}
        self.prev_s5 = {}
        self.latest_s5_recv_mbps = 0.0
        self.latest_s5_tx_pkts = 0
        self.tx_history = deque(maxlen=3)
        self.Primary_Cap = 20.0
        self.Backup_Cap = 17.0

        self.reroute_active = False
        self.low_load_ticks = 0
        self.high_load_ticks = 0
        self.route_changes = 0
        self.last_switch_time = 0.0
        self.last_flowmod_ms = 0.0
        self.first_offload_logged = False

        self.cooldown_sec = 5.0 if self.base_mode in ['ml_proactive', 'ramp_agg_5s'] else 0.0
        self.use_agg_recovery = self.base_mode in [
            'thresh_imm_agg', 'thresh_agg', 'ramp_agg', 'ramp_agg_5s',
            'rf_agg_nocooldown', 'rf_gbr_agg_nocooldown', 'ml_proactive'
        ]
        self.use_ml = self.base_mode not in ['thresh_imm_agg', 'thresh_agg', 'ramp_agg', 'ramp_agg_5s']
        self.use_gbr = self.base_mode in ['rf_gbr_nocooldown', 'rf_gbr_agg_nocooldown', 'ml_proactive']

        self.congestion_threshold_pct = 70.0
        self.recovery_threshold_pct = 45.0

        self.model_bundle = None
        if self.use_ml and os.path.exists('sdn_congestion_rf.pkl'):
            with open('sdn_congestion_rf.pkl', 'rb') as f:
                self.model_bundle = pickle.load(f)

        self.csv_file = f'telemetry_{self.mode}.csv'
        self._init_csv()
        self.monitor_thread = hub.spawn(self._monitor_loop)

    def _init_csv(self):
        with open(self.csv_file, 'w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp', 'elapsed_sec', 'offered_mbps', 'p4_tx_mbps', 'p5_tx_mbps',
                'total_tx_mbps', 'recv_delivered_mbps', 'p4_util_pct', 'p5_util_pct',
                'predicted_util_pct', 'sent_pkts', 'recv_pkts', 'p4_drops', 'p5_drops',
                'route_changes', 'inference_ms', 'flowmod_ms', 'active_path'
            ])
        self.start_time = time.time()

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER and dp.id in self.datapaths:
            del self.datapaths[dp.id]

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        ofproto = dp.ofproto
        parser = dp.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                   [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)])]
        dp.send_msg(parser.OFPFlowMod(datapath=dp, priority=0, match=match, instructions=actions))
        self._install_base_paths(dp)

    def add_flow(self, dp, priority, match, actions):
        ofproto = dp.ofproto
        parser = dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(parser.OFPFlowMod(datapath=dp, priority=priority, match=match, instructions=inst))

    def del_flow(self, dp, priority, match):
        ofproto = dp.ofproto
        parser = dp.ofproto_parser
        mod = parser.OFPFlowMod(
            datapath=dp, command=ofproto.OFPFC_DELETE_STRICT,
            out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
            priority=priority, match=match
        )
        dp.send_msg(mod)

    def _install_base_paths(self, dp):
        parser = dp.ofproto_parser
        dpid = dp.id
        if dpid == 1:
            for h_idx in [1, 2, 3]:
                mac_str = f'00:00:00:00:00:0{h_idx}'
                self.add_flow(dp, 10, parser.OFPMatch(eth_dst=mac_str), [parser.OFPActionOutput(h_idx)])
                self.add_flow(dp, 10, parser.OFPMatch(eth_type=0x0806, arp_tpa=f'10.0.0.{h_idx}'), [parser.OFPActionOutput(h_idx)])
            for h_idx in [4, 5, 6]:
                mac_str = f'00:00:00:00:00:0{h_idx}'
                self.add_flow(dp, 10, parser.OFPMatch(eth_dst=mac_str), [parser.OFPActionOutput(4)])
                self.add_flow(dp, 10, parser.OFPMatch(eth_type=0x0806, arp_tpa=f'10.0.0.{h_idx}'), [parser.OFPActionOutput(4)])
        elif dpid in [2, 3, 4]:
            self.add_flow(dp, 10, parser.OFPMatch(in_port=1), [parser.OFPActionOutput(2)])
            self.add_flow(dp, 10, parser.OFPMatch(in_port=2), [parser.OFPActionOutput(1)])
        elif dpid == 5:
            for h_idx in [4, 5, 6]:
                port_out = h_idx - 3
                mac_str = f'00:00:00:00:00:0{h_idx}'
                self.add_flow(dp, 10, parser.OFPMatch(eth_dst=mac_str), [parser.OFPActionOutput(port_out)])
                self.add_flow(dp, 10, parser.OFPMatch(eth_type=0x0806, arp_tpa=f'10.0.0.{h_idx}'), [parser.OFPActionOutput(port_out)])
            for h_idx in [1, 2, 3]:
                mac_str = f'00:00:00:00:00:0{h_idx}'
                self.add_flow(dp, 10, parser.OFPMatch(eth_dst=mac_str), [parser.OFPActionOutput(4)])
                self.add_flow(dp, 10, parser.OFPMatch(eth_type=0x0806, arp_tpa=f'10.0.0.{h_idx}'), [parser.OFPActionOutput(4)])

    def _trigger_proactive_reroute(self, enable_backup=True, audit_info=None):
        if 1 not in self.datapaths or 5 not in self.datapaths:
            return
        t_fm_start = time.perf_counter()
        dp1, dp5 = self.datapaths[1], self.datapaths[5]
        p1, p5 = dp1.ofproto_parser, dp5.ofproto_parser
        pairs = [('00:00:00:00:00:05', '00:00:00:00:00:02')]

        if enable_backup and not self.reroute_active:
            for dst_mac, rev_mac in pairs:
                self.add_flow(dp1, 20, p1.OFPMatch(eth_dst=dst_mac), [p1.OFPActionOutput(5)])
                self.add_flow(dp5, 20, p5.OFPMatch(eth_dst=rev_mac), [p5.OFPActionOutput(5)])
            self.reroute_active = True
            self.low_load_ticks = 0
            self.route_changes += 1
            self.last_switch_time = time.time()
            self.last_flowmod_ms = (time.perf_counter() - t_fm_start) * 1000.0

            if not self.first_offload_logged and audit_info is not None:
                self.first_offload_logged = True
                audit_info['dispatch_ms'] = round(self.last_flowmod_ms, 3)
                with open(f"trigger_audit_{self.mode}.json", "w") as af:
                    json.dump(audit_info, af)

        elif not enable_backup and self.reroute_active:
            for dst_mac, rev_mac in pairs:
                self.del_flow(dp1, 20, p1.OFPMatch(eth_dst=dst_mac))
                self.del_flow(dp5, 20, p5.OFPMatch(eth_dst=rev_mac))
            self.reroute_active = False
            self.low_load_ticks = 0
            self.high_load_ticks = 0
            self.route_changes += 1
            self.last_switch_time = time.time()
            self.last_flowmod_ms = (time.perf_counter() - t_fm_start) * 1000.0

    def _monitor_loop(self):
        while True:
            for dpid in [5, 1]:
                if dpid in self.datapaths:
                    dp = self.datapaths[dpid]
                    dp.send_msg(dp.ofproto_parser.OFPPortStatsRequest(dp, 0, dp.ofproto.OFPP_ANY))
            hub.sleep(1.0)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        now = time.time()
        dpid = ev.msg.datapath.id
        stats_map = {s.port_no: s for s in ev.msg.body if s.port_no <= 10}

        if dpid == 5:
            if all(p in stats_map for p in [1, 2, 3]):
                cur_bytes = sum(stats_map[p].tx_bytes for p in [1, 2, 3])
                cur_pkts = sum(stats_map[p].tx_packets for p in [1, 2, 3])
                if self.prev_s5:
                    dt = max(now - self.prev_s5['time'], 0.5)
                    self.latest_s5_recv_mbps = max(0.0, ((cur_bytes - self.prev_s5['bytes']) * 8.0) / (dt * 1e6))
                    self.latest_s5_tx_pkts = max(0, cur_pkts - self.prev_s5['pkts'])
                self.prev_s5 = {'time': now, 'bytes': cur_bytes, 'pkts': cur_pkts}
            return

        if dpid != 1 or not all(p in stats_map for p in [1, 2, 3, 4, 5]):
            return

        elapsed = round(now - self.start_time, 2)
        s4, s5 = stats_map[4], stats_map[5]
        in_bytes = sum(stats_map[p].rx_bytes for p in [1, 2, 3])
        in_pkts = sum(stats_map[p].rx_packets for p in [1, 2, 3])

        if self.prev_s1:
            dt = max(now - self.prev_s1['time'], 0.5)
            offered_mbps = max(0.0, ((in_bytes - self.prev_s1['in_bytes']) * 8.0) / (dt * 1e6))
            sent_pkts = max(0, in_pkts - self.prev_s1['in_pkts'])
            p4_tx = max(0.0, ((s4.tx_bytes - self.prev_s1[4][0]) * 8.0) / (dt * 1e6))
            p4_rx = max(0.0, ((s4.rx_bytes - self.prev_s1[4][1]) * 8.0) / (dt * 1e6))
            p5_tx = max(0.0, ((s5.tx_bytes - self.prev_s1[5][0]) * 8.0) / (dt * 1e6))
            p4_drops = max(0, s4.tx_dropped - self.prev_s1[4][2])
            p5_drops = max(0, s5.tx_dropped - self.prev_s1[5][2])

            p4_util = min(100.0, (p4_tx / self.Primary_Cap) * 100.0)
            p5_util = min(100.0, (p5_tx / self.Backup_Cap) * 100.0)
            total_tx = p4_tx + p5_tx
            recv_mbps = self.latest_s5_recv_mbps
            recv_pkts = self.latest_s5_tx_pkts
            equivalent_primary_util = (total_tx / self.Primary_Cap) * 100.0

            prev_load = self.tx_history[-1] if len(self.tx_history) > 0 else p4_tx
            self.tx_history.append(p4_tx)
            delta_mbps = p4_tx - prev_load
            roll_mean = float(np.mean(self.tx_history))
            roll_std = float(np.std(self.tx_history))

            t_inf_start = time.perf_counter()
            if self.use_ml and self.model_bundle is not None:
                X_vec = np.array([[p4_rx, p4_tx, p4_util, delta_mbps, roll_mean, roll_std]])
                pred_cls = int(self.model_bundle['rf_cls'].predict(X_vec)[0])
                if self.use_gbr:
                    pred_util = float(self.model_bundle['gbr_reg'].predict(X_vec)[0])
                else:
                    pred_util = p4_util
                inf_ms = (time.perf_counter() - t_inf_start) * 1000.0
            else:
                pred_cls = 0
                pred_util = p4_util
                inf_ms = 0.0

            if (now - self.last_switch_time) >= self.cooldown_sec:
                if not self.reroute_active:
                    audit_payload = {
                        'mode': self.mode,
                        'elapsed_sec': elapsed,
                        'measured_p4_util_pct': round(p4_util, 2),
                        'predicted_util_pct': round(pred_util, 2),
                        'rf_pred_cls': pred_cls,
                        'delta_mbps': round(delta_mbps, 2),
                        'inference_ms': round(inf_ms, 3)
                    }
                    if self.base_mode == 'thresh_imm_agg':
                        if p4_util >= self.congestion_threshold_pct:
                            audit_payload['trigger_reason'] = 'Measured_U>=70%_1Tick'
                            self._trigger_proactive_reroute(enable_backup=True, audit_info=audit_payload)
                    elif self.base_mode == 'thresh_agg':
                        if p4_util >= self.congestion_threshold_pct:
                            self.high_load_ticks += 1
                            if self.high_load_ticks >= 2:
                                audit_payload['trigger_reason'] = 'Measured_U>=70%_2Ticks'
                                self._trigger_proactive_reroute(enable_backup=True, audit_info=audit_payload)
                        else:
                            self.high_load_ticks = 0
                    elif self.base_mode in ['ramp_agg', 'ramp_agg_5s']:
                        if (p4_util >= 46.0 and delta_mbps >= 2.2) or p4_util >= self.congestion_threshold_pct:
                            audit_payload['trigger_reason'] = 'Ramp_Pre70%' if p4_util < 70.0 else 'Ramp_Post70%'
                            self._trigger_proactive_reroute(enable_backup=True, audit_info=audit_payload)
                    elif not self.use_gbr:
                        if (p4_util >= 45.0 and pred_cls == 1) or p4_util >= self.congestion_threshold_pct:
                            audit_payload['trigger_reason'] = 'RF_Cls=1_Pre70%' if p4_util < 70.0 else 'RF_Post70%'
                            self._trigger_proactive_reroute(enable_backup=True, audit_info=audit_payload)
                    else:
                        if (p4_util >= 45.0) and (pred_util >= self.congestion_threshold_pct or pred_cls == 1):
                            if p4_util < 70.0 and pred_util >= 70.0:
                                reason_tag = 'GBR_Pre70%(U_hat>=70%)'
                            elif p4_util < 70.0 and pred_cls == 1:
                                reason_tag = 'RF_Pre70%(y_cls=1)'
                            else:
                                reason_tag = 'ML_Post70%(U>=70%)'
                            audit_payload['trigger_reason'] = reason_tag
                            self._trigger_proactive_reroute(enable_backup=True, audit_info=audit_payload)
                else:
                    if self.use_agg_recovery:
                        if total_tx > 1.0 and equivalent_primary_util < self.recovery_threshold_pct:
                            self.low_load_ticks += 1
                            if self.low_load_ticks >= 2:
                                self._trigger_proactive_reroute(enable_backup=False)
                        else:
                            self.low_load_ticks = 0
                    else:
                        if p4_util < 68.0:
                            self._trigger_proactive_reroute(enable_backup=False)

            active_label = 'Split_Primary+Backup' if self.reroute_active else 'Primary_Only'
            with open(self.csv_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    round(now, 2), elapsed, round(offered_mbps, 3), round(p4_tx, 3), round(p5_tx, 3),
                    round(total_tx, 3), round(recv_mbps, 3), round(p4_util, 2), round(p5_util, 2),
                    round(pred_util, 2), sent_pkts, recv_pkts, p4_drops, p5_drops,
                    self.route_changes, round(inf_ms, 3), round(self.last_flowmod_ms, 3), active_label
                ])

        self.prev_s1 = {
            'time': now,
            'in_bytes': in_bytes,
            'in_pkts': in_pkts,
            4: (s4.tx_bytes, s4.rx_bytes, s4.tx_dropped),
            5: (s5.tx_bytes, s5.rx_bytes, s5.tx_dropped)
        }
