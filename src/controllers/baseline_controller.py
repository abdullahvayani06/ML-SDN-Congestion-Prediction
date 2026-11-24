#!/usr/bin/env python3
"""
CS-351 CCN Complex Engineering Problem (CEP)
File: baseline_controller.py
Description: Baseline Static Shortest-Path Controller polling both Ingress (s1)
             and Egress/Receiver (s5) switches to measure true receiver delivery and packet loss.
"""

import csv
import os
import time
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub


class BaselineShortestPathController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(BaselineShortestPathController, self).__init__(*args, **kwargs)
        self.mode = os.environ.get('SDN_MODE', 'baseline')
        self.datapaths = {}
        self.prev_s1 = {}
        self.prev_s5 = {}
        self.latest_s5_recv_mbps = 0.0
        self.latest_s5_tx_pkts = 0
        self.Primary_Cap = 20.0
        self.Backup_Cap = 17.0
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
        self._install_static_shortest_path(dp)

    def add_flow(self, dp, priority, match, actions):
        ofproto = dp.ofproto
        parser = dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(parser.OFPFlowMod(datapath=dp, priority=priority, match=match, instructions=inst))

    def _install_static_shortest_path(self, dp):
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
            p5_tx = max(0.0, ((s5.tx_bytes - self.prev_s1[5][0]) * 8.0) / (dt * 1e6))
            p4_drops = max(0, s4.tx_dropped - self.prev_s1[4][1])
            p5_drops = max(0, s5.tx_dropped - self.prev_s1[5][1])

            p4_util = min(100.0, (p4_tx / self.Primary_Cap) * 100.0)
            p5_util = min(100.0, (p5_tx / self.Backup_Cap) * 100.0)
            total_tx = p4_tx + p5_tx
            recv_mbps = self.latest_s5_recv_mbps if self.latest_s5_recv_mbps > 0 else total_tx
            recv_pkts = self.latest_s5_tx_pkts

            with open(self.csv_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    round(now, 2), elapsed, round(offered_mbps, 3), round(p4_tx, 3), round(p5_tx, 3),
                    round(total_tx, 3), round(recv_mbps, 3), round(p4_util, 2), round(p5_util, 2),
                    round(p4_util, 2), sent_pkts, recv_pkts, p4_drops, p5_drops,
                    0, 0.0, 0.0, 'Primary_Only'
                ])

        self.prev_s1 = {
            'time': now,
            'in_bytes': in_bytes,
            'in_pkts': in_pkts,
            4: (s4.tx_bytes, s4.tx_dropped),
            5: (s5.tx_bytes, s5.tx_dropped)
        }
