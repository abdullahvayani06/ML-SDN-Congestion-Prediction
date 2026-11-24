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
