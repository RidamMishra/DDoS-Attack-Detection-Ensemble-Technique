import switch
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from datetime import datetime

class CollectTrainingStatsApp(switch.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(CollectTrainingStatsApp, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self.monitor)

        with open("FlowStatsfile.csv", "w") as file0:
            file0.write('timestamp,datapath_id,ip_src,ip_dst,pktcount,bytecount,dur,protocol,'
                        'tx_bytes,rx_bytes,tx_kbps,rx_kbps,tot_kbps,label\n')

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath

        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def monitor(self):
        while True:
            for dp in self.datapaths.values():
                self.request_stats(dp)
            hub.sleep(10)

    def request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        with open("FlowStatsfile.csv", "a+") as file0:
            body = ev.msg.body
            for stat in sorted([flow for flow in body if flow.priority == 1], key=lambda flow:
                               (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):

                ip_src = stat.match['ipv4_src']
                ip_dst = stat.match['ipv4_dst']
                ip_proto = stat.match['ip_proto']

                pktcount = stat.packet_count
                bytecount = stat.byte_count
                dur = stat.duration_sec + (stat.duration_nsec / 1e9)

                
                tx_bytes = bytecount
                rx_bytes = bytecount

                try:
                    tx_kbps = (tx_bytes * 8) / (dur * 1024)
                    rx_kbps = (rx_bytes * 8) / (dur * 1024)
                except ZeroDivisionError:
                    tx_kbps = 0
                    rx_kbps = 0

                tot_kbps = tx_kbps + rx_kbps

                file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    timestamp, ev.msg.datapath.id, ip_src, ip_dst, pktcount, bytecount, dur, ip_proto,
                    tx_bytes, rx_bytes, tx_kbps, rx_kbps, tot_kbps, 1  # Assuming '1' as label
                ))
