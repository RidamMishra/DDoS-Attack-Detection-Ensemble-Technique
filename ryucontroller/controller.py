from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end - start))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):

        timestamp = datetime.now().timestamp()

        with open("PredictFlowStatsfile.csv", "w") as file0:
            file0.write('timestamp,datapath_id,ip_src,ip_dst,pktcount,bytecount,dur,protocol,'
                        'tx_bytes,rx_bytes,tx_kbps,rx_kbps,tot_kbps\n')
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

                file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    timestamp, ev.msg.datapath.id, ip_src, ip_dst, pktcount, bytecount, dur, ip_proto,
                    tx_bytes, rx_bytes, tx_kbps, rx_kbps, tot_kbps
                ))

    def flow_training(self):
        self.logger.info("Flow Training ...")

        flow_dataset = pd.read_csv('FlowStatsfile.csv')

        flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
        flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
        flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')

        X_flow = flow_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')

        y_flow = flow_dataset.iloc[:, -1].values

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        self.flow_model = classifier.fit(X_flow_train, y_flow_train)

        y_flow_pred = self.flow_model.predict(X_flow_test)

        self.logger.info("------------------------------------------------------------------------------")

        self.logger.info("Confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        self.logger.info(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)
        self.logger.info("Success accuracy = {0:.2f} %".format(acc * 100))
        fail = 1.0 - acc
        self.logger.info("Fail accuracy = {0:.2f} %".format(fail * 100))
        self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')

            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')

            X_predict_flow = predict_flow_dataset.iloc[:, :].values
            X_predict_flow = X_predict_flow.astype('float64')

            y_flow_pred = self.flow_model.predict(X_predict_flow)

            legitimate_traffic = 0
            ddos_traffic = 0

            for i in y_flow_pred:
                if i == 0:
                    legitimate_traffic += 1
                else:
                    ddos_traffic += 1
                    victim = int(predict_flow_dataset.iloc[i, 5]) % 20

            self.logger.info("------------------------------------------------------------------------------")
            if (legitimate_traffic / len(y_flow_pred) * 100) > 80:
                self.logger.info("Legitimate traffic ...")
            else:
                self.logger.info("DDoS traffic ...")
                self.logger.info("Victim is host: h{}".format(victim))
            self.logger.info("------------------------------------------------------------------------------")

            with open("PredictFlowStatsfile.csv", "w") as file0:
                file0.write('timestamp,datapath_id,ip_src,ip_dst,pktcount,bytecount,dur,protocol,'
                            'tx_bytes,rx_bytes,tx_kbps,rx_kbps,tot_kbps\n')

        except Exception as e:
            self.logger.error("Flow prediction error: {}".format(e))
