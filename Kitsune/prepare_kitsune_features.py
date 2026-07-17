import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

from src.flow_processing.FlowMatcher import FlowMatcher


class PrepareKitsuneFeatures:

    def __init__(self,file_path, labels_path):
        self.path = file_path
        self.label_path = labels_path

        self.__prep__()

        self.matcher = FlowMatcher(self.label_path)
        self.label_packets()


    def _get_tshark_path(self):
        system_path = os.environ['PATH']
        for path in system_path.split(os.pathsep):
            filename = os.path.join(path, 'tshark')
            if os.path.isfile(filename):
                print(f"Found tshark at: {filename}")
                return filename
        
        print("tshark not found in PATH. Please install Wireshark and add tshark to your PATH.")
        return ''


    def __prep__(self):
        ### Find file: ###
        if not os.path.isfile(self.path):  # file does not exist
            print("File: " + self.path + " does not exist")
            raise Exception()

        ### check file type ###
        type = self.path.split('.')[-1]

        self._tshark = self._get_tshark_path()
        ##If file is TSV (pre-parsed by wireshark script)
        if type == "tsv":
            self.parse_type = "tsv"
            print(f"File: {self.path} is a TSV file. Ready for processing.")

        elif type == "pcap":
            # Try parsing via tshark dll of wireshark (faster)
            if os.path.isfile(self._tshark):
                self.pcap2tsv_with_tshark()  # creates local tsv file
                self.path += ".tsv"
                self.parse_type = "tsv"
        else:
            print("File: " + self.path + " is not a tsv orpcap file")
            raise Exception()


    def pcap2tsv_with_tshark(self):
        print('Parsing with tshark...')
        fields = "-e frame.number -e frame.time_epoch -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e icmp.type -e icmp.code -e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 -e arp.dst.hw_mac -e arp.dst.proto_ipv4 -e ipv6.src -e ipv6.dst"
        cmd =  '"' + self._tshark + '" -r '+ self.path +' -T fields '+ fields +' -E header=y -E occurrence=f > '+self.path+".tsv"
        subprocess.call(cmd,shell=True)
        print("tshark parsing complete. File saved as: "+self.path +".tsv")


    def label_packets(self):

        output = "packet_labels.csv"

        packets = pd.read_csv(tsv_path, sep="\t", usecols=["frame.time_epoch"])

        print(packets["frame.time_epoch"].min())
        print(packets["frame.time_epoch"].max())

        matched = 0
        unknown = 0

        first_chunk = True

        for packets in pd.read_csv(
            self.path,
            sep="\t",
            chunksize=100000,
        ):

            labels = []
            hashes = []

            for _, pkt in packets.iterrows():

                if matched + unknown == 0:
                    print(pkt[[
                        "ip.src",
                        "ip.dst",
                        "tcp.srcport",
                        "tcp.dstport",
                        "udp.srcport",
                        "udp.dstport",
                        "frame.time_epoch"
                    ]])

                try:

                    if pd.notna(pkt["tcp.srcport"]):

                        sport = int(pkt["tcp.srcport"])
                        dport = int(pkt["tcp.dstport"])

                    elif pd.notna(pkt["udp.srcport"]):

                        sport = int(pkt["udp.srcport"])
                        dport = int(pkt["udp.dstport"])

                    else:
                        labels.append("Unknown")
                        hashes.append(None)
                        unknown += 1
                        continue

                    match = self.matcher.match_packet(
                        pkt["ip.src"],
                        pkt["ip.dst"],
                        sport,
                        dport,
                        float(pkt["frame.time_epoch"]),
                    )

                    if match is None:

                        labels.append("Unknown")
                        hashes.append(None)
                        unknown += 1

                    else:

                        labels.append(match["label"])
                        hashes.append(match["flow_hash"])
                        matched += 1

                except Exception:

                    labels.append("Unknown")
                    hashes.append(None)
                    unknown += 1

            results = pd.DataFrame(
                {
                    "frame.number": packets["frame.number"],
                    "frame.time_epoch": packets["frame.time_epoch"],
                    "label": labels,
                    "flow_hash": hashes,
                }
            )   

            results.to_csv(
                output,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )

            first_chunk = False

            print(
                f"Processed {matched + unknown:,} packets",
                end="\r",
            )

        print()
        print(f"Matched : {matched:,}")
        print(f"Unknown : {unknown:,}")
        print(f"Rate    : {matched/(matched+unknown):.2%}")
        

    # def trim_to_simulation_period():
    #     pass


if __name__ == "__main__":
    # uv run python -m Kitsune.prepare_kitsune_features

    # pcap_path = "data/raw/aitv2/santos_merged_pcaps/merged_sorted.pcap"
    tsv_path = "data/raw/aitv2/santos_merged_pcaps/merged_sorted.pcap.tsv"
    labels_path = "data/raw/aitv2/santos_netflows/all_netflows.csv"
    
    PrepareKitsuneFeatures(tsv_path, labels_path)