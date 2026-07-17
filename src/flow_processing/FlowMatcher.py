from collections import defaultdict
import pandas as pd


class FlowMatcher:

    def __init__(self, labels_file, tolerance=0.0):
        """
        Parameters
        ----------
        labels_file : str
            Path to all_netflows.csv

        tolerance : float
            Seconds allowed outside the flow interval.
        """

        self.tolerance = tolerance
        self.flow_index = defaultdict(list)

        self._build_index(labels_file)
    

    @staticmethod
    def canonical_key(src_ip, sport, dst_ip, dport):

        left = (src_ip, int(sport))
        right = (dst_ip, int(dport))

        if left <= right:
            return (
                src_ip,
                int(sport),
                dst_ip,
                int(dport)
            )

        return (
            dst_ip,
            int(dport),
            src_ip,
            int(sport)
        )


    def _build_index(self, labels_file):
        print(f"Building flow index from {labels_file}...")

        df = pd.read_csv(labels_file)

        pcap_start = 1642080045.726471
        pcap_end = 1642085572.962172

        overlap = df[
            (df["end_time_match"] >= pcap_start) &
            (df["start_time_match"] <= pcap_end)
        ]

        print(len(overlap))
        print(overlap.head())

        print(df["start_time_match"].min())
        print(df["start_time_match"].max())

        print(df["end_time_match"].min())
        print(df["end_time_match"].max())

        for _, flow in df.iterrows():

            key = self.canonical_key(
                flow["src_ip"],
                int(flow["sport"]),
                flow["dst_ip"],
                int(flow["dport"])
            )

            self.flow_index[key].append({
                "start": float(flow["start_time_match"]),
                "end": float(flow["end_time_match"]),
                "label": flow["label"],
                "flow_hash": flow["flow_hash"]
            })

        # Sort candidate flows by start time
        for key in self.flow_index:
            self.flow_index[key].sort(key=lambda x: x["start"])
        
        print(f"Flow index built with {len(self.flow_index)} unique flow keys.")

        first_key = next(iter(self.flow_index))

        print("Example indexed key:")
        print(first_key)

        print("Example flow:")
        print(self.flow_index[first_key][0])
    

    def match_packet(
        self,
        src_ip,
        dst_ip,
        sport,
        dport,
        timestamp
    ):

        key = self.canonical_key(
            src_ip, 
            sport, 
            dst_ip, 
            dport
        )

        candidates = self.flow_index.get(key, [])

        for flow in candidates:

            if (
                flow["start"] - self.tolerance
                <= timestamp
                <= flow["end"] + self.tolerance
            ):

                return {
                    "label": flow["label"],
                    "flow_hash": flow["flow_hash"],
                    "start": flow["start"],
                    "end": flow["end"]
                }

        return None