
from dataclasses import dataclass

@dataclass
class FeatureSpec:
    full_nn_features: list[str]
    reduced_nn_features: list[str]
    aug_nn_features: list[str]
    logic_features: list[str]
    metadata_features: list[str]


FEATURES = FeatureSpec(

    full_nn_features = [
        # "src_ip",
        # "dst_ip",
        "sport",
        "dport",
        "proto",
        "service",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "conn_state",
        "local_orig",
        "local_resp",
        "missed_bytes",
        # "history",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
        # "tunnel_parents",
        "ip_proto",
    ],

    # Exclude ports
    reduced_nn_features = [
        # "src_ip",
        # "dst_ip",
        # "sport",
        # "dport",
        "proto",
        "service",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "conn_state",
        "local_orig",
        "local_resp",
        "missed_bytes",
        # "history",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
        # "tunnel_parents",
        "ip_proto",
    ],

    # Add augmented features 
    aug_nn_features = [
        # "src_ip",
        # "dst_ip",
        # "sport",
        # "dport",
        "proto",
        "service",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "conn_state",
        "local_orig",
        "local_resp",
        "missed_bytes",
        # "history",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
        # "tunnel_parents",
        "ip_proto",
        
        "unique_sources",
        "fanin_rate",
        "unique_targets",
        "fanout_rate",
        "dst_ratio",
        "unique_ports",
        "connection_count",
    ],

    logic_features=[
        # Zeek features
        "src_ip",
        "dst_ip",
        "sport",
        "dport",
        "proto",
        "service",
        "local_orig",
        "local_resp",
        # Augmented features
        "unique_sources",
        "fanin_rate",
        "unique_targets",
        "fanout_rate",
        "dst_ratio",
        "unique_ports",
        "connection_count",
    ],

    metadata_features=[
        "orig_index",
        "start_time",
    ],
)