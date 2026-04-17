
from dataclasses import dataclass

@dataclass
class FeatureSpec:
    full_nn_features: list[str]
    reduced_nn_features: list[str]
    behavioral_nn_features: list[str]
    logic_features: list[str]
    metadata_features: list[str]


FEATURES = FeatureSpec(

    # Full feature set excluding IP addresses (to prevent memorization)
    full_nn_features = [
        "sport",
        "dport",
        "proto",
        "service",
        "local_orig",
        "local_resp",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "history",
        "conn_state",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
    ],

    # Removes ports to reduce identifier bias
    reduced_nn_features = [
        "proto",
        "service",
        "local_orig",
        "local_resp",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "history",
        "conn_state",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
    ],

    # Removes ports and protocol-level semantics
    behavioral_nn_features =[
        "local_orig",
        "local_resp",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "history",
        "conn_state",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
    ],

    logic_features=[
        # Default flow features
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