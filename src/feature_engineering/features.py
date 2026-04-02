
from dataclasses import dataclass

@dataclass
class FeatureSpec:
    all_nn_features: list[str]
    sub_nn_features: list[str]
    logic_features: list[str]
    metadata_features: list[str]


FEATURES = FeatureSpec(

    all_nn_features = [
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

    sub_nn_features=[
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
        "src_ip",
        "dst_ip",
        "sport",
        "dport",
        "proto",
        "service",
        "local_orig",
        "local_resp",
    ],

    metadata_features=[
        "orig_index",
        "start_time",
    ],

)