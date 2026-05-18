
from dataclasses import dataclass

@dataclass
class FeatureSpec:
    flow_features: list[str]
    behavioral_features: list[str]
    port_features: list[str]
    rule_features: list[str]
    metadata_features: list[str]


FEATURES = FeatureSpec(

    flow_features = [
        "proto",
        "service",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "conn_state",
        "local_orig",
        "local_resp",
        "missed_bytes",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
        "ip_proto",
    ],

    behavioral_features = [
        "connections_per_src_60s",
        "unique_targets_60s",
        "unique_dports_60s",
        "syn_failure_ratio_60s",
        "reject_ratio_60s",
        "reset_ratio_60s",
        "connections_per_dst_60s",
        "unique_sources_per_dst_60s",
    ],

    port_features=[
        "sport",
        "dport",
    ],

    rule_features=[
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

FLOW_ONLY_FEATURES = (
    FEATURES.flow_features
)

BASE_FEATURES = (
    FEATURES.flow_features
    + FEATURES.behavioral_features
)

PORT_AWARE_FEATURES = (
    BASE_FEATURES
    + FEATURES.port_features
)

DPL_FEATURES = (
    FEATURES.rule_features
    + FEATURES.behavioral_features
    + FEATURES.metadata_features
)