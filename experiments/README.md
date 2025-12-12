# ML Experiments

## Experimental Setup

There are three variables that we want to experiment with:
- Different models
    - Decision Tree
    - Random Forest
    - SVM
    - MLP
- Different feature sets (based on flow features extracted from zeek conn logs)
    - All features excluding:
        - IPs and Time
        - IPs, Time, Ports
        - IPs, Time, Ports, Service
    - Note: the goal is to remove features that might cause bias, and are not explainatory 
- Different training data splits (7 variations)
    1. Scenario 1:
        - inside data only (60/40 split)
        - train on inside, test on DMZ
    2. Scenario 2:
        - inside data only (60/40 split)
        - train on inside, test on DMZ
    3. Train on scenario 1, test on scenario 2:
        - inside data only
        - dmz data only
        - inside and dmz data 

In the beginning, only binary classification: attack vs benign

<!-- ## Features

Run the following script to create ML-ready feature matrices from labeled DARPA 2000 flow CSVs:
```bash
uv run python experiments/features/build_features.py
```

This script:
- loads labeled flow data
- selects useful flow-level features
- encodes categorical fields (proto, service, conn_state)
- encodes or drops IP addresses (configurable)
- normalizes numeric features
- produces X (features) and y (labels)
- saves them to disk for downstream ML experiments

Usage:
```bash
uv run python experiments/features/build_features.py \
    --input data/DARPA_2000/inside/inside_labeled_flows_all.csv \
    --output_dir experiments/features/output \
    --ip-encoding none
``` -->

### All Features

```bash
[
    "flow_id",
    "start_time",
    "end_time",
    "duration",
    "src_ip",
    "sport",
    "dst_ip",
    "dport",
    "proto",
    "service",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "tunnel_parents",
    "ip_proto" 
]
```