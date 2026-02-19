# Preprocessing

## Zeek Flow Extraction

The raw DARPA 2000 data is distributed as PCAP files. These are converted into flow-level records using Zeek.

Zeek extracts connection-level metadata such as:
- Timestamps
- Source and destination IP addresses
- Source and destination ports
- Protocol and service
- Packet and byte counts
- Connection state

Flows are defined using Zeek’s standard 5-tuple aggregation:
```css
(src IP, dst IP, src port, dst port, protocol)
```

The resulting `conn.log` files are converted to CSV format for downstream processing.

## Labeling Process

To construct a labeled intrusion detection dataset, the Zeek-derived flow records in `all_flows.csv` are relabeled using the per-phase attack flow files `phase*_flows.csv`.

Labeling is performed using **exact feature-level matching**:
- Each flow is uniquely identified by computing a hash over **all flow attributes except `flow_id`**.
- The same hashing procedure is applied to the per-phase flow CSV files.
- A flow in the global dataset is assigned a phase label if its hash matches a flow in a corresponding phase file.

Each flow is assigned:
- `phase = 0` for benign traffic (no match found)
- `phase ∈ {1..5}` for attack flows corresponding to the multi-step attack stages

Because the per-phase flow files are generated from the same Zeek logs as the global dataset, this method ensures:
- Deterministic and reproducible labeling
- Exact semantic flow matching
- No dependence on timestamp tolerances
- No ambiguity from partial tuple alignment

## Output File

The labeling process produces a single CSV file containing all flows with an additional `phase` column:
```ini
phase = 0      → benign traffic  
phase = 1–5    → attack stage
```

This file serves as the basis for downstream machine learning models and DeepProbLog neuro-symbolic experiments.

## Processing Steps

This section describes how to reproduce the preprocessing pipeline used to convert the raw DARPA 2000 PCAP files into Zeek flow CSVs and labeled datasets.

1. **Install Zeek**

Installation instructions can be found in the official documentation: https://docs.zeek.org/en/v8.0.4/install.html

**For Ubuntu 24.04:**

```bash
# Add repository and install zeek
echo 'deb http://download.opensuse.org/repositories/security:/zeek/xUbuntu_24.04/ /' | sudo tee /etc/apt/sources.list.d/security:zeek.list

curl -fsSL https://download.opensuse.org/repositories/security:zeek/xUbuntu_24.04/Release.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/security_zeek.gpg > /dev/null

sudo apt update
sudo apt install zeek

# Add Zeek to PATH
echo 'export PATH="/opt/zeek/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

2. **Extract Dataset Archives**

Unzip the DARPA 2000 archives:

```bash
data/raw/darpa2000/<scenario_network>.tar.gz
```

After extraction, each directory will contain six PCAP files.

3. **Convert PCAP Files to Zeek Logs**

From the project root, run the Zeek processing script for each scenario and network:

```bash
uv run -m src.flow_processing.pcap_to_zeek --dataset darpa2000 --scenario_network s1_inside
```

4. **Convert Zeek `conn.log` Files to CSV**

From the project root, run the Zeek-to-CSV conversion script for each scenario and network:
```bash
uv run -m src.flow_processing.zeek_conn_to_csv --dataset darpa2000 --scenario_network s1_inside --overwrite
```

5. **Label Flows With Attack Phases**

From the project root, run the labeling script for each scenario and network:
```bash
uv run -m src.flow_processing.label_flows --dataset darpa2000 --scenario_network s1_inside --overwrite
```

The labeled dataset is written to:
```bash
data/interim/darpa2000/s1_inside/flows_labeled/all_flows_labeled.csv
```