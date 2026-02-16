# DARPA 2000

## Multi-Step Attack Scenario

The DARPA 2000 intrusion detection evaluation dataset contains a multi-step cyber attack executed in a simulated U.S. Air Force Base network. The adversary performs **reconnaissance**, **exploitation**, **privilege escalation**, **malware installation**, and ultimately launches a **distributed denial-of-service (DDoS) attack**.

The attack unfolds in the following phases:

| Phase | Description                                                                    |
| ----- | ------------------------------------------------------------------------------ |
| **1** | IP sweep across multiple subnets to identify active hosts                      |
| **2** | Probe hosts to detect the vulnerable **sadmind** remote administration service |
| **3** | Exploit *sadmind* to gain **root access** and create a malicious user          |
| **4** | Install and configure the **mstream** DDoS master and server components        |
| **5** | Launch a coordinated **mstream DDoS attack**                                   |


Each phase represents a distinct stage in the attacker’s kill chain and is reflected in the ground-truth attack traces provided with the dataset.

## Dataset Structure

The DARPA 2000 dataset is organized into two monitored network segments:
- **inside** – internal Air Force network traffic
- **dmz** – externally facing DMZ traffic

Each segment contains:
- A **full packet capture (PCAP)** with all traffic (benign + malicious)
- **Per-phase PCAPs** containing only traffic relevant to a specific attack phase
- **IDMEF XML label files** describing attack sessions

## Preprocessing

### Zeek Flow Extraction

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

### Labeling Process

To construct a labeled intrusion detection dataset, the Zeek-derived flow records in `all_flows.csv` are relabeled using the per-phase attack flow files.

Labeling is performed using **exact feature-level matching**, rather than tuple-based or timestamp-based alignment:
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

### Output File

The labeling process produces a single CSV file containing all flows with an additional `phase` column:
```ini
phase = 0      → benign traffic  
phase = 1–5    → attack stage
```

The labeled dataset is saved to:
```bash
data/DARPA/<scenario>/<network>/labeled_flows/all_flows_labeled.csv
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
data/DARPA/<scenario>/<network>/all_flows.tar.zip
data/DARPA/<scenario>/<network>/per_phase_flows.zip
```

After extraction, each directory will contain:
- Raw PCAP files
- Per-phase PCAPs

3. **Convert PCAP Files to Zeek Logs**

**Process the full PCAP (all flows)**

Navigate to:
```bash
cd data/DARPA/<scenario>/<network>/all_flows/
```

Run Zeek:
```bash
zeek -Cr LLS_DDOS_1.0-<scenario>-<network>.dump
```

Keep only `conn.log` and remove other logs if desired.

**Process Per-Phase PCAPs**

Navigate to:
```bash
cd data/DARPA/<scenario>/<network>/per_phase_flows/
```

Process each phase:
```bash
for f in phase-*-tcpdump-out-dump; do
    phase=$(echo $f | grep -oP 'phase-\K\d+')
    echo "Processing phase $phase ..."
    zeek -Cr "$f"
    mv conn.log "phase${phase}_conn.log"
    rm -f dns.log http.log ssh.log weird.log ssl.log files.log packet_filter.log 2>/dev/null
done
```

4. **Convert Zeek `conn.log` Files to CSV**

From the project root:
```bash
uv run data/DARPA/scripts/zeek_conn_to_csv.py data/DARPA/<scenario>/<network>/all_flows true
```

Notes:
- `true` -> process a single `conn.log` (all flows)
- `false` -> process multiple per-phase `conn.log` files

5. **Label Flows With Attack Phases**

From the project root:
```bash
uv run data/DARPA/scripts/label_flows.py data/DARPA/<scenario>/<network> true
```

Notes:
- `true` -> label flows in the combined all-flows CSV
- `false` -> label per-phase flow CSVs

The labeled dataset is written to:
```bash
data/DARPA/<scenario>/<network>/labeled_flows/all_flows_labeled.csv
```
