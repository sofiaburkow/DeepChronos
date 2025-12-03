# DARPA 2000

## Multi-Step Attack Scenario

The DARPA 2000 intrusion detection evaluation dataset contains a multi-step cyber attack executed in a simulated Air Force Base network. The adversary performs reconnaissance, exploitation, installation of malware, and ultimately launches a distributed denial-of-service (DDoS) attack.

The attack proceeds through the following phases:

| Phase | Description                                                                       |
| ----- | --------------------------------------------------------------------------------- |
| **1** | IP sweep across several subnets to identify active hosts                          |
| **2** | Probe hosts to check for the vulnerable **sadmind** remote administration service |
| **3** | Exploit sadmind to gain **root access**, create a malicious user                  |
| **4** | Install and configure the **mstream** DDoS server and master components           |
| **5** | Initiate a short but coordinated **mstream DDoS attack**                          |

Each phase corresponds to a distinct step in the attacker’s kill chain and is reflected in the provided ground-truth labels.

## Dataset Structure

The DARPA 2000 files are organized into two monitored network segments:
- **inside** - internal Air Force network traffic
- **dmz** - externally facing DMZ network traffic

Each segment includes:
- a **full packet capture** containing all traffic (benign + malicious)
- **per-phase PCAPs** that contain only the activity relevant to a specific attack phase
- **IDMEF XML label files** specifying which sessions correspond to attack activity

In practice:
- Use **full-segment PCAPs** (inside or dmz) when building ML/NIDS datasets, as they include both positive and negative examples.
- Use the **per-phase XML files** to attach ground-truth attack labels for supervised learning or evaluation.

## Preprocessing 

The raw DARPA 2000 dataset is provided as packet capture (PCAP) files. To convert these into a machine-learning–friendly format, the PCAPs are processed using Zeek, which extracts flow-level metadata such as timestamps, IPs, ports, protocol, service, and byte/packet counts.
The resulting flow CSV files are stored in:

```bash
data/DARPA_2000/<inside|dmz>/all_flows/
```

### Attack Labels (IDMEF XML)

Ground-truth attack annotations are provided in IDMEF XML format:

```bash
data/DARPA_2000/<inside|dmz>/labels/
```

Each XML file corresponds to a specific attack phase (1–5) and contains the sessions associated with that part of the multi-step intrusion.

### Labeling Procedure

To build a labeled intrusion detection dataset, the Zeek-derived flow records are merged with the IDMEF alerts using:
- 4-tuple matching: (source IP, destination IP, source port, destination port)
- Timestamp alignment: flows and alerts must overlap or occur within the same second

Each flow is assigned:
- `attack = 1` if it matches a labeled attack session
- `attack_phase = {1..5}` for the corresponding stage
- `attack_id` uniquely identifying the alert

### Output Files

Two output CSV files are produced:

1. Full labeled dataset

Contains all flows (benign + malicious):
```bash
data/DARPA_2000/<inside|dmz>/<inside|dmz>_labeled_flows_all.csv
```

2. Filtered attack-only dataset

Contains only the flows associated with attack activity:
```bash
data/DARPA_2000/<inside|dmz>/<inside|dmz>_labeled_flows_attack.csv
```

These files serve as the basis for downstream ML modeling and DeepProbLog neuro-symbolic experiments.

## Processing Steps

This section describes how to reproduce the preprocessing pipeline used to convert the raw DARPA 2000 PCAP files into Zeek flow CSVs and labeled attack datasets.

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

Unzip the provided DARPA 2000 archives:
```bash
data/DARPA_2000/<inside|dmz>/all_flows.tar.zip
data/DARPA_2000/<inside|dmz>/per_phase_flows.zip
data/DARPA_2000/<inside|dmz>/labels.tar.zip
```

After extraction, each directory will contain:
- Raw PCAP files
- Per-phase PCAPs
- IDMEF XML label files

3. **Convert PCAP Files to Zeek Logs**

**Process the full PCAP (all flows)**

From the project root:
```bash
cd data/DARPA_2000/<inside|dmz>/all_flows/
```

Run Zeek:
```bash
zeek -Cr LLS_DDOS_1.0-<inside|dmz>.dump
```

Remove unneeded logs (keeping only `conn.log`):
```bash
rm analyzer.log dns.log ftp.log packet_filter.log smtp.log weird.log files.log http.log ntp.log reporter.log ssh.log
# For inside segment, also remove the following:
rm dhcp.log snmp.log smb_files.log syslog.log
```

Process per-phase PCAPs (optional)
If you want per-phase flow files:
```bash
cd data/DARPA_2000/<inside|dmz>/per_phase_flows/
```

Run this loop:
```bash
Note: If you want to process all phases at the same time, you can run the loop below in a terminal from the same directory where the PCAP files are located.
```bash
for f in phase-*-tcpdump-out-dump; do
    phase=$(echo $f | grep -oP 'phase-\K\d+')
    echo "Processing phase $phase ..."
    zeek -Cr "$f"
    mv conn.log "phase${phase}_conn.log"
    # Remove other log files
    rm -f dns.log http.log ssh.log weird.log ssl.log files.log packet_filter.log 2>/dev/null
done
```

4. **Convert Zeek `conn.log` Files to CSV**

From the project root, ute the provided converter script:
```bash
uv run scripts/zeek_conn_to_csv.py data/DARPA_2000/<inside|dmz>/all_flows true
```

Notes:
- The final `true` argument means: "Process a single conn.log file (all flows) in this directory."
- If you pass `false`, the script expects multiple per-phase `conn.log` files (e.g., `phase1_conn.log`, etc.).

5. **Convert XML Labels to CSV**
From the project root:
```bash
uv run scripts/xml_alerts_to_csv.py data/DARPA_2000/<inside|dmz>/labels
```

5. **Label Flows With Attack Phases (XML Matching)**

From the project root:
```bash
uv run scripts/label_flows.py data/DARPA_2000/<inside|dmz> true
```

Notes:
- `true` means: "Label flows in the combined all-flows CSV."
- `false` means the script expects per-phase flow CSVs instead.

The output will be written to:
```bash
data/DARPA_2000/<inside|dmz>/<inside|dmz>_labeled_flows_all.csv
data/DARPA_2000/<inside|dmz>/<inside|dmz>_labeled_flows_attack.csv
```

