# Flow Processing

This document describes how to convert raw PCAP files into Zeek flow records and generate labeled datasets for model training.

## 1. Install Zeek

Installation instructions are available in the official Zeek documentation:

https://docs.zeek.org/en/v8.0.4/install.html

### Ubuntu 24.04

```bash
# Add repository and install Zeek
echo 'deb http://download.opensuse.org/repositories/security:/zeek/xUbuntu_24.04/ /' | sudo tee /etc/apt/sources.list.d/security:zeek.list

curl -fsSL https://download.opensuse.org/repositories/security:/zeek/xUbuntu_24.04/Release.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/security_zeek.gpg > /dev/null

sudo apt update
sudo apt install zeek

# Add Zeek to PATH
echo 'export PATH="/opt/zeek/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 2. Extract the Dataset

All datasets are provided as compressed archives in `data/raw/<dataset>/`.Use the appropriate decompression tool to extract the files.

## 3. Convert PCAP Files to Zeek Logs

From the project root, process each dataset scenario:

```bash
uv run -m src.flow_processing.pcap_to_zeek --dataset <dataset> --scenario <scenario>
```

Supported datasets:

* `darpa2000`
* `aitv2`

The `scenario` specifies the network scenario to process. For example:

* `s1_inside` or `s1_dmz` (DARPA 2000)
* `santos` or `fox` (AIT-LDS V2)

## 4. Convert Zeek Logs to CSV

Convert the generated `conn.log` files into CSV format:

```bash
uv run -m src.flow_processing.zeek_conn_to_csv --dataset <dataset> --scenario <scenario> --overwrite
```

## 5. Label Flows

Generate attack phase labels:

```bash
uv run -m src.flow_processing.label_flows --dataset <dataset> --scenario <scenario> --overwrite
```

Output locations:

**DARPA 2000**

```text
data/interim/<dataset>/<scenario>/flows_labeled/all_flows_labeled.csv
```

**AIT-LDS V2**

```text
data/interim/<dataset>/<scenario>/flows_labeled/all_flows_labeled_unprocessed.csv
```

## Additional Processing for AIT-LDS V2

To generate the final dataset for AIT-LDS V2, run:

```text
src/notebooks/data_processing/ait.ipynb
```

The notebook trims the simulation period, removes duplicate flows, and filters flows according to the attack phase definitions used in this project.

The final dataset is written to:

```text
data/interim/<dataset>/<scenario>/flows_labeled/all_flows_labeled.csv
```

If the notebook contains multiple dataset-specific configurations, remember to uncomment the configuration corresponding to the scenario being processed.
