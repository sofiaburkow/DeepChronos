# Flow Processing

This document describes how to convert the PCAP files into labeled flows.

## Dataset Options

**dataset:** either `darpa2000` or `aitv2`.

**scenario:** e.g., `santos` (AIT-LDS V2) or `s1_inside` (DARPA 2000). 

## Install Zeek

Before processing the PCAP files, you need to install Zeek (formerly Bro) on your system. Zeek is a powerful network analysis framework that can process PCAP files and generate detailed logs. Installation instructions are available in the official Zeek documentation: `https://docs.zeek.org/en/v8.0.4/install.html`

**Ubuntu 24.04:**

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

## Processing Steps

### 1. Prepare the Dataset

Follow the dataset preparation instructions in the `data/raw/aitv2/README.md` or `data/raw/darpa2000/README.md` files to prepare the PCAP files for processing.


### 2. Convert PCAP Files to Zeek Logs

From the project root, process each dataset scenario:

```bash
uv run -m src.flow_processing.pcap_to_zeek --dataset <dataset> --scenario <scenario>
```

The conn logs will be saved to `data/interim/<dataset>/<scenario>/zeek_logs/`.

### 3. Convert Zeek Logs to CSV

Convert the generated `conn.log` files into CSV format:

```bash
uv run -m src.flow_processing.zeek_conn_to_csv --dataset <dataset> --scenario <scenario> --overwrite
```

The CSV files will be saved to `data/interim/<dataset>/<scenario>/flows_unlabeled/`.

### 4. Label Flows

Label the flows for each dataset:

```bash
uv run -m src.flow_processing.label_flows --dataset <dataset> --scenario <scenario> --overwrite
```

**Output locations:**

DARPA 2000: `data/interim/<dataset>/<scenario>/flows_labeled/all_flows_labeled.csv`

AIT-LDS V2: `data/interim/<dataset>/<scenario>/flows_labeled/all_flows_labeled_unprocessed.csv`


### 5. Final Processing Steps for AIT-LDS V2 Flows

The AIT-LDS V2 requires additional processing to trim the dataset to the simulation period, remove duplicate flows, and label flows according to broader MSA categories. 

Run the notebook `src/notebooks/data_processing/ait.ipynb` to process the AIT-LDS V2 flows and generate the final labeled dataset. 

Remember to set the `scenario` variable in the notebook to the appropriate value for the scenario you are processing (e.g., `scenario = "santos"`).

The final dataset is written to: `data/interim/<dataset>/<scenario>/flows_labeled/all_flows_labeled.csv`.
