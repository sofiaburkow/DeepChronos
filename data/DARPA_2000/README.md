# DARPA 2000

## Multi-Step Attack Scenario

This dataset contains a multi-step attack scenario that unfolds over several phases. The attack begins with an IP sweep to identify potential targets, followed by a scan for vulnerabilities in the `sadmind` service. Once a vulnerability is found, the attacker exploits it to gain root access to the target system. After gaining control, the attacker installs the `mstream` DDoS tool and subsequently launches a distributed denial-of-service (DDoS) attack.

The phases of the attack are as follows:

| Phase | Action                 |
| ----- | ---------------------- |
| 1     | IP sweep               |
| 2     | sadmind scan           |
| 3     | sadmind exploit (root) |
| 4     | install mstream DDoS   |
| 5     | launch DDoS            |

## Preprocessing 

The raw packet capture (PCAP) files have been processed using Zeek to extract flow-level features. The resulting flow data is stored in CSV format in the `data/DARPA_2000/flows/` directory. Each CSV file corresponds to a specific phase of the attack.

The attack labels are provided in IDMEF XML format and can be found in the `data/DARPA_2000/labels/` directory. Each XML file contains alerts corresponding to the different phases of the attack.

To create a labeled dataset, the Zeek flow data is merged with the IDMEF alerts based on the 5-tuple (source IP, destination IP, source port, destination port, protocol) and time overlap. The final labeled dataset is saved as `data/DARPA_2000/darpa2000_labeled_dataset.csv`.

### Processing Steps

1. Install `zeek`

Installation instructions can be found here: https://docs.zeek.org/en/v8.0.4/install.html

For Ubuntu 24.04:
```bash
# First, install the correct Linux binary package
echo 'deb http://download.opensuse.org/repositories/security:/zeek/xUbuntu_24.04/ /' | sudo tee /etc/apt/sources.list.d/security:zeek.list
curl -fsSL https://download.opensuse.org/repositories/security:zeek/xUbuntu_24.04/Release.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/security_zeek.gpg > /dev/null
sudo apt update
sudo apt install zeek

# Then, add zeek to your PATH
echo 'export PATH="/opt/zeek/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

2. TODO: unzip dataset files if necessary.

3. Convert PCAP Files to Zeek Logs

    1. Navigate to the directory containing the PCAP files for the target network segment (inside or dmz):

    ```bash
    cd data/DARPA_2000/inside/flows
    # or 
    cd data/DARPA_2000/dmz/flows
    ```

    2. Process all phase PCAP files with Zeek using the following loop:

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

    **Notes:**
    - The `-C` flag tells Zeek to ignore checksum errors, which are common in the DARPA 2000 dataset.
    - The `-r` flag specifies the input PCAP file.
    - Only the connection logs (`conn.log`) are preserved.

4. Convert Zeek `conn.log` files to CSV format.

From the root directory, run the following commands:
```bash
uv run scripts/zeek_conn_to_csv.py data/DARPA_2000/inside/flows
uv run scripts/zeek_conn_to_csv.py data/DARPA_2000/dmz/flows
```

5. Label Flows with Attack Phases

From the root directory, run the following command:
```bash
uv run scripts/label_flows.py data/DARPA_2000/inside

# followed by

uv run scripts/label_flows.py data/DARPA_2000/dmz
```


