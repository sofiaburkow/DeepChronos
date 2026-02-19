## Datasets

### DARPA 2000

#### Multi-Step Attack Scenario

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

#### Dataset Structure

The DARPA 2000 dataset is organized into two monitored network segments:
- **inside** – internal Air Force network traffic
- **dmz** – externally facing DMZ traffic

Each segment contains:
- A **full packet capture (PCAP)** with all traffic (benign + malicious)
- **Per-phase PCAPs** containing only traffic relevant to a specific attack phase
- **IDMEF XML label files** describing attack sessions