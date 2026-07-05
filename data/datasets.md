# Datasets

This project evaluates multi-step attack (MSA) detection on two publicly available datasets:

- **DARPA 2000 Intrusion Detection Dataset** – a classic benchmark for multi-step attack detection.
- **AIT Log Dataset V2 (AIT-LDS V2)** – a more recent dataset containing realistic enterprise network traffic and attack scenarios.

Both datasets provide network packet captures (PCAPs) together with labels describing the attack progression.

---

## DARPA 2000

DARPA 2000 contains network traffic collected from a simulated Air Force network with sensors placed both inside the network and in the DMZ.

The dataset includes two distributed denial-of-service (DDoS) attack scenarios:

- **LLDOS 1.0** – attack performed by a novice attacker.
- **LLDOS 2.0.2** – attack performed by a more stealthy attacker.

Since LLDOS 1.0 contains substantially more attack samples, all experiments in this repository are performed using the **LLDOS 1.0** scenario (both inside and DMZ traffic).

### Attack Phases

The modeled attack consists of five sequential phases:

1. Scanning
2. Probing
3. Exploitation
4. Installation
5. Launching DDoS

---

## AIT Log Dataset V2

AIT-LDS V2 consists of eight synthetic enterprise network scenarios collected by the Austrian Institute of Technology (AIT). Each scenario contains:

- 4–6 days of benign network activity
- A labeled multi-step attack launched during the final day
- Network traffic (PCAPs) and host logs

Although all scenarios contain the same attack, they differ in scan intensity, timing, event frequency, and network behavior.

### Scenarios Used

TODO

### Original Attack Phases

The original AIT attack contains six phases:

1. Scanning
2. Exploitation
3. Password Cracking
4. Privilege Escalation
5. Remote Command Execution
6. Data Exfiltration

### Modeled Attack

This repository uses only **network traffic (PCAPs)**. Consequently, phases that rely primarily on host logs cannot be observed directly and are excluded.

Specifically:

- **Privilege Escalation** is excluded.
- **Remote Command Execution** is excluded.

Additionally, the original dataset starts the **Data Exfiltration** activity before the remaining attack phases. To obtain a consistent temporal multi-step attack sequence, we model the attack using the following order:

1. Data Exfiltration
2. Scanning
3. Exploitation
4. Password Cracking

This ordering is used throughout all AIT-LDS V2 experiments.

---

## Dataset Preparation

Both datasets are processed into network flows before training. The resulting flow files are then used by the DeepProbLog training pipeline.

The repository contains scripts for preprocessing, feature extraction, and dataset generation prior to training.