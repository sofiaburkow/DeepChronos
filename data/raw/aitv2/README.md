# AIT-LDS V2 Network Log Processing

## Initial Steps

### 1. Download and Extract Dataset

Download the desired AIT-LDS2.0 dataset (e.g., Santos) from https://doi.org/10.5281/zenodo.5789064

Unzip the dataset and place it in `data/raw/aitv2/<dataset_name>` (e.g., `data/raw/aitv2/santos`)

### 2. Process PCAPs

All the servers (webserver, inet-firewall, vpn, ...) have packet captures, usually placed in the `.\<company name>\gather\<server name>\logs\suricata` folder. To prepare the PCAPs for further processing, you need to run the `process_pcaps.sh` script located in the `data/raw/aitv2/<dataset_name>` folder: 

```bash 
./process_pcaps.sh <dataset_name> (e.g., `./process_pcaps.sh santos`). 
```

This script will merge PCAPs originating from the same server and rename them to a standard .pcap format. The merged PCAPs will be placed in the `data/raw/aitv2/<dataset_name>_merged_pcaps` folder.

### 3. Prepare Netflows

Download the corresponding netflows from https://doi.org/10.5281/zenodo.6610489 and place them in the same folder as the merged PCAPs (e.g.: `.\<company name>\gather\merged_pcaps`). These are used to label the flows extracted from the PCAPs.

## Further Processing

### 1. Label PCAPs
        
Next, follow the instructions in `src/flow_processing/README.md` to process the merged PCAPs and generate Zeek logs and CSV files.

### 2. Prepare Final Dataset

Finally, follow the instructions in `src/feature_engineering/README.md` to generate the final dataset for model training.

## Remarks

Due to the large size of the AIT-LDS V2 datasets, it is recommended to process each scenario separately and remove the raw data for that scenario after processing to save disk space.
