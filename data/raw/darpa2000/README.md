# DARPA 2000 Processing

## Steps

### 1. Unzip the Dataset

The DARPA 2000 datasets are provided in a compressed format. Unzip the dataset and place it in `data/raw/darpa2000/<dataset_name>` (e.g., `data/raw/darpa2000/inside`).

### 2. Label PCAPs
        
Follow the instructions in `src/flow_processing/README.md` to process the merged PCAPs and generate Zeek logs and CSV files.

### 3. Prepare Final Dataset

Finally, follow the instructions in `src/feature_engineering/README.md` to generate the final dataset for model training.

