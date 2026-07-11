#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_directory>"
    exit 1
fi

DATASET_DIR="$1"

if [ ! -d "$DATASET_DIR/gather" ]; then
    echo "Error: $DATASET_DIR/gather does not exist"
    exit 1
fi

OUT_DIR="./${DATASET_DIR}_merged_pcaps"

mkdir -p "$OUT_DIR"

echo "Processing dataset: $DATASET_DIR"
echo "Output directory: $OUT_DIR"

declare -A SERVER_FILES

# Find all PCAPs
while read -r file; do

    # Extract server name from:
    # dataset/gather/<server>/...
    server=$(echo "$file" | sed -E "s|.*/gather/([^/]+)/.*|\1|")

    # Normalize
    server=$(echo "$server" | tr '-' '_')

    SERVER_FILES["$server"]+="$file "

done < <(find "$DATASET_DIR/gather" -name "*.pcap*" -type f)


# Merge per server
for server in "${!SERVER_FILES[@]}"; do

    echo "Merging $server..."

    mergecap \
        -a ${SERVER_FILES[$server]} \
        -w "$OUT_DIR/log_${server}.pcap"

done

echo "Finished."