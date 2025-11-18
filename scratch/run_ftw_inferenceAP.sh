#!/bin/bash

# Define paths
CSV_FILE="/home/ubuntu/workspace/sentinel2_tileindex_WRImexicotestsite_urls.csv"
DATA_DIR="ftw_data_Mexico_2020" #ftw_data_AP_2022"
OUTPUT_DIR="ftw_output_Mexico_2020" #"ftw_output_AP_2022" 

# Ensure directories exist
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# Model checkpoint path
MODEL_PATH="/home/ubuntu/workspace/3_Class_FULL_FTW_Pretrained.ckpt"

# Read CSV and extract tile IDs and image pairs
declare -A SCENES
while IFS=, read -r TILE WIN_A WIN_B; do
    # Skip the header line
    if [[ "$TILE" != "Tile" ]]; then
        SCENES["$TILE"]="$WIN_A $WIN_B"
    fi
done < "$CSV_FILE"

# Step 1: Download Sentinel-2 data (Skip if already exists)
echo "Checking Sentinel-2 data..."
for tile in "${!SCENES[@]}"; do
    read -r WIN_A WIN_B <<< "${SCENES[$tile]}"
    OUTPUT_FILE="$DATA_DIR/AP_${tile}.tif"
    
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "Downloading data for $tile..."
        ftw inference download \
            --win_a "$WIN_A" \
            --win_b "$WIN_B" \
            -f -o "$OUTPUT_FILE"
        echo "Downloaded data saved to $OUTPUT_FILE"
    else
        echo "Data for $tile already exists. Skipping download."
    fi
done

# Step 2: Run inference (Skip if already inferred)
echo "Running model inference..."
for tile in "${!SCENES[@]}"; do
    INPUT_FILE="$DATA_DIR/AP_${tile}.tif"
    OUTPUT_FILE="$OUTPUT_DIR/AP_${tile}-inf.tif"

    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "Processing $tile..."
        ftw inference run "$INPUT_FILE" \
            -f -o "$OUTPUT_FILE" \
            --gpu 0 -m "$MODEL_PATH"
        echo "Inference complete for $tile. Output saved to $OUTPUT_FILE"
    else
        echo "Inference output for $tile already exists. Skipping inference."
    fi
done

# Step 3: Polygonize outputs (Skip if already polygonized)
#echo "Generating vector outputs..."
#for tile in "${!SCENES[@]}"; do
#   INPUT_FILE="$OUTPUT_DIR/AP_${tile}-inf.tif"
#    OUTPUT_FILE="$OUTPUT_DIR/AP_${tile}-inf.parquet"

#    if [ ! -f "$OUTPUT_FILE" ]; then
#        echo "Polygonizing $tile..."
#        ftw inference polygonize "$INPUT_FILE" \
#            --out "$OUTPUT_FILE"
#        echo "Polygonized output saved as $OUTPUT_FILE"
#    else
#        echo "Polygonized output for $tile already exists. Skipping polygonization."
#    fi
#done

echo "All processing complete!"
