#!/bin/bash

# Define directories
DATA_DIR="ftw_data"
OUTPUT_DIR="ftw_output"

# Ensure directories exist
mkdir -p $DATA_DIR
mkdir -p $OUTPUT_DIR

# Model checkpoint path
MODEL_PATH="/home/ubuntu/workspace/3_Class_FULL_FTW_Pretrained.ckpt"

# Sentinel-2 images for tile T20KNE
IMAGES=(
    "S2B_MSIL2A_20221128T141709_R010_T20KNE_20221129T203001"
    "S2A_MSIL2A_20220328T141741_R010_T20KNE_20220329T024308"
    "S2B_MSIL2A_20220830T141709_R010_T20KNE_20220831T202051"
)

# Step 1: Generate all possible pairs and run inference
echo "Running model inference on all pairs..."
for ((i=0; i<${#IMAGES[@]}; i++)); do
    for ((j=i+1; j<${#IMAGES[@]}; j++)); do
        WIN_A=${IMAGES[$i]}
        WIN_B=${IMAGES[$j]}
        PAIR_ID="T20KNE_${WIN_A}_vs_${WIN_B}"
        PAIR_TIF="$DATA_DIR/${PAIR_ID}.tif"

        # Download data only for paired images
        if [ ! -f "$PAIR_TIF" ]; then
            echo "Downloading pair: $WIN_A vs $WIN_B"
            ftw inference download \
                --win_a $WIN_A \
                --win_b $WIN_B \
                -f -o $PAIR_TIF
            echo "Downloaded paired data saved to $PAIR_TIF"
        else
            echo "Paired data for $PAIR_ID already exists. Skipping download."
        fi

        # Run inference
        if [ ! -f "$OUTPUT_DIR/${PAIR_ID}-inf.tif" ]; then
            echo "Running inference for $PAIR_ID..."
            ftw inference run $PAIR_TIF \
                -f -o $OUTPUT_DIR/${PAIR_ID}-inf.tif \
                --gpu 0 -m $MODEL_PATH
            echo "Inference complete for $PAIR_ID. Output saved to $OUTPUT_DIR/${PAIR_ID}-inf.tif"
        else
            echo "Inference output for $PAIR_ID already exists. Skipping inference."
        fi

        # Polygonize outputs
        if [ ! -f "$OUTPUT_DIR/${PAIR_ID}-inf.parquet" ]; then
            echo "Polygonizing $PAIR_ID..."
            ftw inference polygonize $OUTPUT_DIR/${PAIR_ID}-inf.tif \
                --out $OUTPUT_DIR/${PAIR_ID}-inf.parquet
            echo "Polygonized output saved as $OUTPUT_DIR/${PAIR_ID}-inf.parquet"
        else
            echo "Polygonized output for $PAIR_ID already exists. Skipping polygonization."
        fi
    done
done

echo "All processing complete!"
