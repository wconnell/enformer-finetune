#!/bin/bash

# Define URLs and file paths
DATA_DIR="data"
URL1="https://www.dropbox.com/s/n549ucfqu0u9tu6/master_random_frezzed_regions_train_test_validation_generated_genome_all_dataset.txt?dl=1"
FILE1="${DATA_DIR}/master_random_frezzed_regions_train_test_validation_generated_genome_all_dataset.txt"
# original testing data
URL2="https://www.encodeproject.org/files/ENCFF470BSF/@@download/ENCFF470BSF.bigWig"
FILE2="${DATA_DIR}/ENCFF470BSF.bw"
# new data
URL3="https://www.encodeproject.org/files/ENCFF972GVB/@@download/ENCFF972GVB.bigWig"
FILE3="${DATA_DIR}/ENCFF972GVB.bw"
URL4="https://raw.githubusercontent.com/igvteam/igv/master/genomes/sizes/hg38.chrom.sizes"
FILE4="${DATA_DIR}/hg38.chrom.sizes"
URL5="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
FILE5="${DATA_DIR}/hg38.fa.gz"

# Create data directory if it does not exist
mkdir -p "$DATA_DIR" || { echo "Failed to create data directory"; exit 1; }

# Download files using curl in parallel
echo "Downloading files..."
{
    curl -L "$URL1" -o "$FILE1" || { echo "Failed to download $URL1"; exit 1; }
} &
{
    curl -L "$URL2" -o "$FILE2" || { echo "Failed to download $URL2"; exit 1; }
} &
{
    curl -L "$URL3" -o "$FILE3" || { echo "Failed to download $URL3"; exit 1; }
} &
{
    curl -L "$URL4" -o "$FILE4" || { echo "Failed to download $URL4"; exit 1; }
} &
{
    curl -L "$URL5" -o "$FILE5" || { echo "Failed to download $URL5"; exit 1; }
} &

# Wait for all background processes to finish
wait

# Unzip the genome file
echo "Extracting files..."
gunzip "$FILE5" || { echo "Failed to unzip $FILE5"; exit 1; }

echo "Done."
