#!/bin/bash

set -e

TARGET_DIR="${1:-/content/Cats}"

if [[ "$TARGET_DIR" != /content/* ]]; then
    echo "Error: path must start with /content/"
    exit 1
fi

rm -rf "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

unzip -q -o /content/darknet_tflite/nn/Cats/set_01.zip -d "${TARGET_DIR}"
cp /content/darknet_tflite/nn/Cats/{Cats.cfg,Cats.names,Cats.data} "${TARGET_DIR}"

sed -i "s|/content/Cats|"${TARGET_DIR}"|" "${TARGET_DIR}"/Cats.data 

# Create train.txt with sample images (for demonstration)
find "${TARGET_DIR}/set_01" -name "*.jpg" | awk '
{ lines[NR]=$0 }
END {
  cutoff = int(NR * 0.8)
  for (i=1; i<=NR; i++) {
    if (i <= cutoff)
      print lines[i] > "'"${TARGET_DIR}"'/Cats_train.txt"
    else
      print lines[i] > "'"${TARGET_DIR}"'/Cats_valid.txt"
  }
}' 
