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
find "${TARGET_DIR}/set_01" -name "*.jpg" > "${TARGET_DIR}/Cats_train.txt"
cp "${TARGET_DIR}/Cats_train.txt" "${TARGET_DIR}/Cat_valid.txt"
