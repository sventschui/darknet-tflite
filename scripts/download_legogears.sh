#!/bin/bash

set -e

if [[ ! -f "/content/legogears_2_dataset.zip" ]]; then
    wget -O "/content/legogears_2_dataset.zip" "https://www.ccoderun.ca/programming/2024-05-01_LegoGears/legogears_2_dataset.zip";
fi

TARGET_DIR="${1:-/content/LegoGears_v2}"

if [[ "$TARGET_DIR" != /content/* ]]; then
    echo "Error: path must start with /content/"
    exit 1
fi

rm -rf "${TARGET_DIR}"

unzip -q -o /content/legogears_2_dataset.zip -d "${TARGET_DIR}"

NESTED_DIR="${TARGET_DIR}/LegoGears_v2"
mv "${NESTED_DIR}"/* "${NESTED_DIR}"/.* "${NESTED_DIR}"/..?* "${TARGET_DIR}" 2>/dev/null || true
rmdir "${NESTED_DIR}"

sed -i "s|/home/stephane/nn/LegoGears|"${TARGET_DIR}"|" "${TARGET_DIR}"/LegoGears.data 

# Create train.txt with sample images
find "${TARGET_DIR}/set_01" -name "*.jpg" > "${TARGET_DIR}/LegoGears_train.txt"
cp "${TARGET_DIR}/LegoGears_train.txt" "${TARGET_DIR}/LegoGears_valid.txt"