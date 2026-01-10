#!/usr/bin/env bash

set -e


SCRIPT=$(readlink -f $0)
SCRIPT_DIR=`dirname $SCRIPT`

# Loop over all .txt files in the current directory
for txtfile in ${SCRIPT_DIR}/set_01/*.txt; do
  # Skip if no .txt files exist
  [ -e "$txtfile" ] || continue

  # Extract filename without extension
  filename="$(basename "$txtfile")"
  name="${filename%.txt}"

  # Download the corresponding JPG
  url="http://catpi.local:8081/${name}.jpg"
  output="${SCRIPT_DIR}/set_01/${name}.jpg"

  echo "Downloading $url"
  curl -f -o "$output" "$url" || echo "Failed to download $url"
done
