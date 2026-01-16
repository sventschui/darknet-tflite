#!/usr/bin/env bash

set -eo pipefail

SCRIPT=$(readlink -f $0)
SCRIPT_DIR=`dirname $SCRIPT`

PROJECT_ID="1"
LABEL_STUDIO_HOST="http://192.168.100.80:8082"
IMG_HOST="http://192.168.100.80:8081"

read -s -p "Enter LabelStudio API KEY: " API_KEY
echo ""

if [[ "${API_KEY}" == "" ]]; then
  exit 1;
fi

# Export
WORKDIR="${SCRIPT_DIR}/TMP_EXPORT"
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"

curl "${LABEL_STUDIO_HOST}/api/projects/${PROJECT_ID}/export?exportType=YOLO" \
  -H "Authorization: Token ${API_KEY}" \
  -o "${WORKDIR}/labels.zip"

unzip -q -d "${WORKDIR}" "${WORKDIR}/labels.zip"

cp "${WORKDIR}/classes.txt" "${SCRIPT_DIR}/Cats.names"

mkdir -p "${SCRIPT_DIR}/set_01"

for f in "${WORKDIR}/labels/"*.txt; do
  filename="$(basename "$f")"
  name="${filename#*__}"

  cp "$f" "${SCRIPT_DIR}/set_01/${name}"
done

for txtfile in ${SCRIPT_DIR}/set_01/*.txt; do
  filename="$(basename "$txtfile")"
  name="${filename%.txt}"

  url="${IMG_HOST}/${name}.jpg"
  output="${SCRIPT_DIR}/set_01/${name}.jpg"

  if [[ ! -f "$output" ]]; then
    echo "Downloading $url"

    curl -f -o "$output" "$url"
  fi
done

rm -f "${SCRIPT_DIR}/set_01.zip"

(cd "${SCRIPT_DIR}" && zip set_01.zip set_01/*)