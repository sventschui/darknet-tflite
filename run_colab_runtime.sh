#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

WORK_DIR="${SCRIPT_DIR}/work"
PACKAGES_DIR="${SCRIPT_DIR}/packages"
FUNCTIONS_DIR="${SCRIPT_DIR}/functions"

mkdir -p "${WORK_DIR}"

if [[ "$(uname)" != "Darwin" ]]; then
    OPTS="--gpus=all"
else
    OPTS=""
fi

docker run \
    $OPTS \
    --name colab-runtime \
    --rm \
    -v ${SCRIPT_DIR}/colab-config.json:/datalab/web/config/settings.json \
    -p 127.0.0.1:9000:8080 \
    -p 127.0.0.1:9001:8081 \
    -v "${WORK_DIR}:/content" \
    -v "${FUNCTIONS_DIR}:/content/functions" \
    -v "${SCRIPT_DIR}:/content/darknet_tflite" \
    -v "${PACKAGES_DIR}:/packages" \
    europe-docker.pkg.dev/colab-images/public/runtime --help
