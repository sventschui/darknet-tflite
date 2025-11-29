#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

WORK_DIR="${SCRIPT_DIR}/work"
PACKAGES_DIR="${SCRIPT_DIR}/packages"

mkdir -p "${WORK_DIR}"

if [[ "$(uname)" != "Darwin" ]]; then
    OPTS="--gpus=all"
else
    OPTS="-e USE_GPU=OFF"
fi

docker run \
    $OPTS \
    -v ${SCRIPT_DIR}/colab-config.json:/datalab/web/config/settings.json \
    -p 127.0.0.1:9000:8080 \
    -p 127.0.0.1:9001:8081 \
    -v "${WORK_DIR}:/content" \
    -v "${PACKAGES_DIR}:/packages" \
    europe-docker.pkg.dev/colab-images/public/runtime --help
