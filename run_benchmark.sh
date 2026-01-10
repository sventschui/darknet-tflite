#!/bin/bash

set -eo pipefail

# ===============================
# Config
# ===============================

HOST="192.168.100.80"
USER=""
REMOTE_WORKDIR="./darknet-benchmark"

# ===============================
# Copy scripts
# ===============================

echo "⬆️ Copy assets to ${HOST}..."

REMOTE="${HOST}"

if [[ "$USER" != "" ]]; then
    REMOTE="${USER}@${REMOTE}"
fi

ssh "${REMOTE}" "mkdir -p '${REMOTE_WORKDIR}'"
rsync -avz --progress benchmark functions darknet.py requirements.txt --exclude 'benchmark/out' "${REMOTE}:${REMOTE_WORKDIR}/"
ssh "${REMOTE}" "cd '${REMOTE_WORKDIR}' && wget --no-clobber https://codeberg.org/CCodeRun/darknet/raw/branch/master/src-python/darknet.py"

# ===============================
# Create venv
# ===============================
echo "🛠️ Create venv and install dependencies..."
# TODO: Install https://github.com/feranick/libedgetpu/releases/tag/16.0TF2.19.1-1
ssh "${REMOTE}" "cd '${REMOTE_WORKDIR}' && [ -d .venv ] || python3 -m venv .venv && ./.venv/bin/pip install -q -r requirements.txt"

# ===============================
# Run benchmark
# ===============================
echo "🏃 Running benchmarks..."
ssh "${REMOTE}" "cd '${REMOTE_WORKDIR}' && ./.venv/bin/python -m benchmark"

echo $?

echo "✅ Done"