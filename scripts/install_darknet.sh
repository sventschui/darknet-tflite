#!/bin/bash

set -e

# Install darknet
if [[ "${DARKNET_VERSION}" == "" ]]; then
    echo "Missing DARKNET_VERSION env var!"
    exit 1
fi

if [[ ! -d '/usr/lib64-nvidia' ]]; then
    cpu_gpu="cpu"
else
    cpu_gpu="gpu"
fi
arch="$(dpkg --print-architecture)"
deb_file="darknet-${DARKNET_VERSION}-Linux-${arch}-${cpu_gpu}.deb"
deb_path="/content/darknet_tflite/packages/${deb_file}"

sudo dpkg -i "$deb_path" || apt --fix-broken install
