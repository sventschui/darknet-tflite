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
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)

    case "$gpu_name" in
    *T4*)
        gpu_class="T4"
        ;;
    *L4*|*A100*|*H100*)
        gpu_class="L4"
        ;;
    *)
        echo "Unsupported GPU: $gpu_name"
        exit 1
        ;;
    esac

    echo "Detected GPU: $gpu_name"
    echo "GPU class: $gpu_class"

    cpu_gpu="gpu-${gpu_class}"
fi
arch="$(dpkg --print-architecture)"
deb_file="darknet-${DARKNET_VERSION}-Linux-${arch}-${cpu_gpu}.deb"
deb_path="/content/darknet_tflite/packages/${deb_file}"

sudo dpkg -i "$deb_path" || apt --fix-broken install
