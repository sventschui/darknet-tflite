#!/bin/bash

set -e

SCRIPT=$(readlink -f $0)
SCRIPT_DIR=`dirname $SCRIPT`

source "${SCRIPT_DIR}/functions.sh"

# Install darknet
${SCRIPT_DIR}/install_darknet.sh

# Download darknet python bindings
wget -O "/content/darknet_tflite/darknet.py" "https://raw.githubusercontent.com/hank-ai/darknet/$(darknet_git_revision)/src-python/darknet.py"

# Install python packages
pip install -r /content/darknet_tflite/requirements.txt

# Install edgetpu compiler
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list

apt-get update && apt-get install -y edgetpu-compiler