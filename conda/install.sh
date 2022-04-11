#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

ENVIRONMENT_FILE=environment.yml

echo "Using ${ENVIRONMENT_FILE} ..."

# Ensure mamba is installed.
conda install -y mamba

# Create library environment.
mamba env create -f ${SCRIPT_DIR}/${ENVIRONMENT_FILE} \
&& eval "$(conda shell.bash hook)" \
&& conda activate torchbox3d \
&& python -m pip install -e $SCRIPT_DIR/..
