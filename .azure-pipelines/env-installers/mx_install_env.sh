#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"
MXENV=${this_dir}/mx_env

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda create --yes --prefix ${MXENV}
conda env update --prefix ${MXENV} --file ${this_dir}/mx_rtenv.yml