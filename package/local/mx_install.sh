#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

current_dir=$(pwd)

set -o pipefail -e

source ${this_dir}/linux/mx_install_vars.sh

bash ${MXSRCDIR}/package/local/linux/mx_install_env.sh

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ${MXENV}

bash ${MXSRCDIR}/package/local/linux/mx_install_all.sh

cd ${current_dir}
