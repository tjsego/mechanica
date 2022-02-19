#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

current_dir=$(pwd)

set -o pipefail -e

if [[ $(uname) == Darwin ]]; then
    subdir=osx
else
    subdir=linux
fi

source ${this_dir}/${subdir}/mx_install_vars.sh

bash ${MXSRCDIR}/package/local/${subdir}/mx_install_env.sh

source ${HOME}/miniconda3/etc/profile.d/conda.sh

# Install CUDA support if requested
if [ -z "${MX_WITHCUDA+x}" ]; then
    if [ ${MX_WITHCUDA} -eq 1 ]; then 
        # Validate specified compute capability
        if [ ! -z "${CUDAARCHS+x}" ]; then
            echo "No compute capability specified"
            exit 1
        fi

        echo "Detected CUDA support request"
        echo "Installing additional dependencies..."

        export MXCUDAENV=${MXENV}
        conda install -y -c nvidia -p ${MXENV} cuda
    fi
fi

conda activate ${MXENV}

bash ${MXSRCDIR}/package/local/${subdir}/mx_install_all.sh

cd ${current_dir}
