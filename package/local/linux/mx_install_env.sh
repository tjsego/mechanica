#!/bin/bash

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda create --yes --prefix ${MXENV}
conda env update --prefix ${MXENV} --file ${MXSRCDIR}/package/local/linux/mx_env.yml
