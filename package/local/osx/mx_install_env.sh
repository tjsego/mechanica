#!/bin/bash

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda create --yes --prefix ${MXENV}
conda env update --prefix ${MXENV} --file ${MXSRCDIR}/package/local/osx/mx_env.yml
