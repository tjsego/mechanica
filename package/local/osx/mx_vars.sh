#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"
source ${this_dir}/mx_site_vars.sh

if [[ ! -d "${MXPYSITEDIR}" ]]; then 
    exit 1
fi
if [[ ! -d "${MXENV}" ]]; then 
    exit 2
fi

export PYTHONPATH=${MXPYSITEDIR}:${PYTHONPATH}
