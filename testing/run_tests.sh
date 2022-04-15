#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

MXTESTS_TESTSDIR=${this_dir}/build
if [[ ! -d "${MXENV}" ]]; then
    exit 1
fi
if [[ $(uname) == Darwin ]]; then
    export DYLD_LIBRARY_PATH=${MXENV}/lib:${DYLD_LIBRARY_PATH}
else
    export LD_LIBRARY_PATH=${MXENV}/lib:${LD_LIBRARY_PATH}
fi
cd ${MXTESTS_TESTSDIR}
ctest --output-on-failure
