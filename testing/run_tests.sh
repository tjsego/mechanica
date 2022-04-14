#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

set MXTESTS_TESTSDIR=${this_dir}/build

cd ${MXTESTS_TESTSDIR}
ctest --output-on-failure
