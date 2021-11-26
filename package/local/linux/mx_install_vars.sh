#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

# build configuration
export MXBUILD_CONFIG=Release

# path to source root
export MXSRCDIR=${this_dir}/../../..

# path to build root
export MXBUILDDIR=${MXSRCDIR}/../mechanica_build

# path to install root
export MXINSTALLDIR=${MXSRCDIR}/../mechanica_install

# path to environment root
export MXENV=${MXINSTALLDIR}/mx_env

# path to cuda root directory
export MXCUDAENV=""
