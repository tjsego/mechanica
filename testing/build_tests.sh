#!/bin/bash

current_dir=$(pwd)

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

if [[ ! -d "${MXENV}" ]]; then
    exit 1
elif [[ ! -d "${MXINSTALLDIR}" ]]; then
    exit 1
fi

MXTESTS_BUILDDIR=${this_dir}/build

mkdir ${MXTESTS_BUILDDIR}

declare -a CMAKE_CONFIG_ARGS

CMAKE_CONFIG_ARGS+=(-DCMAKE_BUILD_TYPE:STRING=${MXBUILD_CONFIG})
CMAKE_CONFIG_ARGS+=(-DCMAKE_PREFIX_PATH:PATH="${MXENV};${MXINSTALLDIR};${MXINSTALLDIR}/lib")
CMAKE_CONFIG_ARGS+=(-DCMAKE_FIND_ROOT_PATH:PATH=${MXENV})
CMAKE_CONFIG_ARGS+=(-DPython_EXECUTABLE:PATH=${MXENV}/bin/python)

if [[ $(uname) == Darwin ]]; then
    export MACOSX_DEPLOYMENT_TARGET=${MXOSX_SYSROOT}
else
    CMAKE_CONFIG_ARGS+=(-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld)

    export CC=${MXENV}/bin/clang
    export CXX=${MXENV}/bin/clang++
fi

cd ${this_dir}

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS[@]}" \
      -S . \
      -B "${MXTESTS_BUILDDIR}"

cmake --build "${MXTESTS_BUILDDIR}" --config ${MXBUILD_CONFIG}

cd ${current_dir}
