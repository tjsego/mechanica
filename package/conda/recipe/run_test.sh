#!/bin/bash

MXBUILD_CONFIG=Release
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

set MXTESTS_ROOT=$(pwd)/testing
set MXTESTS_BUILDDIR=${MXTESTS_ROOT}/build

mkdir ${MXTESTS_BUILDDIR}

declare -a CMAKE_CONFIG_ARGS

CMAKE_CONFIG_ARGS+=(-DCMAKE_BUILD_TYPE:STRING=${MXBUILD_CONFIG})
CMAKE_CONFIG_ARGS+=(-DCMAKE_PREFIX_PATH:PATH=${PREFIX})
CMAKE_CONFIG_ARGS+=(-DCMAKE_FIND_ROOT_PATH:PATH=${PREFIX})
CMAKE_CONFIG_ARGS+=(-DPython_EXECUTABLE:PATH=${PREFIX}/bin/python)
CMAKE_CONFIG_ARGS+=(-DMX_INSTALL_ROOT:PATH=${PREFIX})

if [[ $(uname) == Darwin ]]; then
    export MACOSX_DEPLOYMENT_TARGET=${MXOSX_SYSROOT}
    CMAKE_CONFIG_ARGS+=(-DCMAKE_OSX_SYSROOT:PATH=${CONDA_BUILD_SYSROOT})
else
    CMAKE_CONFIG_ARGS+=(-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld)

    export CC=${PREFIX}/bin/clang
    export CXX=${PREFIX}/bin/clang++
fi

cd ${MXTESTS_ROOT}

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS[@]}" \
      -S . \
      -B "${MXTESTS_BUILDDIR}"

cmake --build "${MXTESTS_BUILDDIR}" --config ${MXBUILD_CONFIG}

cd ${MXTESTS_BUILDDIR}
ctest --output-on-failure
