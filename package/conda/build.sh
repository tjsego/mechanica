#!/bin/bash

MXBUILD_CONFIG=Release

mkdir -p -v mx_build_conda
cd mx_build_conda

declare -a CMAKE_CONFIG_ARGS

CMAKE_CONFIG_ARGS+=(-DCMAKE_BUILD_TYPE:STRING=${MXBUILD_CONFIG})
CMAKE_CONFIG_ARGS+=(-DCMAKE_PREFIX_PATH:PATH=${PREFIX})
CMAKE_CONFIG_ARGS+=(-DCMAKE_FIND_ROOT_PATH:PATH=${PREFIX})
CMAKE_CONFIG_ARGS+=(-DCMAKE_INSTALL_PREFIX:PATH=${PREFIX})
CMAKE_CONFIG_ARGS+=(-DPython_EXECUTABLE:PATH=${PREFIX}/bin/python)
CMAKE_CONFIG_ARGS+=(-DLIBXML_INCLUDE_DIR:PATH=${PREFIX}/include/libxml2)

if [[ $(uname) == Darwin ]]; then
  export MACOSX_DEPLOYMENT_TARGET=${MXOSX_SYSROOT}
  export CONDA_BUILD_SYSROOT="$(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX${MXOSX_SYSROOT}.sdk"
else
  CMAKE_CONFIG_ARGS+=(-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld)

  export CC=${PREFIX}/bin/clang
  export CXX=${PREFIX}/bin/clang++

  # Helping corrade rc find the right libstdc++
  export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
fi

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS[@]}" \
      "${SRC_DIR}"

cmake --build . --config ${MXBUILD_CONFIG} --target install
