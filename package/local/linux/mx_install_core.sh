#!/bin/bash

if [[ ! -d "${MXENV}" ]]; then
    exit 1
elif [[ ! -d "${MXSRCDIR}" ]]; then
    exit 2
fi

current_dir=$(pwd)

mkdir -p -v ${MXBUILDDIR}
mkdir -p -v ${MXINSTALLDIR}

cd ${MXBUILDDIR}

export CC=${MXENV}/bin/clang
export CXX=${MXENV}/bin/clang++

cmake -DCMAKE_BUILD_TYPE:STRING=${MXBUILD_CONFIG} \
      -G "Ninja" \
      -DCMAKE_PREFIX_PATH:PATH=${MXENV} \
      -DCMAKE_FIND_ROOT_PATH:PATH=${MXENV} \
      -DCMAKE_INSTALL_PREFIX:PATH=${MXINSTALLDIR} \
      -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
      -DPython_EXECUTABLE:PATH=${MXENV}/bin/python \
      -DLIBXML_INCLUDE_DIR:PATH=${MXENV}/include/libxml2 \
      -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=${MXCUDAENV} \
      "${MXSRCDIR}"

cmake --build . --config ${MXBUILD_CONFIG} --target install

cd ${current_dir}
