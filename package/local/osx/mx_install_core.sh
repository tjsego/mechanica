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

export MACOSX_DEPLOYMENT_TARGET=${MXOSX_SYSROOT}
export CONDA_BUILD_SYSROOT="$(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX${MXOSX_SYSROOT}.sdk"

if [[ ! -d "${CONDA_BUILD_SYSROOT}" ]]; then
    echo "SDK not found"
    exit 3
fi

cmake -DCMAKE_BUILD_TYPE:STRING=${MXBUILD_CONFIG} \
      -G "Ninja" \
      -DCMAKE_PREFIX_PATH:PATH=${MXENV} \
      -DCMAKE_FIND_ROOT_PATH:PATH=${MXENV} \
      -DCMAKE_INSTALL_PREFIX:PATH=${MXINSTALLDIR} \
      -DPython_EXECUTABLE:PATH=${MXENV}/bin/python \
      -DLIBXML_INCLUDE_DIR:PATH=${MXENV}/include/libxml2 \
      "${MXSRCDIR}"

cmake --build . --config ${MXBUILD_CONFIG} --target install

cd ${current_dir}
