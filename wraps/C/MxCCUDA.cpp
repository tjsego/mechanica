/**
 * @file MxCCUDA.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for CUDA-accelerated features
 * @date 2022-04-07
 */

#include "MxCCUDA.h"

#include "mechanica_c_private.h"

#include <cuda/MxSimulatorCUDAConfig.h>
#include <MxSimulator.h>


//////////////////
// Module casts //
//////////////////


namespace mx { 

MxEngineCUDAConfig *castC(struct MxEngineCUDAConfigHandle *handle) {
    return castC<MxEngineCUDAConfig, MxEngineCUDAConfigHandle>(handle);
}

MxBondCUDAConfig *castC(struct MxBondCUDAConfigHandle *handle) {
    return castC<MxBondCUDAConfig, MxBondCUDAConfigHandle>(handle);
}

MxAngleCUDAConfig *castC(struct MxAngleCUDAConfigHandle *handle) {
    return castC<MxAngleCUDAConfig, MxAngleCUDAConfigHandle>(handle);
}

MxSimulatorCUDAConfig *castC(struct MxSimulatorCUDAConfigHandle *handle) {
    return castC<MxSimulatorCUDAConfig, MxSimulatorCUDAConfigHandle>(handle);
}

}

#define MXENGINECUDACONFIGHANDLE_GET(handle, varname) \
    MxEngineCUDAConfig *varname = mx::castC<MxEngineCUDAConfig, MxEngineCUDAConfigHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXBONDCUDACONFIGHANDLE_GET(handle, varname) \
    MxBondCUDAConfig *varname = mx::castC<MxBondCUDAConfig, MxBondCUDAConfigHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXANGLECUDACONFIGHANDLE_GET(handle, varname) \
    MxAngleCUDAConfig *varname = mx::castC<MxAngleCUDAConfig, MxAngleCUDAConfigHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXSIMULATORCUDACONFIGHANDLE_GET(handle, varname) \
    MxSimulatorCUDAConfig *varname = mx::castC<MxSimulatorCUDAConfig, MxSimulatorCUDAConfigHandle>(handle); \
    MXCPTRCHECK(varname);


////////////////////////
// MxEngineCUDAConfig //
////////////////////////


HRESULT MxCEngineCUDAConfig_onDevice(struct MxEngineCUDAConfigHandle *handle, bool *onDevice) {
    MXENGINECUDACONFIGHANDLE_GET(handle, engcuda);
    MXCPTRCHECK(onDevice);
    *onDevice = engcuda->onDevice();
    return S_OK;
}

HRESULT MxCEngineCUDAConfig_getDevice(struct MxEngineCUDAConfigHandle *handle, int *deviceId) {
    MXENGINECUDACONFIGHANDLE_GET(handle, engcuda);
    MXCPTRCHECK(deviceId);
    *deviceId = engcuda->getDevice();
    return S_OK;
}

HRESULT MxCEngineCUDAConfig_setDevice(struct MxEngineCUDAConfigHandle *handle, unsigned int deviceId) {
    MXENGINECUDACONFIGHANDLE_GET(handle, engcuda);
    return engcuda->setDevice(deviceId);
}

HRESULT MxCEngineCUDAConfig_clearDevice(struct MxEngineCUDAConfigHandle *handle) {
    MXENGINECUDACONFIGHANDLE_GET(handle, engcuda);
    return engcuda->clearDevice();
}

HRESULT MxCEngineCUDAConfig_toDevice(struct MxEngineCUDAConfigHandle *handle) {
    MXENGINECUDACONFIGHANDLE_GET(handle, engcuda);
    return engcuda->toDevice();
}

HRESULT MxCEngineCUDAConfig_fromDevice(struct MxEngineCUDAConfigHandle *handle) {
    MXENGINECUDACONFIGHANDLE_GET(handle, engcuda);
    return engcuda->fromDevice();
}

HRESULT MxCEngineCUDAConfig_setBlocks(struct MxEngineCUDAConfigHandle *handle, unsigned int numBlocks) {
    MXENGINECUDACONFIGHANDLE_GET(handle, engcuda);
    return engcuda->setBlocks(numBlocks);
}

HRESULT MxCEngineCUDAConfig_setThreads(struct MxEngineCUDAConfigHandle *handle, unsigned int numThreads) {
    MXENGINECUDACONFIGHANDLE_GET(handle, engcuda);
    return engcuda->setThreads(numThreads);
}

HRESULT MxCEngineCUDAConfig_refreshPotentials(struct MxEngineCUDAConfigHandle *handle) {
    MXENGINECUDACONFIGHANDLE_GET(handle, engcuda);
    return engcuda->refreshPotentials();
}

HRESULT MxCEngineCUDAConfig_refreshFluxes(struct MxEngineCUDAConfigHandle *handle) {
    MXENGINECUDACONFIGHANDLE_GET(handle, engcuda);
    return engcuda->refreshFluxes();
}

HRESULT MxCEngineCUDAConfig_refreshBoundaryConditions(struct MxEngineCUDAConfigHandle *handle) {
    MXENGINECUDACONFIGHANDLE_GET(handle, engcuda);
    return engcuda->refreshBoundaryConditions();
}

HRESULT MxCEngineCUDAConfig_refresh(struct MxEngineCUDAConfigHandle *handle) {
    MXENGINECUDACONFIGHANDLE_GET(handle, engcuda);
    return engcuda->refresh();
}


//////////////////////
// MxBondCUDAConfig //
//////////////////////


HRESULT MxCBondCUDAConfig_onDevice(struct MxBondCUDAConfigHandle *handle, bool *onDevice) {
    MXBONDCUDACONFIGHANDLE_GET(handle, bondcuda);
    MXCPTRCHECK(onDevice);
    *onDevice = bondcuda->onDevice();
    return S_OK;
}

HRESULT MxCBondCUDAConfig_getDevice(struct MxBondCUDAConfigHandle *handle, int *deviceId) {
    MXBONDCUDACONFIGHANDLE_GET(handle, bondcuda);
    MXCPTRCHECK(deviceId);
    *deviceId = bondcuda->getDevice();
    return S_OK;
}

HRESULT MxCBondCUDAConfig_setDevice(struct MxBondCUDAConfigHandle *handle, unsigned deviceId) {
    MXBONDCUDACONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->setDevice(deviceId);
}

HRESULT MxCBondCUDAConfig_toDevice(struct MxBondCUDAConfigHandle *handle) {
    MXBONDCUDACONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->toDevice();
}

HRESULT MxCBondCUDAConfig_fromDevice(struct MxBondCUDAConfigHandle *handle) {
    MXBONDCUDACONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->fromDevice();
}

HRESULT MxCBondCUDAConfig_setBlocks(struct MxBondCUDAConfigHandle *handle, unsigned int numBlocks) {
    MXBONDCUDACONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->setBlocks(numBlocks);
}

HRESULT MxCBondCUDAConfig_setThreads(struct MxBondCUDAConfigHandle *handle, unsigned int numThreads) {
    MXBONDCUDACONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->setThreads(numThreads);
}

HRESULT MxCBondCUDAConfig_refreshBond(struct MxBondCUDAConfigHandle *handle, struct MxBondHandleHandle *bh) {
    MXBONDCUDACONFIGHANDLE_GET(handle, bondcuda);
    MXCPTRCHECK(bh); MXCPTRCHECK(bh->MxObj);
    return bondcuda->refreshBond((MxBondHandle*)bh->MxObj);
}

HRESULT MxCBondCUDAConfig_refreshBonds(struct MxBondCUDAConfigHandle *handle, struct MxBondHandleHandle **bonds, unsigned int numBonds) {
    MXBONDCUDACONFIGHANDLE_GET(handle, bondcuda);
    MXCPTRCHECK(bonds);
    std::vector<MxBondHandle*> _bonds;
    MxBondHandleHandle *bh;
    for(unsigned int i = 0; i < numBonds; i++) {
        bh = bonds[i];
        MXCPTRCHECK(bh); MXCPTRCHECK(bh->MxObj);
        _bonds.push_back((MxBondHandle*)bh->MxObj);
    }
    return bondcuda->refreshBonds(_bonds);
}

HRESULT MxCBondCUDAConfig_refresh(struct MxBondCUDAConfigHandle *handle) {
    MXBONDCUDACONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->refresh();
}


///////////////////////
// MxAngleCUDAConfig //
///////////////////////


HRESULT MxCAngleCUDAConfig_onDevice(struct MxAngleCUDAConfigHandle *handle, bool *onDevice) {
    MXANGLECUDACONFIGHANDLE_GET(handle, bondcuda);
    MXCPTRCHECK(onDevice);
    *onDevice = bondcuda->onDevice();
    return S_OK;
}

HRESULT MxCAngleCUDAConfig_getDevice(struct MxAngleCUDAConfigHandle *handle, int *deviceId) {
    MXANGLECUDACONFIGHANDLE_GET(handle, bondcuda);
    MXCPTRCHECK(deviceId);
    *deviceId = bondcuda->getDevice();
    return S_OK;
}

HRESULT MxCAngleCUDAConfig_toDevice(struct MxAngleCUDAConfigHandle *handle) {
    MXANGLECUDACONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->toDevice();
}

HRESULT MxCAngleCUDAConfig_fromDevice(struct MxAngleCUDAConfigHandle *handle) {
    MXANGLECUDACONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->fromDevice();
}

HRESULT MxCAngleCUDAConfig_setBlocks(struct MxAngleCUDAConfigHandle *handle, unsigned int numBlocks) {
    MXANGLECUDACONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->setBlocks(numBlocks);
}

HRESULT MxCAngleCUDAConfig_setThreads(struct MxAngleCUDAConfigHandle *handle, unsigned int numThreads) {
    MXANGLECUDACONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->setThreads(numThreads);
}

HRESULT MxCAngleCUDAConfig_refreshAngle(struct MxAngleCUDAConfigHandle *handle, struct MxAngleHandleHandle *bh) {
    MXANGLECUDACONFIGHANDLE_GET(handle, bondcuda);
    MXCPTRCHECK(bh); MXCPTRCHECK(bh->MxObj);
    return bondcuda->refreshAngle((MxAngleHandle*)bh->MxObj);
}

HRESULT MxCAngleCUDAConfig_refreshAngles(struct MxAngleCUDAConfigHandle *handle, struct MxAngleHandleHandle **angles, unsigned int numAngles) {
    MXANGLECUDACONFIGHANDLE_GET(handle, bondcuda);
    MXCPTRCHECK(angles);
    std::vector<MxAngleHandle*> _angles;
    MxAngleHandleHandle *ah;
    for(unsigned int i = 0; i < numAngles; i++) {
        ah = angles[i];
        MXCPTRCHECK(ah); MXCPTRCHECK(ah->MxObj);
        _angles.push_back((MxAngleHandle*)ah->MxObj);
    }
    return bondcuda->refreshAngles(_angles);
}

HRESULT MxCAngleCUDAConfig_refresh(struct MxAngleCUDAConfigHandle *handle) {
    MXANGLECUDACONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->refresh();
}


///////////////////////////
// MxSimulatorCUDAConfig //
///////////////////////////


HRESULT MxCSimulator_getCUDAConfig(struct MxSimulatorCUDAConfigHandle *handle) {
    MXCPTRCHECK(handle);
    MxSimulatorCUDAConfig *simcuda = MxSimulator::getCUDAConfig();
    MXCPTRCHECK(simcuda);
    handle->MxObj = (void*)simcuda;
    return S_OK;
}

HRESULT MxCSimulatorCUDAConfig_getEngine(struct MxSimulatorCUDAConfigHandle *handle, struct MxEngineCUDAConfigHandle *itf) {
    MXSIMULATORCUDACONFIGHANDLE_GET(handle, simcuda);
    MXCPTRCHECK(itf);
    itf->MxObj = (void*)(&simcuda->engine);
    return S_OK;
}

HRESULT MxCSimulatorCUDAConfig_getBonds(struct MxSimulatorCUDAConfigHandle *handle, struct MxBondCUDAConfigHandle *itf) {
    MXSIMULATORCUDACONFIGHANDLE_GET(handle, simcuda);
    MXCPTRCHECK(itf);
    itf->MxObj = (void*)(&simcuda->bonds);
    return S_OK;
}

HRESULT MxCSimulatorCUDAConfig_getAngles(struct MxSimulatorCUDAConfigHandle *handle, struct MxAngleCUDAConfigHandle *itf) {
    MXSIMULATORCUDACONFIGHANDLE_GET(handle, simcuda);
    MXCPTRCHECK(itf);
    itf->MxObj = (void*)(&simcuda->angles);
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT MxCCUDAArchs(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(MX_CUDA_ARCHS, str, numChars);
}
