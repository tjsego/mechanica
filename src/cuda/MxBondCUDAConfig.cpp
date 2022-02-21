/**
 * @file MxBondCUDAConfig.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines CUDA runtime control interface for bonds
 * @date 2021-11-26
 * 
 */

#include "MxBondCUDAConfig.h"

#include <engine.h>
#include <MxLogger.h>


bool MxBondCUDAConfig::onDevice() {
    return _Engine.bonds_cuda;
}

int MxBondCUDAConfig::getDevice() {
    return MxBondCUDA_getDevice();
}

HRESULT MxBondCUDAConfig::setDevice(int deviceId) {
    if(MxBondCUDA_setDevice(&_Engine, deviceId) < 0) 
        return E_FAIL;
    return S_OK;
}

HRESULT MxBondCUDAConfig::toDevice() {
    if(MxBondCUDAConfig::onDevice()) {
        Log(LOG_DEBUG) << "Attempting send to device when already sent. Ignoring.";
        return S_OK;
    }

    if(MxBondCUDA_toDevice(&_Engine) < 0) { 
        Log(LOG_CRITICAL) << "Attempting send to device failed (" << engine_err << ").";
        return E_FAIL;
    }

    Log(LOG_INFORMATION) << "Successfully sent bonds to device";

    return S_OK;
}

HRESULT MxBondCUDAConfig::fromDevice() {
    if(!MxBondCUDAConfig::onDevice()) {
        Log(LOG_DEBUG) << "Attempting pull from device when not sent. Ignoring.";
        return S_OK;
    }

    if(MxBondCUDA_fromDevice(&_Engine) < 0) { 
        Log(LOG_CRITICAL) << "Attempting pull from device failed (" << engine_err << ").";
        return E_FAIL;
    }

    Log(LOG_INFORMATION) << "Successfully pulled bonds from device";
    
    return S_OK;
}

HRESULT MxBondCUDAConfig::setBlocks(unsigned int numBlocks) {
    if(MxBondCUDAConfig::onDevice()) 
        mx_error(E_FAIL, "Bonds already on device.");

    if(MxBondCUDA_setBlocks(numBlocks) < 0) 
        return E_FAIL;
    return S_OK;
}

HRESULT MxBondCUDAConfig::setThreads(unsigned int numThreads) {
    if(MxBondCUDAConfig::onDevice()) 
        mx_error(E_FAIL, "Bonds already on device.");

    if(MxBondCUDA_setThreads(numThreads) < 0) 
        return E_FAIL;
    return S_OK;
}

HRESULT MxBondCUDAConfig::refreshBond(MxBondHandle *bh) {
    if(!MxBondCUDAConfig::onDevice()) {
        Log(LOG_DEBUG) << "Attempting to refresh bonds when not on device. Ignoring.";
        return S_OK;
    }

    if(MxBondCUDA_refreshBond(&_Engine, bh) < 0) { 
        Log(LOG_CRITICAL) << "Refresh failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT MxBondCUDAConfig::refreshBonds(std::vector<MxBondHandle*> bonds) {
    if(!MxBondCUDAConfig::onDevice()) {
        Log(LOG_DEBUG) << "Attempting to refresh bonds when not on device. Ignoring.";
        return S_OK;
    }

    if(MxBondCUDA_refreshBonds(&_Engine, bonds.data(), bonds.size()) < 0) { 
        Log(LOG_CRITICAL) << "Refresh failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT MxBondCUDAConfig::refresh() {
    if(!MxBondCUDAConfig::onDevice()) {
        Log(LOG_DEBUG) << "Attempting to refresh bonds when not on device. Ignoring.";
        return S_OK;
    }

    if(MxBondCUDA_refresh(&_Engine) < 0) { 
        Log(LOG_CRITICAL) << "Refresh failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}
