/**
 * @file MxAngleCUDAConfig.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines CUDA runtime control interface for angles
 * @date 2021-11-30
 * 
 */

#include "MxAngleCUDAConfig.h"

#include <engine.h>
#include <MxLogger.h>


bool MxAngleCUDAConfig::onDevice() {
    return _Engine.angles_cuda;
}

int MxAngleCUDAConfig::getDevice() {
    return MxAngleCUDA_getDevice();
}

HRESULT MxAngleCUDAConfig::toDevice() {
    if(MxAngleCUDAConfig::onDevice()) {
        Log(LOG_DEBUG) << "Attempting send to device when already sent. Ignoring.";
        return S_OK;
    }

    if(MxAngleCUDA_toDevice(&_Engine) < 0) { 
        Log(LOG_CRITICAL) << "Attempting send to device failed (" << engine_err << ").";
        return E_FAIL;
    }

    Log(LOG_INFORMATION) << "Successfully sent angles to device";

    return S_OK;
}

HRESULT MxAngleCUDAConfig::fromDevice() {
    if(!MxAngleCUDAConfig::onDevice()) {
        Log(LOG_DEBUG) << "Attempting pull from device when not sent. Ignoring.";
        return S_OK;
    }

    if(MxAngleCUDA_fromDevice(&_Engine) < 0) { 
        Log(LOG_CRITICAL) << "Attempting pull from device failed (" << engine_err << ").";
        return E_FAIL;
    }

    Log(LOG_INFORMATION) << "Successfully pulled angles from device";
    
    return S_OK;
}

HRESULT MxAngleCUDAConfig::setBlocks(unsigned int numBlocks) {
    if(MxAngleCUDAConfig::onDevice()) 
        mx_error(E_FAIL, "Angles already on device.");

    if(MxAngleCUDA_setBlocks(numBlocks) < 0) 
        return E_FAIL;
    return S_OK;
}

HRESULT MxAngleCUDAConfig::setThreads(unsigned int numThreads) {
    if(MxAngleCUDAConfig::onDevice()) 
        mx_error(E_FAIL, "Angles already on device.");

    if(MxAngleCUDA_setThreads(numThreads) < 0) 
        return E_FAIL;
    return S_OK;
}

HRESULT MxAngleCUDAConfig::refreshAngle(MxAngleHandle *bh) {
    if(!MxAngleCUDAConfig::onDevice()) {
        Log(LOG_DEBUG) << "Attempting to refresh angles when not on device. Ignoring.";
        return S_OK;
    }

    if(MxAngleCUDA_refreshAngle(&_Engine, bh) < 0) { 
        Log(LOG_CRITICAL) << "Refresh failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT MxAngleCUDAConfig::refreshAngles(std::vector<MxAngleHandle*> angles) {
    if(!MxAngleCUDAConfig::onDevice()) {
        Log(LOG_DEBUG) << "Attempting to refresh angles when not on device. Ignoring.";
        return S_OK;
    }

    if(MxAngleCUDA_refreshAngles(&_Engine, angles.data(), angles.size()) < 0) { 
        Log(LOG_CRITICAL) << "Refresh failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT MxAngleCUDAConfig::refresh() {
    if(!MxAngleCUDAConfig::onDevice()) {
        Log(LOG_DEBUG) << "Attempting to refresh angles when not on device. Ignoring.";
        return S_OK;
    }

    if(MxAngleCUDA_refresh(&_Engine) < 0) { 
        Log(LOG_CRITICAL) << "Refresh failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}
