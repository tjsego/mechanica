/**
 * @file MxEngineCUDAConfig.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines CUDA runtime control interface for engine
 * @date 2021-11-10
 * 
 */
#include "MxEngineCUDAConfig.h"

#include <engine.h>
#include <MxLogger.h>


MxEngineCUDAConfig::MxEngineCUDAConfig() : 
    on_device{false}
{}

bool MxEngineCUDAConfig::onDevice() {
    return this->on_device;
}

int MxEngineCUDAConfig::getDevice() {
    if(_Engine.nr_devices == 0) return -1;
    return _Engine.devices[0];
}

HRESULT MxEngineCUDAConfig::setDevice(int deviceId) {
    if(this->onDevice()) mx_error(E_FAIL, "Engine already on device");

    if(engine_cuda_setdevice(&_Engine, deviceId) < 0)
        return E_FAIL;

    return S_OK;
}

HRESULT MxEngineCUDAConfig::clearDevice() {
    if(this->onDevice()) mx_error(E_FAIL, "Engine on device");

    if(engine_cuda_cleardevices(&_Engine) < 0)
        return E_FAIL;

    return S_OK;
}

HRESULT MxEngineCUDAConfig::toDevice() {
    if(this->onDevice()) {
        Log(LOG_DEBUG) << "Attempting send to device when already sent. Ignoring.";
        return S_OK;
    }

    if(_Engine.nr_devices == 0) this->setDevice();
    if(engine_toCUDA(&_Engine) < 0) {
        Log(LOG_CRITICAL) << "Attempting send to device failed (" << engine_err << ").";
        return E_FAIL;
    }

    Log(LOG_INFORMATION) << "Successfully sent engine to device";
    this->on_device = true;

    return S_OK;
}

HRESULT MxEngineCUDAConfig::fromDevice() {
    if(!this->onDevice()) {
        Log(LOG_DEBUG) << "Attempting pull from device when not sent. Ignoring.";
        return S_OK;
    }

    if(engine_fromCUDA(&_Engine) < 0) {
        Log(LOG_CRITICAL) << "Attempting pull from device failed (" << engine_err << ").";
        return E_FAIL;
    }

    Log(LOG_INFORMATION) << "Successfully pulled engine from device";
    this->on_device = false;

    return S_OK;
}

HRESULT MxEngineCUDAConfig::setBlocks(unsigned int numBlocks, int deviceId) {
    if(this->onDevice()) {
        mx_error(E_FAIL, "Engine already on device.");
    }
    
    if(_Engine.nr_devices == 0) this->setDevice();
    if(deviceId < 0) deviceId = _Engine.devices[0];
    
    if(engine_cuda_setblocks(&_Engine, deviceId, numBlocks) < 0)
        return E_FAIL;

    return S_OK;
}

HRESULT MxEngineCUDAConfig::setThreads(unsigned int numThreads, int deviceId) {
    if(this->onDevice()) {
        mx_error(E_FAIL, "Engine already on device.");
    }

    if(_Engine.nr_devices == 0) this->setDevice();
    if(deviceId < 0) deviceId = _Engine.devices[0];

    if(engine_cuda_setthreads(&_Engine, deviceId, numThreads) < 0)
        return E_FAIL;

    return S_OK;
}

HRESULT MxEngineCUDAConfig::refreshPotentials() {
    if(!this->onDevice()) {
        Log(LOG_DEBUG) << "Attempting to refresh potentials when not on device. Ignoring.";
        return S_OK;
    }

    if(engine_cuda_refresh_pots(&_Engine) < 0) {
        Log(LOG_CRITICAL) << "Attempting to refresh potentials failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT MxEngineCUDAConfig::refreshFluxes() {
    if(!this->onDevice()) {
        Log(LOG_DEBUG) << "Attempting to refresh fluxes when not on device. Ignoring.";
        return S_OK;
    }

    if(engine_cuda_refresh_fluxes(&_Engine) < 0) {
        Log(LOG_CRITICAL) << "Attempting to refresh fluxes failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT MxEngineCUDAConfig::refreshBoundaryConditions() {
    if(!this->onDevice()) {
        Log(LOG_DEBUG) << "Attempting to refresh boundary conditions when not on device. Ignoring.";
        return S_OK;
    }

    if(engine_cuda_boundary_conditions_refresh(&_Engine) < 0) {
        Log(LOG_CRITICAL) << "Attempting to refresh boundary conditions failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT MxEngineCUDAConfig::refresh() {
    if(!this->onDevice()) {
        Log(LOG_DEBUG) << "Attempting to refresh engine when not on device. Ignoring.";
        return S_OK;
    }

    if(engine_cuda_refresh(&_Engine) < 0) {
        Log(LOG_CRITICAL) << "Attempting to refresh engine failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT MxEngineCUDAConfig::setSeed(const unsigned int seed) {
    if(engine_cuda_rand_norm_setSeed(&_Engine, seed, this->onDevice()) < 0) {
        Log(LOG_CRITICAL) << "Attempting to seed engine seed failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

unsigned int MxEngineCUDAConfig::getSeed() {
    return _Engine.rand_norm_seed_cuda;
}
