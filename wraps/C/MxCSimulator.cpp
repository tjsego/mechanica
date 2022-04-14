/**
 * @file MxCSimulator.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxSimulator
 * @date 2022-03-24
 */

#include "MxCSimulator.h"

#include "mechanica_c_private.h"
#include "MxCUniverse.h"

#include <MxSimulator.h>


namespace mx {

MxSimulator_Config *cast(struct MxSimulator_ConfigHandle *handle) {
    return castC<MxSimulator_Config, MxSimulator_ConfigHandle>(handle);
}

MxSimulator *cast(struct MxSimulatorHandle *handle) {
    return castC<MxSimulator, MxSimulatorHandle>(handle);
}

}

#define MXSIMCONFIG_GET(handle) \
    MxSimulator_Config *conf = mx::castC<MxSimulator_Config, MxSimulator_ConfigHandle>(handle); \
    MXCPTRCHECK(conf);

#define MXSIM_GET(handle) \
    MxSimulator *sim = mx::castC<MxSimulator, MxSimulatorHandle>(handle); \
    MXCPTRCHECK(sim);

//////////////////////////////////
// MxSimulator_EngineIntegrator //
//////////////////////////////////

HRESULT MxCSimulator_EngineIntegrator_init(struct MxSimulator_EngineIntegratorHandle *handle) {
    MXCPTRCHECK(handle);
    handle->FORWARD_EULER = (int)MxSimulator_EngineIntegrator::FORWARD_EULER;
    handle->RUNGE_KUTTA_4 = (int)MxSimulator_EngineIntegrator::RUNGE_KUTTA_4;
    return S_OK;
}

//////////////////////////////////
// MxSimulator_DpiScalingPolicy //
//////////////////////////////////

HRESULT MxCSimulator_DpiScalingPolicy_init(struct MxSimulator_DpiScalingPolicyHandle *handle) {
    MXCPTRCHECK(handle);
    handle->MXSIMULATOR_NONE = (int)MXSIMULATOR_NONE;
    handle->MXSIMULATOR_WINDOWLESS = (int)MXSIMULATOR_WINDOWLESS;
    handle->MXSIMULATOR_GLFW = (int)MXSIMULATOR_GLFW;
    return S_OK;
}

////////////////////////
// MxSimulator_Config //
////////////////////////

HRESULT MxCSimulator_Config_init(struct MxSimulator_ConfigHandle *handle) {
    MXCPTRCHECK(handle);
    handle->MxObj = new MxSimulator_Config();
    return S_OK;
}

HRESULT MxCSimulator_Config_getTitle(struct MxSimulator_ConfigHandle *handle, char **title, unsigned int *numChars) {
    MXSIMCONFIG_GET(handle)
    return mx::capi::str2Char(conf->title(), title, numChars);
}

HRESULT MxCSimulator_Config_setTitle(struct MxSimulator_ConfigHandle *handle, const char *title) {
    MXSIMCONFIG_GET(handle)
    conf->setTitle(title);
    return S_OK;
}

HRESULT MxCSimulator_Config_getWindowSize(struct MxSimulator_ConfigHandle *handle, unsigned int *x, unsigned int *y) {
    MXSIMCONFIG_GET(handle)
    MXCPTRCHECK(x);
    MXCPTRCHECK(y);
    auto ws = conf->windowSize();
    *x = ws.x();
    *y = ws.y();
    return S_OK;
}

HRESULT MxCSimulator_Config_setWindowSize(struct MxSimulator_ConfigHandle *handle, unsigned int x, unsigned int y) {
    MXSIMCONFIG_GET(handle)
    conf->setWindowSize(MxVector2i(x, y));
    return S_OK;
}

HRESULT MxCSimulator_Config_getSeed(struct MxSimulator_ConfigHandle *handle, unsigned int *seed) {
    MXSIMCONFIG_GET(handle)
    MXCPTRCHECK(seed);
    unsigned int *_seed = conf->seed();
    MXCPTRCHECK(_seed);
    *seed = *_seed;
    return S_OK;
}

HRESULT MxCSimulator_Config_setSeed(struct MxSimulator_ConfigHandle *handle, unsigned int seed) {
    MXSIMCONFIG_GET(handle)
    conf->setSeed(seed);
    return S_OK;
}

HRESULT MxCSimulator_Config_getWindowless(struct MxSimulator_ConfigHandle *handle, bool *windowless) {
    MXSIMCONFIG_GET(handle);
    MXCPTRCHECK(windowless);
    *windowless = conf->windowless();
    return S_OK;
}

HRESULT MxCSimulator_Config_setWindowless(struct MxSimulator_ConfigHandle *handle, bool windowless) {
    MXSIMCONFIG_GET(handle);
    conf->setWindowless(windowless);
    return S_OK;
}

HRESULT MxCSimulator_Config_getImportDataFilePath(struct MxSimulator_ConfigHandle *handle, char **filePath, unsigned int *numChars) {
    MXSIMCONFIG_GET(handle)
    std::string *fp = conf->importDataFilePath;
    if(!fp) {
        numChars = 0;
        return S_OK;
    }
    else return mx::capi::str2Char(*fp, filePath, numChars);
}

HRESULT MxCSimulator_Config_getClipPlanes(struct MxSimulator_ConfigHandle *handle, float **clipPlanes, unsigned int *numClipPlanes) {
    MXSIMCONFIG_GET(handle)
    MXCPTRCHECK(clipPlanes);
    MXCPTRCHECK(numClipPlanes);
    *numClipPlanes = conf->clipPlanes.size();
    if(*numClipPlanes > 0) {
        float *cps = (float*)malloc(*numClipPlanes * 4 * sizeof(float));
        if(!cps) 
            return E_OUTOFMEMORY;
        for(unsigned int i = 0; i < *numClipPlanes; i++) {
            auto _cp = conf->clipPlanes[i];
            for(unsigned int j = 0; j < 4; j++) 
                cps[4 * i + j] = _cp[j];
        }
        *clipPlanes = cps;
    }
    return S_OK;
}

HRESULT MxCSimulator_Config_setClipPlanes(struct MxSimulator_ConfigHandle *handle, float *clipPlanes, unsigned int numClipPlanes) {
    MXSIMCONFIG_GET(handle);
    MXCPTRCHECK(clipPlanes);
    conf->clipPlanes.clear();
    for(unsigned int i = 0; i < numClipPlanes; i++) {
        float *b = &clipPlanes[4 * i];
        conf->clipPlanes.push_back(MxVector4f(b[0], b[1], b[2], b[3]));
    }
    return S_OK;
}

HRESULT MxCSimulator_Config_getUniverseConfig(struct MxSimulator_ConfigHandle *handle, struct MxUniverseConfigHandle *confHandle) {
    MXSIMCONFIG_GET(handle)
    MXCPTRCHECK(confHandle);
    confHandle->MxObj = (void*)&conf->universeConfig;
    return S_OK;
}

HRESULT MxCSimulator_Config_destroy(struct MxSimulator_ConfigHandle *handle) {
    return mx::capi::destroyHandle<MxSimulator_Config, MxSimulator_ConfigHandle>(handle) ? S_OK : E_FAIL;
}


/////////////////
// MxSimulator //
/////////////////


HRESULT MxCSimulator_init(const char **argv, unsigned int nargs) {

    return MxSimulator_init(mx::capi::charA2StrV(argv, nargs));
}

HRESULT MxCSimulator_initC(struct MxSimulator_ConfigHandle *handle, const char **appArgv, unsigned int nargs) {

    MXSIMCONFIG_GET(handle)

    return MxSimulator_initC(*conf, mx::capi::charA2StrV(appArgv, nargs));
}

HRESULT MxCSimulator_get(struct MxSimulatorHandle *handle) {
    MxSimulator *sim = MxSimulator::get();
    if(!sim) 
        return E_FAIL;
    handle->MxObj = sim;
    return S_OK;
}

HRESULT MxCSimulator_makeCurrent(struct MxSimulatorHandle *handle) {
    MXSIM_GET(handle)
    return sim->makeCurrent();
}

HRESULT MxCSimulator_run(double et) {
    MxSimulator *sim = MxSimulator::get();
    MXCPTRCHECK(sim);
    return sim->run(et);
}

HRESULT MxCSimulator_show() {
    MxSimulator *sim = MxSimulator::get();
    MXCPTRCHECK(sim);
    return sim->show();
}

HRESULT MxCSimulator_close() {
    MxSimulator *sim = MxSimulator::get();
    MXCPTRCHECK(sim);
    return sim->close();
}

HRESULT MxCSimulator_destroy() {
    MxSimulator *sim = MxSimulator::get();
    MXCPTRCHECK(sim);
    return sim->destroy();
}

HRESULT MxCSimulator_redraw() {
    MxSimulator *sim = MxSimulator::get();
    MXCPTRCHECK(sim);
    return sim->redraw();
}

HRESULT MxCSimulator_getNumThreads(unsigned int *numThreads) {
    MxSimulator *sim = MxSimulator::get();
    MXCPTRCHECK(sim);
    MXCPTRCHECK(numThreads);
    *numThreads = sim->getNumThreads();
    return S_OK;
}

bool MxC_TerminalInteractiveShell() {
    return Mx_TerminalInteractiveShell();
}

HRESULT MxC_setTerminalInteractiveShell(bool _interactive) {
    return Mx_setTerminalInteractiveShell(_interactive);
}