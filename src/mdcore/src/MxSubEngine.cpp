/**
 * @file MxSubEngine.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines interface for solvers that can be injected into the Mechanica engine
 * @date 2022-03-15
 * 
 */

#include "MxSubEngine.h"

#include <engine.h>

HRESULT MxSubEngine::registerEngine() {
    for(auto &se : _Engine.subengines) 
        if(strcmp(this->name, se->name) == 0) 
            return engine_err_subengine;
    
    _Engine.subengines.push_back(this);
    return S_OK;
}
