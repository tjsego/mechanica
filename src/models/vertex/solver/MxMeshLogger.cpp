/**
 * @file MxMeshLogger.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh solver logger
 * @date 2022-06-30
 * 
 */

#include "MxMeshLogger.h"


static std::vector<MxMeshLogEvent> logEvents;


HRESULT MxMeshLogger::clear() {
    logEvents.clear();
    return S_OK;
}

HRESULT MxMeshLogger::log(const MxMeshLogEvent &event) {
    logEvents.push_back(event);
    return S_OK;
}

std::vector<MxMeshLogEvent> MxMeshLogger::events() {
    return logEvents;
}
