/**
 * @file MxMeshLogger.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh solver logger
 * @date 2022-06-30
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_MXMESHLOGGER_H_
#define MODELS_VERTEX_SOLVER_MXMESHLOGGER_H_

#include <mx_port.h>

#include "MxMeshObj.h"

#include <vector>

enum MxMeshLogEventType {
    MXMESHLOGEVENT_NONE = 0,
    MXMESHLOGEVENT_CREATE,
    MXMESHLOGEVENT_DESTROY
};

struct CAPI_EXPORT MxMeshLogEvent {
    MxMeshLogEventType type;
    std::vector<int> objIDs;
    std::vector<MxMeshObj::Type> objTypes;
    int meshID;
};

struct CAPI_EXPORT MxMeshLogger {

    static HRESULT clear();

    static HRESULT log(const MxMeshLogEvent &event);

    static std::vector<MxMeshLogEvent> events();

};

#endif // MODELS_VERTEX_SOLVER_MXMESHLOGGER_H_