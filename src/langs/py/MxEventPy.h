/**
 * @file MxEventPy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxEvent
 * @date 2022-03-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXEVENTPY_H_
#define _SRC_LANGS_PY_MXEVENTPY_H_

#include "MxPy.h"

#include <event/MxEvent.h>
#include "MxEventPyExecutor.h"


struct MxEventPy;

struct MxEventPyPredicatePyExecutor : MxEventPyExecutor<MxEventPy> {
    HRESULT _result = 0;
};

struct MxEventPyInvokePyExecutor : MxEventPyExecutor<MxEventPy> {
    HRESULT _result = 0;
};

struct CAPI_EXPORT MxEventPy : MxEventBase {
    MxEventPy(MxEventPyInvokePyExecutor *invokeExecutor, MxEventPyPredicatePyExecutor *predicateExecutor=NULL);
    ~MxEventPy();

    HRESULT predicate();
    HRESULT invoke();
    HRESULT eval(const double &time);

private:
    MxEventPyInvokePyExecutor *invokeExecutor; 
    MxEventPyPredicatePyExecutor *predicateExecutor;
};

/**
 * @brief Creates an event using prescribed invoke and predicate python function executors
 * 
 * @param invokeExecutor an invoke python function executor; evaluated when an event occurs
 * @param predicateExecutor a predicate python function executor; evaluated to determine if an event occurs
 * @return MxEventPy* 
 */
CAPI_FUNC(MxEventPy*) MxOnEventPy(MxEventPyInvokePyExecutor *invokeExecutor, MxEventPyPredicatePyExecutor *predicateExecutor=NULL);

#endif // _SRC_LANGS_PY_MXEVENTPY_H_
