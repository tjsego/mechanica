/**
 * @file MxKeyEventPy.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxKeyEvent
 * @date 2022-03-23
 * 
 */

#include "MxKeyEventPy.h"


static MxKeyEventPyExecutor *staticMxKeyEventPyExecutor = NULL;

static HRESULT MxKeyEventPyExecutorHandler(struct MxKeyEvent *event) {
    if(staticMxKeyEventPyExecutor) return staticMxKeyEventPyExecutor->invoke(*event);
    return E_FAIL;
}

bool MxKeyEventPyExecutor::hasStaticMxKeyEventPyExecutor() {
    return staticMxKeyEventPyExecutor != NULL;
}

void MxKeyEventPyExecutor::setStaticMxKeyEventPyExecutor(MxKeyEventPyExecutor *executor) {
    staticMxKeyEventPyExecutor = executor;
    
    MxKeyEvent::addHandler(new MxKeyEventHandlerType(MxKeyEventPyExecutorHandler));
}

void MxKeyEventPyExecutor::maybeSetStaticMxKeyEventPyExecutor(MxKeyEventPyExecutor *executor) {
    if(!hasStaticMxKeyEventPyExecutor()) staticMxKeyEventPyExecutor = executor;
}

MxKeyEventPyExecutor *MxKeyEventPyExecutor::getStaticMxKeyEventPyExecutor() {
    return staticMxKeyEventPyExecutor;
}
