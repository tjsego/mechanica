/**
 * @file MxKeyEventPy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxKeyEvent
 * @date 2022-03-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXKEYEVENTPY_H_
#define _SRC_LANGS_PY_MXKEYEVENTPY_H_

#include "MxPy.h"

#include "MxEventPyExecutor.h"

#include <rendering/MxKeyEvent.hpp>


struct CAPI_EXPORT MxKeyEventPyExecutor : MxEventPyExecutor<MxKeyEvent> {
    static bool hasStaticMxKeyEventPyExecutor();
    static void setStaticMxKeyEventPyExecutor(MxKeyEventPyExecutor *executor);
    static void maybeSetStaticMxKeyEventPyExecutor(MxKeyEventPyExecutor *executor);
    static MxKeyEventPyExecutor *getStaticMxKeyEventPyExecutor();
};

#endif // _SRC_LANGS_PY_MXKEYEVENTPY_H_
