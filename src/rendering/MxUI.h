/*
 * MxUI.h
 *
 *  Created on: Oct 6, 2018
 *      Author: andy
 */

#ifndef SRC_MXUI_H_
#define SRC_MXUI_H_

#include "mechanica_private.h"


CAPI_FUNC(HRESULT) MxUI_PollEvents();
CAPI_FUNC(HRESULT) MxUI_WaitEvents(double timeout);
CAPI_FUNC(HRESULT) MxUI_PostEmptyEvent();
CAPI_FUNC(HRESULT) MxUI_InitializeGraphics();
CAPI_FUNC(HRESULT) MxUI_CreateTestWindow();
CAPI_FUNC(HRESULT) MxUI_DestroyTestWindow();

#endif /* SRC_MXUI_H_ */
