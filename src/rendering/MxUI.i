%{
    #include "rendering/MxUI.h"

%}

%include "MxUI.h"

%pythoncode %{
    pollEvents = MxUI_PollEvents
    waitEvents = MxUI_WaitEvents
    postEmptyEvent = MxUI_PostEmptyEvent
    initializeGraphics = MxUI_InitializeGraphics
    createTestWindow = MxUI_CreateTestWindow
    destroyTestWindow = MxUI_DestroyTestWindow
%}
