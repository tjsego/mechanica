%{
    #include "rendering/MxKeyEvent.hpp"

%}

MxEventPyExecutor_extender(MxKeyEventPyExecutor, MxKeyEvent)

%include "MxKeyEvent.hpp"

%extend MxKeyEvent {
    %pythoncode %{
        @property
        def key_name(self) -> str:
            return self.keyName()
    %}
}

%extend MxKeyEventPyExecutor {
    %pythoncode %{
        @staticmethod
        def on_keypress(delegate):
            if MxKeyEventPyExecutor.hasStaticMxKeyEventPyExecutor():
                ex = MxKeyEventPyExecutor.getStaticMxKeyEventPyExecutor()
                cb = ex._callback
                def callback(e):
                    cb(e)
                    delegate(e)
            else:
                callback = delegate
            
            MxKeyEventPyExecutor.setStaticMxKeyEventPyExecutor(initMxKeyEventPyExecutor(callback))
    %}
}

%pythoncode %{
    KeyEvent = MxKeyEvent
    on_keypress = MxKeyEventPyExecutor.on_keypress
%}
