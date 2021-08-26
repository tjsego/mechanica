%{
    #include "rendering/MxKeyEvent.hpp"

%}

MxEventPyExecutor_extender(MxKeyEventPyExecutor, MxKeyEvent)

%include "MxKeyEvent.hpp"

%extend MxKeyEvent {
    %pythoncode %{
        @property
        def key_name(self) -> str:
            """Key pressed for this event"""
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

    def on_keypress(invoke_method):
        """
        Registers a callback for handling keyboard events

        :type invoke_method: PyObject
        :param invoke_method: an invoke method; evaluated when an event occurs. 
            Takes an :class:`MxKeyEvent` instance as argument and returns None
        """
        return MxKeyEventPyExecutor.on_keypress(invoke_method)
%}
