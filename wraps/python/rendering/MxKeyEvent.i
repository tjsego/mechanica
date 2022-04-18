%{
    #include <rendering/MxKeyEvent.hpp>
    #include <langs/py/MxKeyEventPy.h>

%}

MxEventPyExecutor_extender(MxKeyEventPyExecutor, MxKeyEvent)

%include <rendering/MxKeyEvent.hpp>
%include <langs/py/MxKeyEventPy.h>

%extend MxKeyEvent {
    %pythoncode %{
        @property
        def key_name(self) -> str:
            """Key pressed for this event"""
            return self.keyName()

        @property
        def key_alt(self) -> bool:
            """Flag for whether Alt key is pressed"""
            return self.keyAlt()

        @property
        def key_ctrl(self) -> bool:
            """Flag for whether Ctrl key is pressed"""
            return self.keyCtrl()

        @property
        def key_shift(self) -> bool:
            """Flag for whether Shift key is pressed"""
            return self.keyShift()
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
