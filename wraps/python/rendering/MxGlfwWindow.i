%{
    #include <rendering/MxGlfwWindow.h>

%}

%include <rendering/MxGlfwWindow.h>

%extend MxGlfwWindow {
    %pythoncode %{
        @property
        def field(self) -> float:
            return self.getField()

        @field.setter
        def field(self, field: float):
            self.setField(field)
    %}
}

%pythoncode %{
    Window = MxGlfwWindow
%}
