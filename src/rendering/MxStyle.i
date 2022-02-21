%{
    #include "rendering/MxStyle.hpp"

%}

%include "MxStyle.hpp"

%extend MxStyle {
    %pythoncode %{
        @property
        def visible(self) -> bool:
            """Visibility flag"""
            return self.getVisible()

        @visible.setter
        def visible(self, visible: bool):
            self.setVisible(visible)

        @property
        def colormap(self):
            """Style color map"""
            return self.getColorMap()

        @colormap.setter
        def colormap(self, colormap: str):
            return self.setColorMap(colormap)

        def __reduce__(self):
            return MxStyle.fromString, (self.toString(),)
    %}
}

%pythoncode %{
    Style = MxStyle
%}
