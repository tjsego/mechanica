%{
    #include "rendering/MxColorMapper.hpp"

%}

%include "MxColorMapper.hpp"

%extend MxColorMapper {
    %pythoncode %{
        @property
        def names(self):
            return [x for x in MxColorMapper.getNames()]
    %}
}

%pythoncode %{
    ColorMapper = MxColorMapper
    ColorMap = MxColorMapper()
%}
