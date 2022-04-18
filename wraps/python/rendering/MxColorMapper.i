%{
    #include <rendering/MxColorMapper.hpp>

%}

%include <rendering/MxColorMapper.hpp>

%extend MxColorMapper {
    %pythoncode %{
        @property
        def names(self):
            """Names of available color maps"""
            return [x for x in MxColorMapper.getNames()]
    %}
}

%pythoncode %{
    ColorMapper = MxColorMapper
    ColorMap = MxColorMapper()
%}
