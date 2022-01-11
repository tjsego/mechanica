%{
    #include "MxSystem.h"

%}

%rename(_cpu_info) MxSystem::cpu_info();
%ignore MxSystem::cpuInfo;
%ignore MxSystem::compileFlags;
%ignore MxSystem::glInfo;
%ignore MxSystem::eglInfo;
%rename(_gl_info) MxSystem::gl_info();
%rename(_test_headless) MxSystem::test_headless();

%include "MxSystem.h"

%extend MxSystem {
    %pythoncode %{

        @staticmethod
        def cpu_info() -> dict:
            """Dictionary of CPU info"""
            return MxSystem._cpu_info().asdict()

        @staticmethod
        def compile_flags() -> dict:
            """Dictionary of compiler flags"""
            cf = MxCompileFlags()
            return {k: cf.getFlag(k) for k in cf.getFlags()}

        @staticmethod
        def gl_info() -> dict:
            """Dictionary of OpenGL info"""
            return MxSystem._gl_info().asdict()

        @staticmethod
        def test_headless() -> dict:
            return MxSystem._test_headless().asdict()
    %}
}

%pythoncode %{
    system = MxSystemPy
%}
