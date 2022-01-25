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
%rename(_screenshot) MxSystem::screenshot(const std::string&);
%rename(_screenshot2) MxSystem::screenshot(const std::string&, const bool &, const MxVector3f &);

%include "MxSystem.h"

%extend MxSystem {
    %pythoncode %{

        subrenderer = SubRendererFlags
        """Subrenderer flags"""

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

        @staticmethod
        def screenshot(filepath: str, decorate: bool = None, bgcolor=None):
            """
            Save a screenshot of the current scene

            :param filepath: path of file to save
            :type filepath: str
            :param decorate: flag to decorate the scene in the screenshot
            :type decorate: bool
            :param bgcolor: background color of the scene in the screenshot
            :type bgcolor: MxVector3f or [float, float, float] or (float, float, float) or float
            :rtype: int
            :return: HRESULT
            """    
            if decorate is None and bgcolor is None:
                return MxSystem._screenshot(filepath)

            if decorate is None:
                decorate = MxSystem.decorated()
            if bgcolor is None:
                bgcolor = MxSystem.getBackgroundColor()
            elif not isinstance(bgcolor, MxVector3f):
                bgcolor = MxVector3f(bgcolor)
            return MxSystem._screenshot2(filepath, decorate, bgcolor)

    %}
}

%pythoncode %{
    system = MxSystemPy
%}
