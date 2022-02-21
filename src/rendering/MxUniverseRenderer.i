%{
    #include "rendering/MxUniverseRenderer.h"

%}

// problematic for wrapping
%ignore MxUniverseRenderer::gridMesh;
%ignore MxUniverseRenderer::sceneBox;
%ignore MxUniverseRenderer::sphereShader;
%ignore MxUniverseRenderer::flatShader;
%ignore MxUniverseRenderer::wireframeShader;
%ignore MxUniverseRenderer::sphereInstanceBuffer;
%ignore MxUniverseRenderer::largeSphereInstanceBuffer;
%ignore MxUniverseRenderer::sphereMesh;
%ignore MxUniverseRenderer::largeSphereMesh;
%ignore MxUniverseRenderer::cuboidMesh;
%ignore MxUniverseRenderer::discretizationGridMesh;
%ignore MxUniverseRenderer::cuboidInstanceBuffer;
%ignore MxUniverseRenderer::discretizationGridBuffer;
%ignore MxUniverseRenderer::subRenderers;

%include "MxUniverseRenderer.h"

%pythoncode %{
    UniverseRenderer = MxUniverseRenderer

    from enum import Enum as EnumPy

    class SubRendererFlags(EnumPy):
        """Flags for referencing subrenderers in select methods"""

        angle = SUBRENDERER_ANGLE
        """Angle subrenderer"""
        arrow = SUBRENDERER_ARROW
        """Arrow subrenderer"""
        bond = SUBRENDERER_BOND
        """Bond subrenderer"""
        dihedral = SUBRENDERER_DIHEDRAL
        """Dihedral subrenderer"""
%}
