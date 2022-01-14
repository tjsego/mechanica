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
%ignore MxUniverseRenderer::cuboidInstanceBuffer;
%ignore MxUniverseRenderer::angleRenderer;
%ignore MxUniverseRenderer::bondRenderer;
%ignore MxUniverseRenderer::dihedralRenderer;
%ignore MxUniverseRenderer::arrowRenderer;

%include "MxUniverseRenderer.h"

%pythoncode %{
    UniverseRenderer = MxUniverseRenderer
%}
