%{
    #include "io/Mx3DFVertexData.h"
    #include "io/Mx3DFEdgeData.h"
    #include "io/Mx3DFFaceData.h"
    #include "io/Mx3DFMeshData.h"

%}

// Declared here to make good use to type hints

%template(vectorMx3DFVertexData_p) std::vector<Mx3DFVertexData*>;
%template(vectorMx3DFEdgeData_p)   std::vector<Mx3DFEdgeData*>;
%template(vectorMx3DFFaceData_p)   std::vector<Mx3DFFaceData*>;
%template(vectorMx3DFMeshData_p)   std::vector<Mx3DFMeshData*>;

%include "Mx3DFVertexData.i"
%include "Mx3DFEdgeData.i"
%include "Mx3DFFaceData.i"
%include "Mx3DFMeshData.i"
%include "Mx3DFStructure.i"

%{
    #include "io/Mx3DFRenderData.h"
    #include "io/MxIO.h"
    #include "io/Mx3DFIO.h"

%}

%include "io/Mx3DFRenderData.h"
%include "io/MxIO.h"
%include "io/Mx3DFIO.h"

%pythoncode %{
    io = MxIO
%}
