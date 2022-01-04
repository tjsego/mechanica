%{
    #include "io/Mx3DFEdgeData.h"

%}

%include "io/Mx3DFEdgeData.h"

%extend Mx3DFEdgeData{
    %pythoncode %{
        @property
        def vertices(self) -> vectorMx3DFVertexData_p:
            return self.getVertices()

        @property
        def faces(self) -> vectorMx3DFFaceData_p:
            return self.getFaces()

        @property
        def meshes(self) -> vectorMx3DFMeshData_p:
            return self.getMeshes()

        @property
        def num_vertices(self) -> int:
            return self.getNumVertices()

        @property
        def num_faces(self) -> int:
            return self.getNumFaces()

        @property
        def num_meshes(self) -> int:
            return self.getNumMeshes()
    %}
}

%pythoncode %{
    Edge3DFData = Mx3DFEdgeData
%}
