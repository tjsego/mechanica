%{
    #include "io/Mx3DFFaceData.h"

%}

%include "io/Mx3DFFaceData.h"

%extend Mx3DFFaceData{
    %pythoncode %{
        @property
        def vertices(self) -> vectorMx3DFVertexData_p:
            return self.getVertices()

        @property
        def edges(self) -> vectorMx3DFEdgeData_p:
            return self.getEdges()

        @property
        def meshes(self) -> vectorMx3DFMeshData_p:
            return self.getMeshes()

        @property
        def num_vertices(self) -> int:
            return self.getNumVertices()

        @property
        def num_edges(self) -> int:
            return self.getNumEdges()

        @property
        def num_meshes(self) -> int:
            return self.getNumMeshes()
    %}
}

%pythoncode %{
    Face3DFData = Mx3DFFaceData
%}
