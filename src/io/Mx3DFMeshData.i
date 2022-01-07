%{
    #include "io/Mx3DFMeshData.h"

%}

%include "io/Mx3DFMeshData.h"

%extend Mx3DFMeshData{
    %pythoncode %{
        @property
        def vertices(self) -> vectorMx3DFVertexData_p:
            return self.getVertices()

        @property
        def edges(self) -> vectorMx3DFEdgeData_p:
            return self.getEdges()

        @property
        def faces(self) -> vectorMx3DFFaceData_p:
            return self.getFaces()

        @property
        def num_vertices(self) -> int:
            return self.getNumVertices()

        @property
        def num_edges(self) -> int:
            return self.getNumEdges()

        @property
        def num_faces(self) -> int:
            return self.getNumFaces()

        @property
        def centroid(self) -> MxVector3f:
            return self.getCentroid()
    %}
}

%pythoncode %{
    Mesh3DFData = Mx3DFMeshData
%}
