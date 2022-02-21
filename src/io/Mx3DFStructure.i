%{
    #include "io/Mx3DFStructure.h"

%}

%include "io/Mx3DFStructure.h"

%extend Mx3DFStructure{
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
        def meshes(self) -> vectorMx3DFMeshData_p:
            return self.getMeshes()

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
        def num_meshes(self) -> int:
            return self.getNumMeshes()

        @property
        def centroid(self) -> MxVector3f:
            return self.getCentroid()
    %}
}

%pythoncode %{
    Structure3DF = Mx3DFStructure
%}
