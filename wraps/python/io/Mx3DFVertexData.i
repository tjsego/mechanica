%{
    #include "io/Mx3DFVertexData.h"

%}

%include "io/Mx3DFVertexData.h"

%extend Mx3DFVertexData{
    %pythoncode %{
        def is_in(self, *args, **kwargs) -> bool:
            """Alias of :meth:_in"""
            
            return self._in(*args, **kwargs)
        
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
        def num_edges(self) -> int:
            return self.getNumEdges()

        @property
        def num_faces(self) -> int:
            return self.getNumFaces()

        @property
        def num_meshes(self) -> int:
            return self.getNumMeshes()
    %}
}

%pythoncode %{
    Vertex3DFData = Mx3DFVertexData
%}
