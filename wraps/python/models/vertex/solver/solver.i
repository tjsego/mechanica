%module vertex_solver

%{
    #include <models/vertex/solver/MxMeshObj.h>
    #include <models/vertex/solver/MxMeshLogger.h>
    #include <models/vertex/solver/MxMeshVertex.h>
    #include <models/vertex/solver/MxMeshSurface.h>
    #include <models/vertex/solver/MxMeshBody.h>
    #include <models/vertex/solver/MxMeshStructure.h>
    #include <models/vertex/solver/MxMesh.h>
    #include <models/vertex/solver/MxMeshSolver.h>
    #include <models/vertex/solver/MxMeshBind.h>
    #include <models/vertex/solver/actors/MxMeshVolumeConstraint.h>
    #include <models/vertex/solver/actors/MxMeshSurfaceAreaConstraint.h>
    #include <models/vertex/solver/actors/MxMeshBodyForce.h>
    #include <models/vertex/solver/actors/MxMeshSurfaceTraction.h>
    #include <models/vertex/solver/actors/MxMeshNormalStress.h>
%}


%template(vectorMxMeshVertex) std::vector<MxMeshVertex*>;
%template(vectorMxMeshSurface) std::vector<MxMeshSurface*>;
%template(vectorMxMeshBody) std::vector<MxMeshBody*>;
%template(vectorMxMeshStructure) std::vector<MxMeshStructure*>;
%template(vectorMxMesh) std::vector<MxMesh*>;

// todo: correct so that this block isn't necessary
%ignore MxMeshObjActor::energy(MxMeshObj *, MxMeshObj *, float &);
%ignore MxMeshObjActor::force(MxMeshObj *, MxMeshObj *, float *);
%ignore MxMeshVolumeConstraint::energy(MxMeshObj *, MxMeshObj *, float &);
%ignore MxMeshVolumeConstraint::force(MxMeshObj *, MxMeshObj *, float *);
%ignore MxMeshSurfaceAreaConstraint::energy(MxMeshObj *, MxMeshObj *, float &);
%ignore MxMeshSurfaceAreaConstraint::force(MxMeshObj *, MxMeshObj *, float *);
%ignore MxMeshBodyForce::energy(MxMeshObj *, MxMeshObj *, float &);
%ignore MxMeshBodyForce::force(MxMeshObj *, MxMeshObj *, float *);
%ignore MxMeshSurfaceTraction::energy(MxMeshObj *, MxMeshObj *, float &);
%ignore MxMeshSurfaceTraction::force(MxMeshObj *, MxMeshObj *, float *);
%ignore MxMeshNormalStress::energy(MxMeshObj *, MxMeshObj *, float &);
%ignore MxMeshNormalStress::force(MxMeshObj *, MxMeshObj *, float *);

%include <models/vertex/solver/MxMeshLogger.h>
%include <models/vertex/solver/MxMeshVertex.h>
%include <models/vertex/solver/MxMeshSurface.h>
%include <models/vertex/solver/MxMeshBody.h>
%include <models/vertex/solver/MxMeshStructure.h>
%include <models/vertex/solver/MxMesh.h>
%include <models/vertex/solver/MxMeshSolver.h>
%include <models/vertex/solver/MxMeshBind.h>
%include <models/vertex/solver/actors/MxMeshVolumeConstraint.h>
%include <models/vertex/solver/actors/MxMeshSurfaceAreaConstraint.h>
%include <models/vertex/solver/actors/MxMeshBodyForce.h>
%include <models/vertex/solver/actors/MxMeshSurfaceTraction.h>
%include <models/vertex/solver/actors/MxMeshNormalStress.h>
