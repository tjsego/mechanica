/**
 * @file MxMeshSurfaceTraction.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh surface traction
 * @date 2022-06-15
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_ACTORS_MXMESHSURFACETRACTION_H_
#define MODELS_VERTEX_SOLVER_ACTORS_MXMESHSURFACETRACTION_H_

#include <models/vertex/solver/MxMeshObj.h>

#include <types/mx_types.h>


struct MxMeshSurfaceTraction : MxMeshObjActor {

    MxVector3f comps;

    MxMeshSurfaceTraction(const MxVector3f &_force) {
        comps = _force;
    }

    HRESULT energy(MxMeshObj *source, MxMeshObj *target, float &e);

    HRESULT force(MxMeshObj *source, MxMeshObj *target, float *f);

};

#endif // MODELS_VERTEX_SOLVER_ACTORS_MXMESHSURFACETRACTION_H_