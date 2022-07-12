/**
 * @file MxMeshBodyForce.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh body force
 * @date 2022-06-15
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_ACTORS_MXMESHBODYFORCE_H_
#define MODELS_VERTEX_SOLVER_ACTORS_MXMESHBODYFORCE_H_

#include <models/vertex/solver/MxMeshObj.h>

#include <types/mx_types.h>


struct MxMeshBodyForce : MxMeshObjActor {

    MxVector3f comps;

    MxMeshBodyForce(const MxVector3f &_force) {
        comps = _force;
    }

    HRESULT energy(MxMeshObj *source, MxMeshObj *target, float &e);

    HRESULT force(MxMeshObj *source, MxMeshObj *target, float *f);

};

#endif // MODELS_VERTEX_SOLVER_ACTORS_MXMESHBODYFORCE_H_