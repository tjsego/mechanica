/**
 * @file MxMeshSurfaceAreaConstraint.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh surface area constraint
 * @date 2022-06-15
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_ACTORS_MXMESHSURFACEAREACONSTRAINT_H_
#define MODELS_VERTEX_SOLVER_ACTORS_MXMESHSURFACEAREACONSTRAINT_H_

#include <models/vertex/solver/MxMeshObj.h>


struct MxMeshSurfaceAreaConstraint : MxMeshObjActor {

    float lam;
    float constr;

    MxMeshSurfaceAreaConstraint(const float &_lam, const float &_constr) {
        lam = _lam;
        constr = _constr;
    }

    HRESULT energy(MxMeshObj *source, MxMeshObj *target, float &e);

    HRESULT force(MxMeshObj *source, MxMeshObj *target, float *f);
};

#endif // MODELS_VERTEX_SOLVER_ACTORS_MXMESHSURFACEAREACONSTRAINT_H_