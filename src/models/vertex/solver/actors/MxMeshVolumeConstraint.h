/**
 * @file MxMeshVolumeConstraint.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh volume constraint
 * @date 2022-06-15
 * 
 */

#ifndef MODELS_VERTEX_SOLVER_ACTORS_MXMESHVOLUMECONSTRAINT_H_
#define MODELS_VERTEX_SOLVER_ACTORS_MXMESHVOLUMECONSTRAINT_H_

#include <models/vertex/solver/MxMeshObj.h>


struct MxMeshVolumeConstraint : MxMeshObjActor {

    float lam;
    float constr;

    MxMeshVolumeConstraint(const float &_lam, const float &_constr) {
        lam = _lam;
        constr = _constr;
    }

    HRESULT energy(MxMeshObj *source, MxMeshObj *target, float &e);

    HRESULT force(MxMeshObj *source, MxMeshObj *target, float *f);

};

#endif // MODELS_VERTEX_SOLVER_ACTORS_MXMESHVOLUMECONSTRAINT_H_