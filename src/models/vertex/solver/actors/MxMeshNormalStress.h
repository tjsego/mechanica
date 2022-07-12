/**
 * @file MxMeshNormalStress.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh surface normal force
 * @date 2022-06-15
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_ACTORS_MXMESHSURFACENORMAL_H_
#define MODELS_VERTEX_SOLVER_ACTORS_MXMESHSURFACENORMAL_H_

#include <models/vertex/solver/MxMeshObj.h>

#include <types/mx_types.h>


struct MxMeshNormalStress : MxMeshObjActor {

    float mag;

    MxMeshNormalStress(const float &_mag) {
        mag = _mag;
    }

    HRESULT energy(MxMeshObj *source, MxMeshObj *target, float &e);

    HRESULT force(MxMeshObj *source, MxMeshObj *target, float *f);

};

#endif // MODELS_VERTEX_SOLVER_ACTORS_MXMESHSURFACENORMAL_H_