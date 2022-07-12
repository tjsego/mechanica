/**
 * @file MxMeshBodyForce.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh body force
 * @date 2022-06-15
 * 
 */

#include "MxMeshBodyForce.h"

#include <models/vertex/solver/MxMeshVertex.h>
#include <models/vertex/solver/MxMeshBody.h>

#include <engine.h>


HRESULT MxMeshBodyForce::energy(MxMeshObj *source, MxMeshObj *target, float &e) {
    MxVector3f fv;
    force(source, target, fv.data());
    e = fv.dot(((MxMeshVertex*)target)->particle()->getVelocity()) * _Engine.dt;
    return S_OK;
}

HRESULT MxMeshBodyForce::force(MxMeshObj *source, MxMeshObj *target, float *f) {
    MxMeshBody *b = (MxMeshBody*)source;
    float bArea = b->getArea();
    if(bArea == 0.f) {
        return S_OK;
    }
    
    MxVector3f fv = comps * b->getVertexArea((MxMeshVertex*)target) / bArea;

    f[0] += fv[0];
    f[1] += fv[1];
    f[2] += fv[2];
    return S_OK;
}
