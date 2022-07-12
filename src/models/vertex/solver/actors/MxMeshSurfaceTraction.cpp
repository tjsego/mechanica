/**
 * @file MxMeshSurfaceTraction.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh surface traction
 * @date 2022-06-15
 * 
 */

#include "MxMeshSurfaceTraction.h"

#include <models/vertex/solver/MxMeshVertex.h>
#include <models/vertex/solver/MxMeshSurface.h>

#include <engine.h>

HRESULT MxMeshSurfaceTraction::energy(MxMeshObj *source, MxMeshObj *target, float &e) {
    MxVector3f fv;
    force(source, target, fv.data());
    e = fv.dot(((MxMeshVertex*)target)->particle()->getVelocity()) * _Engine.dt;
    return S_OK;
}

HRESULT MxMeshSurfaceTraction::force(MxMeshObj *source, MxMeshObj *target, float *f) {
    MxMeshSurface *s = (MxMeshSurface*)source;
    MxMeshVertex *v = (MxMeshVertex*)target;
    MxVector3f vForce = comps * s->getVertexArea(v);
    f[0] += vForce[0];
    f[1] += vForce[1];
    f[2] += vForce[2];
    return S_OK;
}
