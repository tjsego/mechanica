/**
 * @file MxMeshNormalStress.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh surface normal force
 * @date 2022-06-15
 * 
 */

#include "MxMeshNormalStress.h"

#include <models/vertex/solver/MxMeshVertex.h>
#include <models/vertex/solver/MxMeshSurface.h>
#include <models/vertex/solver/MxMeshBody.h>

#include <engine.h>

HRESULT MxMeshNormalStress::energy(MxMeshObj *source, MxMeshObj *target, float &e) {
    MxVector3f fv;
    force(source, target, fv.data());
    e = fv.dot(((MxMeshVertex*)target)->particle()->getVelocity()) * _Engine.dt;
    return S_OK;
}

HRESULT MxMeshNormalStress::force(MxMeshObj *source, MxMeshObj *target, float *f) {
    MxMeshSurface *s = (MxMeshSurface*)source;
    MxMeshVertex *v = (MxMeshVertex*)target;
    
    auto bodies = s->getBodies();
    if(bodies.size() == 2) 
        return S_OK;
    
    MxVector3f snormal = s->getNormal().normalized();
    if(bodies.size() == 1) 
        snormal *= s->volumeSense(bodies[0]);

    MxVector3f vForce = snormal * mag * s->getVertexArea(v);
    f[0] += vForce[0];
    f[1] += vForce[1];
    f[2] += vForce[2];
    return S_OK;
}
