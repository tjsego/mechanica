/**
 * @file MxMeshVolumeConstraint.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh volume constraint
 * @date 2022-06-15
 * 
 */

#include "MxMeshVolumeConstraint.h"

#include <models/vertex/solver/MxMeshVertex.h>
#include <models/vertex/solver/MxMeshSurface.h>
#include <models/vertex/solver/MxMeshBody.h>


HRESULT MxMeshVolumeConstraint::energy(MxMeshObj *source, MxMeshObj *target, float &e) {
    MxMeshBody *b = (MxMeshBody*)source;
    float dvol = b->getVolume() - constr;
    e = lam * dvol * dvol;
    return S_OK;
}

HRESULT MxMeshVolumeConstraint::force(MxMeshObj *source, MxMeshObj *target, float *f) {
    MxMeshBody *b = (MxMeshBody*)source;
    MxMeshVertex *v = (MxMeshVertex*)target;
    
    MxVector3f posc = v->getPosition();
    MxVector3f ftotal(0.f);

    for(auto &s : v->getSurfaces()) {
        if(!s->in(b)) 
            continue;
        
        auto svertices = s->getVertices();
        auto nbs_verts = s->neighborVertices(v);

        MxVector3f sftotal = Magnum::Math::cross(s->getCentroid(), nbs_verts[0]->getPosition() - nbs_verts[1]->getPosition());
        for(unsigned int i = 0; i < svertices.size(); i++) {
            sftotal -= s->triangleNormal(i) / svertices.size();
        }

        ftotal += sftotal * s->volumeSense(b);
    }
    
    // float pressure = 2 * lam * (constr - b->getVolume());
    // float fmag = - pressure / 6.f;
    // float fmag = lam * (constr - b->getVolume()) / 3.f;
    ftotal *= (lam * (b->getVolume() - constr) / 3.f);

    f[0] += ftotal[0];
    f[1] += ftotal[1];
    f[2] += ftotal[2];

    return S_OK;
}
