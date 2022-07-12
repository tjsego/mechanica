/**
 * @file MxMeshSurfaceAreaConstraint.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh surface area constraint
 * @date 2022-06-15
 * 
 */

#include "MxMeshSurfaceAreaConstraint.h"

#include <models/vertex/solver/MxMeshVertex.h>
#include <models/vertex/solver/MxMeshSurface.h>
#include <models/vertex/solver/MxMeshBody.h>


HRESULT MxMeshSurfaceAreaConstraint::energy(MxMeshObj *source, MxMeshObj *target, float &e) {
    MxMeshBody *b = (MxMeshBody*)source;
    float darea = b->getArea() - constr;
    e = lam * darea * darea;
    return S_OK;
}

HRESULT MxMeshSurfaceAreaConstraint::force(MxMeshObj *source, MxMeshObj *target, float *f) {
    MxMeshBody *b = (MxMeshBody*)source;
    MxMeshVertex *v = (MxMeshVertex*)target;
    
    MxVector3f posc = v->getPosition();
    MxVector3f ftotal(0.f);

    for(auto &s : v->getSurfaces()) {
        if(!s->in(b)) 
            continue;
        
        auto svertices = s->getVertices();
        unsigned int idx, idxc, idxp, idxn;
        MxVector3f sftotal(0.f);
        
        for(idx = 0; idx < svertices.size(); idx++) {
            if(svertices[idx] == v) 
                idxc = idx;
            sftotal += Magnum::Math::cross(s->triangleNormal(idx).normalized(), 
                                           svertices[idx == svertices.size() - 1 ? 0 : idx + 1]->getPosition() - svertices[idx]->getPosition());
        }
        sftotal /= svertices.size();

        idxp = idxc == 0 ? svertices.size() - 1 : idxc - 1;
        idxn = idxc == svertices.size() - 1 ? 0 : idxc + 1;

        const MxVector3f scentroid = s->getCentroid();

        sftotal += Magnum::Math::cross(s->triangleNormal(idxc).normalized(), scentroid - svertices[idxn]->getPosition());
        sftotal -= Magnum::Math::cross(s->triangleNormal(idxp).normalized(), scentroid - svertices[idxp]->getPosition());
        ftotal += sftotal;
    }

    ftotal *= (lam * (constr - b->getArea()));

    f[0] += ftotal[0];
    f[1] += ftotal[1];
    f[2] += ftotal[2];

    return S_OK;
}
