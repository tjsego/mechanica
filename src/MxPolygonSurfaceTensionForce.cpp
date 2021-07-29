/*
 * MxPartialPolygonSurfaceTensionForce.cpp
 *
 *  Created on: Sep 20, 2018
 *      Author: andy
 */

#include <MxPolygonSurfaceTensionForce.h>
#include "MxCell.h"



MxPolygonSurfaceTensionForce::MxPolygonSurfaceTensionForce(float _surfaceTension):
    surfaceTension{_surfaceTension}
{
}

HRESULT MxPolygonSurfaceTensionForce::setTime(float time)
{
    return S_OK;
}

HRESULT MxPolygonSurfaceTensionForce::applyForce(float time, const std::vector<MxForcable*> &objs) const
{
    for(auto o : objs) {

        MxPolygon *pp = static_cast<MxPolygon*>(o);

        for(uint i = 0; i < pp->vertices.size(); ++i) {
            VertexPtr vi = pp->vertices[i];
            VertexPtr vn = pp->vertices[(i+1)%pp->vertices.size()];
            Vector3 dx = vn->position - vi->position;

            vi->force += surfaceTension * dx;
            vn->force -= surfaceTension * dx;
        }
    }

    return S_OK;
}
