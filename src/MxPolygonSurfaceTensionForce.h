/*
 * MxPolygonSurfaceTensionForce.h
 *
 *  Created on: Sep 20, 2018
 *      Author: andy
 */

#ifndef SRC_MXPOLYGONSURFACETENSIONFORCE_H_
#define SRC_MXPOLYGONSURFACETENSIONFORCE_H_

#include "MxForces.h"

struct MxPolygonSurfaceTensionForce : IForce
{
    MxPolygonSurfaceTensionForce(float surfaceTension);

    FORCABLE_TYPE dataType() { return FORCABLE_TYPE::FORCABLE_POLYGON; }

    /**
     * Called when the main time step changes.
     */
    virtual HRESULT setTime(float time);

    /**
     * Apply forces to a set of objects.
     */
    virtual HRESULT applyForce(float time, const std::vector<MxForcable*> &objs) const;

    float surfaceTension;
};

#endif /* SRC_MXPOLYGONSURFACETENSIONFORCE_H_ */
