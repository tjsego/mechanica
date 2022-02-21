/*
 * MxPolygonAreaConstraint.h
 *
 *  Created on: Sep 20, 2018
 *      Author: andy
 */

#ifndef SRC_MXPOLYGONAREACONSTRAINT_H_
#define SRC_MXPOLYGONAREACONSTRAINT_H_

#include "MxConstraints.h"

struct MxPolygonAreaConstraint : IConstraint
{
    CONSTRAINABLE_TYPE dataType() { return CONSTRAINABLE_TYPE::CONSTRAINABLE_POLYGON; }
    
    MxPolygonAreaConstraint(float targetArea, float lambda);

    virtual HRESULT setTime(float time);

    virtual float energy(const std::vector<MxConstrainable*> &objs);

    virtual HRESULT project(const std::vector<MxConstrainable*> &obj);

    float targetArea;
    float lambda;

    float energy(const MxConstrainable* obj);
};

#endif /* SRC_MXPOLYGONAREACONSTRAINT_H_ */
