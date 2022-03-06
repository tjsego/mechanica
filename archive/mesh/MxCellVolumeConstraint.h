/*
 * MxCellVolumeConstraint.h
 *
 *  Created on: Sep 20, 2018
 *      Author: andy
 */

#ifndef SRC_MXCELLVOLUMECONSTRAINT_H_
#define SRC_MXCELLVOLUMECONSTRAINT_H_

#include "MxConstraints.h"
#include "MxCell.h"

struct MxCellVolumeConstraint : IConstraint
{
    MxCellVolumeConstraint(float targetVolume, float lambda);

    virtual HRESULT setTime(float time);

    virtual float energy(const std::vector<MxConstrainable*> &objs);

    virtual HRESULT project(const std::vector<MxConstrainable*> &obj);

    float targetVolume;
    float lambda;

private:
    float energy(const MxConstrainable *obj);
};

#endif /* SRC_MXCELLVOLUMECONSTRAINT_H_ */
