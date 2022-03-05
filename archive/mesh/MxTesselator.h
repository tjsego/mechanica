/*
 * MxTesselator.h
 *
 *  Created on: Apr 16, 2018
 *      Author: andy
 */

#ifndef SRC_MXTESSELATOR_H_
#define SRC_MXTESSELATOR_H_

#include <Magnum/Magnum.h>
#include <vector>
#include "mechanica_private.h"
#include "MxMeshCore.h"


struct MxTesselatorResult {
    std::vector<MxVector3f> vertices;
    std::vector<int> indices;
};

MxTesselatorResult MxTriangulateFaceSimple(const std::vector<MxVector3f> &vertices);


#endif /* SRC_MXTESSELATOR_H_ */
