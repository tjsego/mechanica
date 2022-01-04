/**
 * @file Mx3DFAngleMeshGenerator.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines generating 3D format meshes from Mechanica angles
 * @date 2021-12-18
 * 
 */

#ifndef SRC_MX_IO_MX3DFANGLEMESHGENERATOR_H_
#define SRC_MX_IO_MX3DFANGLEMESHGENERATOR_H_

#include <angle.h>

#include "Mx3DFMeshGenerator.h"

struct Mx3DFAngleMeshGenerator : Mx3DFMeshGenerator {

    /* Angles of this mesh */
    std::vector<MxAngleHandle> angles;

    /* Mesh refinements applied when generating meshes from angles */
    unsigned int pRefinements = 0;

    /* Radius of rendered angles */
    float radius = 0.01;
    
    // Mx3DFMeshGenerator interface
    
    HRESULT process();
    
};

#endif // SRC_MX_IO_MX3DFANGLEMESHGENERATOR_H_
