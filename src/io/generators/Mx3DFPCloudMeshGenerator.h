/**
 * @file Mx3DFPCloudMeshGenerator.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines generating 3D format meshes from Mechanica particles
 * @date 2021-12-18
 * 
 */

#ifndef SRC_MX_IO_MX3DFPCLOUDMESHGENERATOR_H_
#define SRC_MX_IO_MX3DFPCLOUDMESHGENERATOR_H_

#include <MxParticleList.hpp>

#include "Mx3DFMeshGenerator.h"

struct Mx3DFPCloudMeshGenerator : Mx3DFMeshGenerator {

    /* List of particles of this cloud */
    MxParticleList pList;

    /* Mesh refinements applied when generating meshes from point clouds */
    unsigned int pRefinements = 0;

    // Mx3DFMeshGenerator interface

    HRESULT process();

};

#endif // SRC_MX_IO_MX3DFPCLOUDMESHGENERATOR_H_
