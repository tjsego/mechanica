/**
 * @file Mx3DFBondMeshGenerator.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines generating 3D format meshes from Mechanica bonds
 * @date 2021-12-18
 * 
 */

#ifndef SRC_MX_IO_MX3DFBONDMESHGENERATOR_H_
#define SRC_MX_IO_MX3DFBONDMESHGENERATOR_H_

#include <bond.h>

#include "Mx3DFMeshGenerator.h"

struct Mx3DFBondMeshGenerator : Mx3DFMeshGenerator { 

    /* Bonds of this mesh */
    std::vector<MxBondHandle> bonds;

    /* Mesh refinements applied when generating meshes from bonds */
    unsigned int pRefinements = 0;

    /* Radius of rendered bonds */
    float radius = 0.01;
    
    // Mx3DFMeshGenerator interface
    
    HRESULT process();

};

#endif // SRC_MX_IO_MX3DFBONDMESHGENERATOR_H_
