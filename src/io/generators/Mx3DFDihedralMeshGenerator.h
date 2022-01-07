/**
 * @file Mx3DFDihedralMeshGenerator.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines generating 3D format meshes from Mechanica dihedrals
 * @date 2021-12-18
 * 
 */

#ifndef SRC_MX_IO_MX3DFDIHEDRALMESHGENERATOR_H_
#define SRC_MX_IO_MX3DFDIHEDRALMESHGENERATOR_H_

#include "dihedral.h"

#include "Mx3DFMeshGenerator.h"

struct Mx3DFDihedralMeshGenerator : Mx3DFMeshGenerator {
    
    /* Dihedrals of this mesh */
    std::vector<MxDihedralHandle> dihedrals;

    /* Mesh refinements applied when generating meshes from dihedrals */
    unsigned int pRefinements = 0;

    /* Radius of rendered dihedrals */
    float radius = 0.01;
    
    // Mx3DFMeshGenerator interface
    
    HRESULT process();
    
};

#endif // SRC_MX_IO_MX3DFDIHEDRALMESHGENERATOR_H_
