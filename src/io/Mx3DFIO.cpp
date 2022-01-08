/**
 * @file Mx3DFIO.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format import/export interface
 * @date 2021-12-13
 * 
 */

#include <assimp/postprocess.h>

#include <engine.h>
#include <rendering/MxStyle.hpp>
#include <MxParticleList.hpp>
#include <io/generators/Mx3DFAngleMeshGenerator.h>
#include <io/generators/Mx3DFBondMeshGenerator.h>
#include <io/generators/Mx3DFDihedralMeshGenerator.h>
#include <io/generators/Mx3DFPCloudMeshGenerator.h>

#include "Mx3DFIO.h"


Mx3DFStructure *Mx3DFIO::fromFile(const std::string &filePath) {
    Mx3DFStructure *result = new Mx3DFStructure();
    
    if(result->fromFile(filePath) != S_OK) 
        return NULL;

    return result;
}

Mx3DFMeshData *generate3DFMeshByType(MxParticleType *pType, const unsigned int &pRefinements) {

    Mx3DFPCloudMeshGenerator generatorPCloud;
    generatorPCloud.pList = *pType->items();
    generatorPCloud.pRefinements = pRefinements;

    if(generatorPCloud.pList.nr_parts == 0 || generatorPCloud.process() != S_OK) 
        return 0;

    auto mesh = generatorPCloud.getMesh();
    mesh->name += " (" + std::string(pType->name) + ")";

    mesh->renderData = new Mx3DFRenderData();
    mesh->renderData->color = pType->style->color;
    
    return mesh;

}

HRESULT Mx3DFIO::toFile(const std::string &format, const std::string &filePath, const unsigned int &pRefinements) {

    // Build structure

    Mx3DFStructure structure;

    // Generate point cloud mesh by particle type
    
    if(_Engine.s.nr_parts > 0) 
        for(unsigned int i = 0; i < _Engine.nr_types; i++) {
            auto mesh = generate3DFMeshByType(&_Engine.types[i], pRefinements);
            if(mesh != NULL) 
                structure.add(mesh);
        }

    // Generate bond mesh

    Mx3DFBondMeshGenerator generatorBonds;
    generatorBonds.pRefinements = pRefinements;

    if(_Engine.nr_bonds > 0) {
        
        generatorBonds.bonds.reserve(_Engine.nr_bonds);

        for(unsigned int i = 0; i < _Engine.nr_bonds; i++) {

            auto b = _Engine.bonds[i];
            if(b.flags & BOND_ACTIVE)
                generatorBonds.bonds.push_back(MxBondHandle(b.id));

        }

        if(generatorBonds.process() == S_OK) 
            structure.add(generatorBonds.getMesh());

    }

    // Generate angle mesh

    Mx3DFAngleMeshGenerator generatorAngles;
    generatorAngles.pRefinements = pRefinements;

    if(_Engine.nr_angles > 0) {

        generatorAngles.angles.reserve(_Engine.nr_angles);

        for(unsigned int i = 0; i < _Engine.nr_angles; i++) {

            auto a = _Engine.angles[i];

            if(a.flags & ANGLE_ACTIVE) 
                generatorAngles.angles.push_back(MxAngleHandle(i));

        }

        if(generatorAngles.process() == S_OK) 
            structure.add(generatorAngles.getMesh());

    }

    // Generate dihedral mesh

    Mx3DFDihedralMeshGenerator generatorDihedrals;
    generatorDihedrals.pRefinements = pRefinements;

    if(_Engine.nr_dihedrals > 0) {

        generatorDihedrals.dihedrals.reserve(_Engine.nr_dihedrals);

        for(unsigned int i = 0; i < _Engine.nr_dihedrals; i++) {

            generatorDihedrals.dihedrals.push_back(MxDihedralHandle(i));
            
        }

        if(generatorDihedrals.process() == S_OK) 
            structure.add(generatorDihedrals.getMesh());

    }

    // Export

    if(structure.toFile(format, filePath) != S_OK) 
        return E_FAIL;

    if(structure.clear() != S_OK) 
        return E_FAIL;

    return S_OK;
}
