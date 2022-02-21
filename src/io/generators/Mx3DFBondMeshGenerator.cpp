/**
 * @file Mx3DFBondMeshGenerator.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines generating 3D format meshes from Mechanica bonds
 * @date 2021-12-18
 * 
 */

#include <MxParticle.h>
#include <rendering/MxStyle.hpp>

#include <io/Mx3DFRenderData.h>

#include "Mx3DFBondMeshGenerator.h"


// Mx3DFBondMeshGenerator


HRESULT Mx3DFBondMeshGenerator::process() {

    this->mesh->name = "Bonds";

    // Generate render data

    this->mesh->renderData = new Mx3DFRenderData();
    this->mesh->renderData->color = MxBond_StylePtr->color;

    for(auto bh : this->bonds) {

        MxParticleHandle *pi = bh[0];
        MxParticleHandle *pj = bh[1];

        std::vector<Mx3DFFaceData*> faces;
        std::vector<Mx3DFEdgeData*> edges;
        std::vector<Mx3DFVertexData*> vertices;
        std::vector<MxVector3f> normals;

        generateCylinderMesh(this->mesh, 
                             &faces, 
                             &edges, 
                             &vertices, 
                             &normals, 
                             this->radius, 
                             pi->getPosition(), 
                             pj->getPosition(), 
                             this->pRefinements);

    }

    return S_OK;
}
