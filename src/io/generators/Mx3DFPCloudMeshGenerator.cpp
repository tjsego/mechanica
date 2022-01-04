/**
 * @file Mx3DFPCloudMeshGenerator.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines generating 3D format meshes from Mechanica particles
 * @date 2021-12-18
 * 
 */

#include <MxParticle.h>

#include "Mx3DFPCloudMeshGenerator.h"


// Mx3DFPCloudMeshGenerator


HRESULT Mx3DFPCloudMeshGenerator::process() {

    this->mesh->name = "Particles";

    // Generate render data

    for(unsigned int i = 0; i < this->pList.nr_parts; i++) {

        auto p = this->pList.item(i);

        std::vector<Mx3DFVertexData*> pVertices;
        std::vector<Mx3DFEdgeData*> pEdges;
        std::vector<Mx3DFFaceData*> pFaces;
        std::vector<MxVector3f> pNormals;

        generateBallMesh(this->mesh, &pFaces, &pEdges, &pVertices, &pNormals, p->getRadius(), p->getPosition(), this->pRefinements);
    }

    return S_OK;
}
