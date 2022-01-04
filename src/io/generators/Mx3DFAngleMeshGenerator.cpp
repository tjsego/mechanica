/**
 * @file Mx3DFAngleMeshGenerator.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines generating 3D format meshes from Mechanica angles
 * @date 2021-12-18
 * 
 */

#include <MxParticle.h>
#include <rendering/NOMStyle.hpp>

#include <io/Mx3DFRenderData.h>

#include "Mx3DFAngleMeshGenerator.h"


// Mx3DFAngleMeshGenerator


HRESULT Mx3DFAngleMeshGenerator::process() {

    this->mesh->name = "Angles";

    // Generate render data

    this->mesh->renderData = new Mx3DFRenderData();
    this->mesh->renderData->color = MxAngle_StylePtr->color;

    for(auto ah : this->angles) {

        MxVector3f posi = ah[0]->getPosition();
        MxVector3f posj = ah[1]->getPosition();
        MxVector3f posk = ah[2]->getPosition();

        MxVector3f mij = 0.5 * (posi + posj);
        MxVector3f mkj = 0.5 * (posk + posj);

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
                             posi, 
                             posj, 
                             this->pRefinements);

        faces.clear(); edges.clear(); vertices.clear(); normals.clear();
        generateCylinderMesh(this->mesh, 
                             &faces, 
                             &edges, 
                             &vertices, 
                             &normals, 
                             this->radius, 
                             posk, 
                             posj, 
                             this->pRefinements);

        faces.clear(); edges.clear(); vertices.clear(); normals.clear();
        generateCylinderMesh(this->mesh, 
                             &faces, 
                             &edges, 
                             &vertices, 
                             &normals, 
                             0.5f * this->radius, 
                             mij, 
                             mkj, 
                             this->pRefinements);

    }

    return S_OK;
}
