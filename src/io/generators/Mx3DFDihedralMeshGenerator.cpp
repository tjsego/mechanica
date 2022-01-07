/**
 * @file Mx3DFDihedralMeshGenerator.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines generating 3D format meshes from Mechanica dihedrals
 * @date 2021-12-18
 * 
 */

#include <MxParticle.h>
#include <rendering/NOMStyle.hpp>

#include <io/Mx3DFRenderData.h>

#include "Mx3DFDihedralMeshGenerator.h"


// Mx3DFDihedralMeshGenerator


HRESULT Mx3DFDihedralMeshGenerator::process() {

    this->mesh->name = "Dihedrals";

    // Generate render data

    this->mesh->renderData = new Mx3DFRenderData();
    this->mesh->renderData->color = MxDihedral_StylePtr->color;

    for(auto dh : this->dihedrals) {

        MxVector3f posi = dh[0]->getPosition();
        MxVector3f posj = dh[1]->getPosition();
        MxVector3f posk = dh[2]->getPosition();
        MxVector3f posl = dh[3]->getPosition();

        MxVector3f mik = 0.5 * (posi + posk);
        MxVector3f mjl = 0.5 * (posj + posl);

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
                             posk, 
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
                             this->radius, 
                             posj, 
                             posl, 
                             this->pRefinements);

        faces.clear(); edges.clear(); vertices.clear(); normals.clear();
        generateCylinderMesh(this->mesh, 
                             &faces, 
                             &edges, 
                             &vertices, 
                             &normals, 
                             0.5f * this->radius, 
                             mik, 
                             mjl, 
                             this->pRefinements);

    }

    return S_OK;
}
