/**
 * @file Mx3DFMeshGenerator.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines interface for generating 3D format meshes from Mechanica objects
 * @date 2021-12-18
 * 
 */

#include <Magnum/Trade/MeshData.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/Primitives/Cylinder.h>

#include "Mx3DFMeshGenerator.h"


// Mx3DFMeshGenerator


Mx3DFMeshGenerator::Mx3DFMeshGenerator() {
    this->mesh = new Mx3DFMeshData();
}

Mx3DFMeshData *Mx3DFMeshGenerator::getMesh() {
    return this->mesh;
}


// Supporting functions


HRESULT constructExplicitMxMesh(std::vector<MxVector3f> positions, 
                                std::vector<MxVector3f> normals, 
                                std::vector<std::vector<unsigned int> > indices, 
                                std::vector<Mx3DFVertexData*> *vertices, 
                                std::vector<Mx3DFEdgeData*> *edges, 
                                std::vector<Mx3DFFaceData*> *faces, 
                                Mx3DFMeshData *mesh) 
{ 

    unsigned int numVerts = positions.size();
    unsigned int numFaces = indices.size();

    vertices->reserve(numVerts);

    Mx3DFVertexData *vertex;
    Mx3DFEdgeData *edge;
    Mx3DFFaceData *face;

    // Construct vertices

    for(unsigned int vIdx = 0; vIdx < numVerts; vIdx++) 
        vertices->push_back(new Mx3DFVertexData(positions[vIdx]));
    
    for(unsigned int fIdx = 0; fIdx < numFaces; fIdx++) {

        // Construct face

        face = new Mx3DFFaceData();

        // Get vertices and normal of this face

        auto fIndices = indices[fIdx];

        std::vector<Mx3DFVertexData*> fverts;
        fverts.reserve(fIndices.size());
        MxVector3f fn = {0.f, 0.f, 0.f};
        for(auto fInd : fIndices) {

            fverts.push_back((*vertices)[fInd]);
            fn += normals[fInd];

        }
        face->normal = fn / fn.length();

        // Find or create edges

        std::vector<Mx3DFEdgeData*> fedges(fIndices.size(), 0);

        for(unsigned int i = 0; i < fIndices.size(); i++) {
            auto va = fverts[i];
            auto vb = i == fIndices.size() - 1 ? fverts[0] : fverts[i + 1];

            for(auto e : va->getEdges()) {
                if(e->va == vb || e->vb == vb) {
                    fedges[i] = e;
                    break;
                }
            }

            if(fedges[i] == NULL) {
                edge = new Mx3DFEdgeData(va, vb);
                fedges[i] = edge;
                edges->push_back(edge);
            }
        }

        // Add all edges to face without condition

        face->edges.reserve(fedges.size());
        for(auto e : fedges) {

            face->edges.push_back(e);
            e->faces.push_back(face);

        }

        // Connect mesh and face

        mesh->faces.push_back(face);
        face->meshes.push_back(mesh);
        faces->push_back(face);

    }

    return S_OK;
}


HRESULT generateBallMesh(Mx3DFMeshData *mesh, 
                         std::vector<Mx3DFFaceData*> *faces, 
                         std::vector<Mx3DFEdgeData*> *edges, 
                         std::vector<Mx3DFVertexData*> *vertices, 
                         std::vector<MxVector3f> *normals, 
                         const float &radius, 
                         const MxVector3f &offset, 
                         const unsigned int &numDivs) 
{ 
    Magnum::Trade::MeshData icoSphere = Magnum::Primitives::icosphereSolid(numDivs);
    auto mg_positions = icoSphere.positions3DAsArray();
    auto mg_normals = icoSphere.normalsAsArray();
    auto mg_indices = icoSphere.indicesAsArray();

    unsigned int numVerts = mg_positions.size();
    unsigned int numFaces = mg_indices.size() / 3;
    
    std::vector<MxVector3f> positions;
    std::vector<std::vector<unsigned int> > assmIndices;

    vertices->reserve(numVerts);
    normals->reserve(numVerts);
    faces->reserve(numFaces);
    positions.reserve(numVerts);
    assmIndices.reserve(numFaces);

    // Generate transformation
    MxMatrix4f transformation = Magnum::Matrix4::translation(offset) * Magnum::Matrix4::scaling(MxVector3f(radius));

    for(unsigned int vIdx = 0; vIdx < numVerts; vIdx++) {
        
        // Apply transformation

        MxVector3f pos = mg_positions[vIdx];
        MxVector4f post = {pos.x(), pos.y(), pos.z(), 1.f};
        MxVector3f position = (transformation * post).xyz();

        positions.push_back(position);
        normals->push_back(mg_normals[vIdx]);
        
    }

    for(unsigned int fIdx = 0; fIdx < numFaces; fIdx++) {

        auto bIndices = &mg_indices[3 * fIdx];
        assmIndices.push_back({bIndices[0], bIndices[1], bIndices[2]});

    }

    constructExplicitMxMesh(positions, *normals, assmIndices, vertices, edges, faces, mesh);

    return S_OK;
}

HRESULT generateCylinderMesh(Mx3DFMeshData *mesh, 
                             std::vector<Mx3DFFaceData*> *faces, 
                             std::vector<Mx3DFEdgeData*> *edges, 
                             std::vector<Mx3DFVertexData*> *vertices, 
                             std::vector<MxVector3f> *normals, 
                             const float &radius, 
                             const MxVector3f &startPt, 
                             const MxVector3f &endPt, 
                             const unsigned int &numDivs) 
{
    auto cylVec = endPt - startPt;
    float cylLength = cylVec.length();
    float cylHalfLength = 0.5 * cylLength / radius;

    Magnum::Trade::MeshData cylinder = Magnum::Primitives::cylinderSolid(1, 3 * (numDivs + 1), cylHalfLength);

    auto mg_positions = cylinder.positions3DAsArray();
    auto mg_normals = cylinder.normalsAsArray();
    auto mg_indices = cylinder.indicesAsArray();

    unsigned int numVerts = mg_positions.size();
    unsigned int numFaces = mg_indices.size() / 3;
    
    std::vector<MxVector3f> positions;
    std::vector<std::vector<unsigned int> > assmIndices;

    vertices->reserve(numVerts);
    normals->reserve(numVerts);
    faces->reserve(numFaces);
    positions.reserve(numVerts);
    assmIndices.reserve(numFaces);

    // Generate transformation
    MxVector3f cylVec0 = {0.f, 1.f, 0.f};
    MxVector3f rotVec = Magnum::Math::cross(cylVec0, cylVec);
    rotVec = rotVec / rotVec.length();
    float rotAng = std::acos(cylVec.y() / cylLength);
    MxMatrix4f tRotate = Magnum::Matrix4::rotation(Magnum::Rad(rotAng), rotVec);
    MxMatrix4f transformation = Magnum::Matrix4::scaling(MxVector3f(radius));
    transformation = Magnum::Matrix4::translation({0.f, 0.5f * cylLength, 0.f}) * transformation;
    transformation = tRotate * transformation;
    transformation = Magnum::Matrix4::translation(startPt) * transformation;

    for(unsigned int vIdx = 0; vIdx < numVerts; vIdx++) {
        
        // Apply transformation

        MxVector3f pos = mg_positions[vIdx];
        MxVector4f qt = {pos.x(), pos.y(), pos.z(), 1.f};
        MxVector3f post = (transformation * qt).xyz();

        MxVector3f norm = mg_normals[vIdx];
        qt = {norm.x(), norm.y(), norm.z(), 1.f};
        MxVector3f normt = (tRotate * qt).xyz();

        positions.push_back(post);
        normals->push_back(normt);
        
    }

    for(unsigned int fIdx = 0; fIdx < numFaces; fIdx++) {

        auto bIndices = &mg_indices[3 * fIdx];
        assmIndices.push_back({bIndices[0], bIndices[1], bIndices[2]});

    }

    constructExplicitMxMesh(positions, *normals, assmIndices, vertices, edges, faces, mesh);

    return S_OK;
}
