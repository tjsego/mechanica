/**
 * @file Mx3DFMeshGenerator.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines interface and supporting functions for generating 3D format meshes from Mechanica objects
 * @date 2021-12-18
 * 
 */

#ifndef SRC_MX_IO_MX3DFMESHGENERATOR_H_
#define SRC_MX_IO_MX3DFMESHGENERATOR_H_

#include <io/Mx3DFVertexData.h>
#include <io/Mx3DFEdgeData.h>
#include <io/Mx3DFFaceData.h>
#include <io/Mx3DFMeshData.h>


struct Mx3DFMeshGenerator { 

    Mx3DFMeshGenerator();

    /**
     * @brief Get the mesh of this generator. 
     * 
     * Mesh must first be processed before it is generated. 
     * 
     * @return Mx3DFMeshData* 
     */
    Mx3DFMeshData *getMesh();

    /**
     * @brief Do all instructions to generate mesh. 
     * 
     * @return HRESULT 
     */
    virtual HRESULT process() = 0;

protected:

    Mx3DFMeshData *mesh;

};


// Supprting generator functions

/**
 * @brief Adds elements of a ball at a point to a mesh
 * 
 * @param mesh mesh to append
 * @param faces generated faces
 * @param edges generated edges
 * @param vertices generated vertices
 * @param normals generated normals
 * @param radius radius of ball
 * @param offset location of ball
 * @param numDivs number of refinements
 * @return HRESULT 
 */
HRESULT generateBallMesh(Mx3DFMeshData *mesh, 
                         std::vector<Mx3DFFaceData*> *faces, 
                         std::vector<Mx3DFEdgeData*> *edges, 
                         std::vector<Mx3DFVertexData*> *vertices, 
                         std::vector<MxVector3f> *normals, 
                         const float &radius=1.0, 
                         const MxVector3f &offset={0.f,0.f,0.f}, 
                         const unsigned int &numDivs=0);

HRESULT generateCylinderMesh(Mx3DFMeshData *mesh, 
                             std::vector<Mx3DFFaceData*> *faces, 
                             std::vector<Mx3DFEdgeData*> *edges, 
                             std::vector<Mx3DFVertexData*> *vertices, 
                             std::vector<MxVector3f> *normals, 
                             const float &radius=1.0, 
                             const MxVector3f &startPt={0.f,0.f,0.f}, 
                             const MxVector3f &endPt={1.f,1.f,1.f}, 
                             const unsigned int &numDivs=0);


#endif // SRC_MX_IO_MX3DFMESHGENERATOR_H_
