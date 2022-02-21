/**
 * @file Mx3DFEdgeData.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format edge data
 * @date 2021-12-15
 * 
 */

#ifndef SRC_MX_IO_MX3DEDGEDATA_H_
#define SRC_MX_IO_MX3DEDGEDATA_H_

#include <mechanica_private.h>

#include <vector>


struct Mx3DFVertexData;
struct Mx3DFFaceData;
struct Mx3DFMeshData;
struct Mx3DFStructure;


/**
 * @brief 3D data file edge data
 * 
 */
struct Mx3DFEdgeData {
    
    /** Parent structure */
    Mx3DFStructure *structure = NULL;

    /** ID, if any. Unique to its structure and type */
    int id = -1;

    /** constituent vertices */
    Mx3DFVertexData *va, *vb;

    /** Parent faces */
    std::vector<Mx3DFFaceData*> faces;

    Mx3DFEdgeData(Mx3DFVertexData *_va, Mx3DFVertexData *_vb);

    /**
     * @brief Get all constituent vertices
     * 
     * @return std::vector<Mx3DFVertexData*> 
     */
    std::vector<Mx3DFVertexData*> getVertices();

    /**
     * @brief Get all parent faces
     * 
     * @return std::vector<Mx3DFFaceData*> 
     */
    std::vector<Mx3DFFaceData*> getFaces();

    /**
     * @brief Get all parent meshes
     * 
     * @return std::vector<Mx3DFMeshData*> 
     */
    std::vector<Mx3DFMeshData*> getMeshes();

    /**
     * @brief Get the number of constituent vertices
     * 
     * @return unsigned int 
     */
    unsigned int getNumVertices();

    /**
     * @brief Get the number of parent faces
     * 
     * @return unsigned int 
     */
    unsigned int getNumFaces();

    /**
     * @brief Get the number of parent meshes
     * 
     * @return unsigned int 
     */
    unsigned int getNumMeshes();

    /**
     * @brief Test whether a vertex is a constituent
     * 
     * @param v vertex to test
     * @return true 
     * @return false 
     */
    bool has(Mx3DFVertexData *v);
    
    /**
     * @brief Test whether in a face
     * 
     * @param f face to test
     * @return true 
     * @return false 
     */
    bool in(Mx3DFFaceData *f);
    
    /**
     * @brief Test whether in a mesh
     * 
     * @param m mesh to test
     * @return true 
     * @return false 
     */
    bool in(Mx3DFMeshData *m);
    
    /**
     * @brief Test whether in a structure
     * 
     * @param s structure to test
     * @return true 
     * @return false 
     */
    bool in(Mx3DFStructure *s);

};


#endif // SRC_MX_IO_MX3DEDGEDATA_H_