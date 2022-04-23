/**
 * @file Mx3DFFaceData.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format face data
 * @date 2021-12-15
 * 
 */

#ifndef SRC_MX_IO_MX3DFACEDATA_H_
#define SRC_MX_IO_MX3DFACEDATA_H_

#include <mechanica_private.h>

#include <vector>


struct Mx3DFVertexData;
struct Mx3DFEdgeData;
struct Mx3DFMeshData;
struct Mx3DFStructure;


/**
 * @brief 3D data file face data
 * 
 */
struct CAPI_EXPORT Mx3DFFaceData {
    
    /** Parent structure */
    Mx3DFStructure *structure = NULL;

    /** Face normal */
    MxVector3f normal;

    /** ID, if any. Unique to its structure and type */
    int id = -1;

    /** Constituent edges */
    std::vector<Mx3DFEdgeData*> edges;

    /** Parent meshes */
    std::vector<Mx3DFMeshData*> meshes;

    /**
     * @brief Get all constituent vertices
     * 
     * @return std::vector<Mx3DFVertexData*> 
     */
    std::vector<Mx3DFVertexData*> getVertices();

    /**
     * @brief Get all constituent edges
     * 
     * @return std::vector<Mx3DFEdgeData*> 
     */
    std::vector<Mx3DFEdgeData*> getEdges();

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
     * @brief Get the number of constituent edges
     * 
     * @return unsigned int 
     */
    unsigned int getNumEdges();

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
     * @brief Test whether an edge is a constituent
     * 
     * @param e edge to test
     * @return true 
     * @return false 
     */
    bool has(Mx3DFEdgeData *e);
    
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

#endif // SRC_MX_IO_MX3DFACEDATA_H_
