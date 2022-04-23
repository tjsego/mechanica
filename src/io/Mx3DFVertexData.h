/**
 * @file Mx3DFVertexData.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format vertex data
 * @date 2021-12-15
 * 
 */

#ifndef SRC_MX_IO_MX3DVERTEXDATA_H_
#define SRC_MX_IO_MX3DVERTEXDATA_H_

#include <mechanica_private.h>

#include <vector>


struct Mx3DFEdgeData;
struct Mx3DFFaceData;
struct Mx3DFMeshData;
struct Mx3DFStructure;


/**
 * @brief 3D data file vertex data
 * 
 */
struct CAPI_EXPORT Mx3DFVertexData {
    
    /** Parent structure */
    Mx3DFStructure *structure = NULL;

    /** Global position */
    MxVector3f position;

    /** ID, if any. Unique to its structure and type */
    int id = -1;

    /** Parent edges, if any */
    std::vector<Mx3DFEdgeData*> edges;

    Mx3DFVertexData(const MxVector3f &_position, Mx3DFStructure *_structure=NULL);

    /**
     * @brief Get all parent edges
     * 
     * @return std::vector<Mx3DFEdgeData*> 
     */
    std::vector<Mx3DFEdgeData*> getEdges();

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
     * @brief Get the number of parent edges
     * 
     * @return unsigned int 
     */
    unsigned int getNumEdges();

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
     * @brief Test whether in an edge
     * 
     * @param e edge to test
     * @return true 
     * @return false 
     */
    bool in(Mx3DFEdgeData *e);
    
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

#endif // SRC_MX_IO_MX3DVERTEXDATA_H_