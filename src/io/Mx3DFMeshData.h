/**
 * @file Mx3DFMeshData.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format mesh data
 * @date 2021-12-15
 * 
 */

#ifndef SRC_MX_IO_MX3DMESHDATA_H_
#define SRC_MX_IO_MX3DMESHDATA_H_

#include <mechanica_private.h>

#include <vector>

#include "Mx3DFRenderData.h"


struct Mx3DFVertexData;
struct Mx3DFEdgeData;
struct Mx3DFFaceData;
struct Mx3DFStructure;


/**
 * @brief 3D data file mesh data
 * 
 */
struct Mx3DFMeshData {
    
    /** Parent structure */
    Mx3DFStructure *structure = NULL;

    /** ID, if any. Unique to its structure and type */
    int id = -1;

    /** Constituent faces */
    std::vector<Mx3DFFaceData*> faces;

    /** Mesh name */
    std::string name;

    /** Rendering data */
    Mx3DFRenderData *renderData = NULL;

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
     * @brief Get all constituent faces
     * 
     * @return std::vector<Mx3DFFaceData*> 
     */
    std::vector<Mx3DFFaceData*> getFaces();

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
     * @brief Get the number of constituent faces
     * 
     * @return unsigned int 
     */
    unsigned int getNumFaces();

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
     * @brief Test whether a face is a constituent
     * 
     * @param f face to test
     * @return true 
     * @return false 
     */
    bool has(Mx3DFFaceData *f);
    
    /**
     * @brief Test whether in a structure
     * 
     * @param s structure to test
     * @return true 
     * @return false 
     */
    bool in(Mx3DFStructure *s);

    /**
     * @brief Get the centroid of the mesh
     * 
     * @return MxVector3f 
     */
    MxVector3f getCentroid();

    // Transformations

    /**
     * @brief Translate the mesh by a displacement
     * 
     * @param displacement 
     * @return HRESULT 
     */
    HRESULT translate(const MxVector3f &displacement);
    
    /**
     * @brief Translate the mesh to a position
     * 
     * @param position 
     * @return HRESULT 
     */
    HRESULT translateTo(const MxVector3f &position);

    /**
     * @brief Rotate the mesh about a point
     * 
     * @param rotMat 
     * @param rotPt 
     * @return HRESULT 
     */
    HRESULT rotateAt(const MxMatrix3f &rotMat, const MxVector3f &rotPt);
    
    /**
     * @brief Rotate the mesh about its centroid
     * 
     * @param rotMat 
     * @return HRESULT 
     */
    HRESULT rotate(const MxMatrix3f &rotMat);
    
    /**
     * @brief Scale the mesh about a point
     * 
     * @param scales 
     * @param scalePt 
     * @return HRESULT 
     */
    HRESULT scaleFrom(const MxVector3f &scales, const MxVector3f &scalePt);
    
    /**
     * @brief Scale the mesh uniformly about a point
     * 
     * @param scale 
     * @param scalePt 
     * @return HRESULT 
     */
    HRESULT scaleFrom(const float &scale, const MxVector3f &scalePt);
    
    /**
     * @brief Scale the structure about its centroid
     * 
     * @param scales 
     * @return HRESULT 
     */
    HRESULT scale(const MxVector3f &scales);
    
    /**
     * @brief Scale the structure uniformly about its centroid
     * 
     * @param scale 
     * @return HRESULT 
     */
    HRESULT scale(const float &scale);

};

#endif // SRC_MX_IO_MX3DMESHDATA_H_
