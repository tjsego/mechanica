/**
 * @file Mx3DFStructure.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format structure container
 * @date 2021-12-15
 * 
 */

// todo: implement a sewing method on Mx3DFStructure to join same vertices but in different meshes
//  Currently, vertices are allocated by mesh; if a vertex is shared by meshes, they will show up as separate entities in each mesh. 
//  This method should detect such occurrences and make one shared vertex among many meshes

#ifndef SRC_MX_IO_MX3DFSTRUCTURE_H_
#define SRC_MX_IO_MX3DFSTRUCTURE_H_

#include <mx_port.h>

#include "Mx3DFVertexData.h"
#include "Mx3DFEdgeData.h"
#include "Mx3DFFaceData.h"
#include "Mx3DFMeshData.h"

#include <unordered_map>


struct CAPI_EXPORT Mx3DFComponentContainer {
    std::vector<Mx3DFVertexData*> vertices;
    std::vector<Mx3DFEdgeData*> edges;
    std::vector<Mx3DFFaceData*> faces;
    std::vector<Mx3DFMeshData*> meshes;
};


/**
 * @brief Container for relevant data found in a 3D data file. 
 * 
 * The structure object owns all constituent data. 
 * 
 * Recursively adds/removes constituent data and all child data. 
 * However, the structure enforces no rules on the constituent container data. 
 * For example, when an edge is added, all constituent vertices are added. 
 * However, no assignment is made to ensure that the parent edge 
 * is properly stored in the parent container of the vertices, neither are 
 * parent edges added when a vertex is added. 
 * 
 */
struct CAPI_EXPORT Mx3DFStructure {

    /** Inventory of structure objects */
    Mx3DFComponentContainer inventory;

    /** Inventory of objects scheduled for deletion */
    Mx3DFComponentContainer queueRemove;

    /** Default radius applied to vertices when generating meshes from point clouds */
    float vRadiusDef = 0.1;

    ~Mx3DFStructure();

    // Structure management

    /**
     * @brief Load from file
     * 
     * @param filePath file absolute path
     * @return HRESULT 
     */
    HRESULT fromFile(const std::string &filePath);

    /**
     * @brief Write to file
     * 
     * @param format output format of file
     * @param filePath file absolute path
     * @return HRESULT 
     */
    HRESULT toFile(const std::string &format, const std::string &filePath);

    /**
     * @brief Flush stucture. All scheduled processes are executed. 
     * 
     */
    HRESULT flush();

    // Inventory management

    /**
     * @brief Extend a structure
     * 
     * @param s stucture to extend with
     */
    HRESULT extend(const Mx3DFStructure &s);

    /**
     * @brief Clear all data of the structure
     * 
     * @return HRESULT 
     */
    HRESULT clear();
    
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
     * @brief Get all constituent meshes
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
     * @brief Get the number of constituent faces
     * 
     * @return unsigned int 
     */
    unsigned int getNumFaces();

    /**
     * @brief Get the number of constituent meshes
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
     * @brief Test whether a face is a constituent
     * 
     * @param f face to test
     * @return true 
     * @return false 
     */
    bool has(Mx3DFFaceData *f);

    /**
     * @brief Test whether a mesh is a constituent
     * 
     * @param m mesh to test
     * @return true 
     * @return false 
     */
    bool has(Mx3DFMeshData *m);

    /**
     * @brief Add a vertex
     * 
     * @param v vertex to add
     */
    void add(Mx3DFVertexData *v);

    /**
     * @brief Add an edge and all constituent data
     * 
     * @param e edge to add
     */
    void add(Mx3DFEdgeData *e);

    /**
     * @brief Add a face and all constituent data
     * 
     * @param f face to add
     */
    void add(Mx3DFFaceData *f);

    /**
     * @brief Add a mesh and all constituent data
     * 
     * @param m mesh to add
     */
    void add(Mx3DFMeshData *m);

    /**
     * @brief Remove a vertex
     * 
     * @param v vertex to remove
     */
    void remove(Mx3DFVertexData *v);

    /**
     * @brief Remove a edge and all constituent data
     * 
     * @param e edge to remove
     */
    void remove(Mx3DFEdgeData *e);

    /**
     * @brief Remove a face and all constituent data
     * 
     * @param f face to remove
     */
    void remove(Mx3DFFaceData *f);

    /**
     * @brief Remove a mesh and all constituent data
     * 
     * @param m mesh to remove
     */
    void remove(Mx3DFMeshData *m);

    void onRemoved(Mx3DFVertexData *v);
    void onRemoved(Mx3DFEdgeData *e);
    void onRemoved(Mx3DFFaceData *f);

    // Transformations

    /**
     * @brief Get the centroid of the structure
     * 
     * @return MxVector3f 
     */
    MxVector3f getCentroid();

    /**
     * @brief Translate the structure by a displacement
     * 
     * @param displacement 
     * @return HRESULT 
     */
    HRESULT translate(const MxVector3f &displacement);

    /**
     * @brief Translate the structure to a position
     * 
     * @param position 
     * @return HRESULT 
     */
    HRESULT translateTo(const MxVector3f &position);

    /**
     * @brief Rotate the structure about a point
     * 
     * @param rotMat 
     * @param rotPt 
     * @return HRESULT 
     */
    HRESULT rotateAt(const MxMatrix3f &rotMat, const MxVector3f &rotPt);

    /**
     * @brief Rotate the structure about its centroid
     * 
     * @param rotMat 
     * @return HRESULT 
     */
    HRESULT rotate(const MxMatrix3f &rotMat);

    /**
     * @brief Scale the structure about a point
     * 
     * @param scales 
     * @param scalePt 
     * @return HRESULT 
     */
    HRESULT scaleFrom(const MxVector3f &scales, const MxVector3f &scalePt);

    /**
     * @brief Scale the structure uniformly about a point
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

private:

    int id_vertex = 0;
    int id_edge = 0;
    int id_face = 0;
    int id_mesh = 0;

};

#endif // SRC_MX_IO_MX3DFSTRUCTURE_H_
