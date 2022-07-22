/**
 * @file MxCIO.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxIO and associated features
 * @date 2022-04-06
 */

#ifndef _WRAPS_C_MXCIO_H_
#define _WRAPS_C_MXCIO_H_

#include <mx_port.h>

typedef HRESULT (*MxFIOModuleToFileFcn)(struct MxMetaDataHandle, struct MxIOElementHandle*);
typedef HRESULT (*MxFIOModuleFromFileFcn)(struct MxMetaDataHandle, struct MxIOElementHandle);

// Handles

struct CAPI_EXPORT MxMetaDataHandle {
    unsigned int versionMajor;
    unsigned int versionMinor;
    unsigned int versionPatch;
};

struct CAPI_EXPORT MxFIOStorageKeysHandle {
    char *KEY_TYPE;
    char *KEY_VALUE;
    char *KEY_METADATA;
    char *KEY_SIMULATOR;
    char *KEY_UNIVERSE;
    char *KEY_MODULES;
};

/**
 * @brief Handle to a @ref MxIOElement instance
 * 
 */
struct CAPI_EXPORT MxIOElementHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref Mx3DFRenderData instance
 * 
 */
struct CAPI_EXPORT Mx3DFRenderDataHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref Mx3DFVertexData instance
 * 
 */
struct CAPI_EXPORT Mx3DFVertexDataHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref Mx3DFEdgeData instance
 * 
 */
struct CAPI_EXPORT Mx3DFEdgeDataHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref Mx3DFFaceData instance
 * 
 */
struct CAPI_EXPORT Mx3DFFaceDataHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref Mx3DFMeshData instance
 * 
 */
struct CAPI_EXPORT Mx3DFMeshDataHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref Mx3DFStructure instance
 * 
 */
struct CAPI_EXPORT Mx3DFStructureHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxFIOModule instance. 
 * 
 */
struct CAPI_EXPORT MxFIOModuleHandle {
    void *MxObj;
};



////////////////
// MxMetaData //
////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCMetaData_init(struct MxMetaDataHandle *handle);


//////////////////////
// MxFIOStorageKeys //
//////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFIOStorageKeys_init(struct MxFIOStorageKeysHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFIOStorageKeys_destroy(struct MxFIOStorageKeysHandle *handle);


/////////////////
// MxIOElement //
/////////////////


/**
 * @brief Initialize an empty instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_init(struct MxIOElementHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_destroy(struct MxIOElementHandle *handle);

/**
 * @brief Get the instance value type
 * 
 * @param handle populated handle
 * @param type value type
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_getType(struct MxIOElementHandle *handle, char **type, unsigned int *numChars);

/**
 * @brief Set the instance value type
 * 
 * @param handle populated handle
 * @param type value type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_setType(struct MxIOElementHandle *handle, const char *type);

/**
 * @brief Get the instance value
 * 
 * @param handle populated handle
 * @param value value string
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_getValue(struct MxIOElementHandle *handle, char **value, unsigned int *numChars);

/**
 * @brief Set the instance value
 * 
 * @param handle populated handle
 * @param value value string
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_setValue(struct MxIOElementHandle *handle, const char *value);

/**
 * @brief Test whether an instance has a parent element
 * 
 * @param handle populated handle
 * @param hasParent true when instance has a parent element
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_hasParent(struct MxIOElementHandle *handle, bool *hasParent);

/**
 * @brief Get an instance parent
 * 
 * @param handle populated handle
 * @param parent instance parent
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_getParent(struct MxIOElementHandle *handle, struct MxIOElementHandle *parent);

/**
 * @brief Set an instance parent
 * 
 * @param handle populated handle
 * @param parent instance parent
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_setParent(struct MxIOElementHandle *handle, struct MxIOElementHandle *parent);

/**
 * @brief Get the number of child elements
 * 
 * @param handle populated handle
 * @param numChildren number of child elements
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_getNumChildren(struct MxIOElementHandle *handle, unsigned int *numChildren);

/**
 * @brief Get the child element keys
 * 
 * @param handle populated handle
 * @param keys child element keys
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_getKeys(struct MxIOElementHandle *handle, char ***keys);

/**
 * @brief Get a child
 * 
 * @param handle populated handle
 * @param key child key
 * @param child child element
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_getChild(struct MxIOElementHandle *handle, const char *key, struct MxIOElementHandle *child);

/**
 * @brief Set a child
 * 
 * @param handle populated handle
 * @param key child key
 * @param child child element
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIOElement_setChild(struct MxIOElementHandle *handle, const char *key, struct MxIOElementHandle *child);


/////////////////////
// Mx3DFRenderData //
/////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFRenderData_init(struct Mx3DFRenderDataHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFRenderData_destroy(struct Mx3DFRenderDataHandle *handle);

/**
 * @brief Get the color
 * 
 * @param handle populated handle
 * @param color data color
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFRenderData_getColor(struct Mx3DFRenderDataHandle *handle, float **color);

/**
 * @brief Set the color
 * 
 * @param handle populated handle
 * @param color data color
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFRenderData_setColor(struct Mx3DFRenderDataHandle *handle, float *color);


/////////////////////
// Mx3DFVertexData //
/////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param position global position
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_init(struct Mx3DFVertexDataHandle *handle, float *position);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_destroy(struct Mx3DFVertexDataHandle *handle);

/**
 * @brief Test whether an instance has a parent structure
 * 
 * @param handle populated handle
 * @param hasStructure flag signifying whether an instance has a parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_hasStructure(struct Mx3DFVertexDataHandle *handle, bool *hasStructure);

/**
 * @brief Get the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_getStructure(struct Mx3DFVertexDataHandle *handle, struct Mx3DFStructureHandle *structure);

/**
 * @brief Set the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_setStructure(struct Mx3DFVertexDataHandle *handle, struct Mx3DFStructureHandle *structure);

/**
 * @brief Get the global position
 * 
 * @param handle populated handle
 * @param position global position
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_getPosition(struct Mx3DFVertexDataHandle *handle, float **position);

/**
 * @brief Set the global position
 * 
 * @param handle populated handle
 * @param position global position
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_setPosition(struct Mx3DFVertexDataHandle *handle, float *position);

/**
 * @brief Get the ID, if any. Unique to its structure and type. -1 if not set.
 * 
 * @param handle populated handle
 * @param value ID, if any; otherwise -1
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_getId(struct Mx3DFVertexDataHandle *handle, int *value);

/**
 * @brief Set the ID
 * 
 * @param handle populated handle
 * @param value ID
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_setId(struct Mx3DFVertexDataHandle *handle, unsigned int value);

/**
 * @brief Get all parent edges
 * 
 * @param handle populated handle
 * @param edges parent edges
 * @param numEdges number of edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_getEdges(struct Mx3DFVertexDataHandle *handle, struct Mx3DFEdgeDataHandle **edges, unsigned int *numEdges);

/**
 * @brief Set all parent edges
 * 
 * @param handle populated handle
 * @param edges parent edges
 * @param numEdges number of edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_setEdges(struct Mx3DFVertexDataHandle *handle, struct Mx3DFEdgeDataHandle *edges, unsigned int numEdges);

/**
 * @brief Get all parent faces
 * 
 * @param handle populated handle
 * @param faces parent faces
 * @param numFaces number of faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_getFaces(struct Mx3DFVertexDataHandle *handle, struct Mx3DFFaceDataHandle **faces, unsigned int *numFaces);

/**
 * @brief Get all parent meshes
 * 
 * @param handle populated handle
 * @param meshes parent meshes
 * @param numMeshes number of meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_getMeshes(struct Mx3DFVertexDataHandle *handle, struct Mx3DFMeshDataHandle **meshes, unsigned int *numMeshes);

/**
 * @brief Get the number of parent edges
 * 
 * @param handle populated handle
 * @param value number of parent edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_getNumEdges(struct Mx3DFVertexDataHandle *handle, unsigned int *value);

/**
 * @brief Get the number of parent faces
 * 
 * @param handle populated handle
 * @param value number of parent faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_getNumFaces(struct Mx3DFVertexDataHandle *handle, unsigned int *value);

/**
 * @brief Get the number of parent meshes
 * 
 * @param handle populated handle
 * @param value number of parent meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_getNumMeshes(struct Mx3DFVertexDataHandle *handle, unsigned int *value);

/**
 * @brief Test whether in an edge
 * 
 * @param handle populated handle
 * @param edge edge to test
 * @param isIn flag signifying whether in an edge
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_inEdge(struct Mx3DFVertexDataHandle *handle, struct Mx3DFEdgeDataHandle *edge, bool *isIn);

/**
 * @brief Test whether in a face
 * 
 * @param handle populated handle
 * @param face face to test
 * @param isIn flag signifying whether in a face
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_inFace(struct Mx3DFVertexDataHandle *handle, struct Mx3DFFaceDataHandle *face, bool *isIn);

/**
 * @brief Test whether in a mesh
 * 
 * @param handle populated handle
 * @param mesh mesh to test
 * @param isIn flag signifying whether in a mesh
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_inMesh(struct Mx3DFVertexDataHandle *handle, struct Mx3DFMeshDataHandle *mesh, bool *isIn);

/**
 * @brief Test whether in a structure
 * 
 * @param handle populated handle
 * @param structure structure to test
 * @param isIn flag signifying whether in a structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFVertexData_inStructure(struct Mx3DFVertexDataHandle *handle, struct Mx3DFStructureHandle *structure, bool *isIn);


///////////////////
// Mx3DFEdgeData //
///////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param va first vertex
 * @param vb second vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_init(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFVertexDataHandle *va, struct Mx3DFVertexDataHandle *vb);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_destroy(struct Mx3DFEdgeDataHandle *handle);

/**
 * @brief Test whether an instance has a parent structure
 * 
 * @param handle populated handle
 * @param hasStructure flag signifying whether an instance has a parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_hasStructure(struct Mx3DFEdgeDataHandle *handle, bool *hasStructure);

/**
 * @brief Get the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_getStructure(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFStructureHandle *structure);

/**
 * @brief Set the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_setStructure(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFStructureHandle *structure);

/**
 * @brief Get the ID, if any. Unique to its structure and type. -1 if not set.
 * 
 * @param handle populated handle
 * @param value ID, if any; otherwise -1
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_getId(struct Mx3DFEdgeDataHandle *handle, int *value);

/**
 * @brief Set the ID
 * 
 * @param handle populated handle
 * @param value ID
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_setId(struct Mx3DFEdgeDataHandle *handle, unsigned int value);

/**
 * @brief Get all child vertices
 * 
 * @param handle populated handle
 * @param va first vertex
 * @param vb second vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_getVertices(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFVertexDataHandle *va, struct Mx3DFVertexDataHandle *vb);

/**
 * @brief Set all child vertices
 * 
 * @param handle populated handle
 * @param va first vertex
 * @param vb second vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_setVertices(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFVertexDataHandle *va, struct Mx3DFVertexDataHandle *vb);

/**
 * @brief Get all parent faces
 * 
 * @param handle populated handle
 * @param faces parent faces
 * @param numFaces number of faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_getFaces(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFFaceDataHandle **faces, unsigned int *numFaces);

/**
 * @brief Set all parent faces
 * 
 * @param handle populated handle
 * @param faces parent faces
 * @param numFaces number of faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_setFaces(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFFaceDataHandle *faces, unsigned int numFaces);

/**
 * @brief Get all parent meshes
 * 
 * @param handle populated handle
 * @param meshes parent meshes
 * @param numMeshes number of meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_getMeshes(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFMeshDataHandle **meshes, unsigned int *numMeshes);

/**
 * @brief Get the number of parent faces
 * 
 * @param handle populated handle
 * @param value number of parent faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_getNumFaces(struct Mx3DFEdgeDataHandle *handle, unsigned int *value);

/**
 * @brief Get the number of parent meshes
 * 
 * @param handle populated handle
 * @param value number of parent meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_getNumMeshes(struct Mx3DFEdgeDataHandle *handle, unsigned int *value);

/**
 * @brief Test whether has a vertex
 * 
 * @param handle populated handle
 * @param vertex vertex to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_hasVertex(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFVertexDataHandle *vertex, bool *isIn);

/**
 * @brief Test whether in a face
 * 
 * @param handle populated handle
 * @param face face to test
 * @param isIn flag signifying whether in a face
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_inFace(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFFaceDataHandle *face, bool *isIn);

/**
 * @brief Test whether in a mesh
 * 
 * @param handle populated handle
 * @param mesh mesh to test
 * @param isIn flag signifying whether in a mesh
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_inMesh(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFMeshDataHandle *mesh, bool *isIn);

/**
 * @brief Test whether in a structure
 * 
 * @param handle populated handle
 * @param structure structure to test
 * @param isIn flag signifying whether in a structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFEdgeData_inStructure(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFStructureHandle *structure, bool *isIn);


///////////////////
// Mx3DFFaceData //
///////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_init(struct Mx3DFFaceDataHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_destroy(struct Mx3DFFaceDataHandle *handle);

/**
 * @brief Test whether an instance has a parent structure
 * 
 * @param handle populated handle
 * @param hasStructure flag signifying whether an instance has a parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_hasStructure(struct Mx3DFFaceDataHandle *handle, bool *hasStructure);

/**
 * @brief Get the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_getStructure(struct Mx3DFFaceDataHandle *handle, struct Mx3DFStructureHandle *structure);

/**
 * @brief Set the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_setStructure(struct Mx3DFFaceDataHandle *handle, struct Mx3DFStructureHandle *structure);

/**
 * @brief Get the ID, if any. Unique to its structure and type. -1 if not set.
 * 
 * @param handle populated handle
 * @param value ID, if any; otherwise -1
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_getId(struct Mx3DFFaceDataHandle *handle, int *value);

/**
 * @brief Set the ID
 * 
 * @param handle populated handle
 * @param value ID
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_setId(struct Mx3DFFaceDataHandle *handle, unsigned int value);

/**
 * @brief Get all child vertices
 * 
 * @param handle populated handle
 * @param vertices child vertices
 * @param numVertices number of vertices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_getVertices(struct Mx3DFFaceDataHandle *handle, struct Mx3DFVertexDataHandle **vertices, unsigned int *numVertices);

/**
 * @brief Get the number of child vertices
 * 
 * @param handle populated handle
 * @param value number of child vertices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_getNumVertices(struct Mx3DFFaceDataHandle *handle, unsigned int *value);

/**
 * @brief Get all child edges
 * 
 * @param handle populated handle
 * @param edges child edges
 * @param numEdges number of child edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_getEdges(struct Mx3DFFaceDataHandle *handle, struct Mx3DFEdgeDataHandle **edges, unsigned int *numEdges);

/**
 * @brief Set all child edges
 * 
 * @param handle populated handle
 * @param edges child edges
 * @param numEdges number of child edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_setEdges(struct Mx3DFFaceDataHandle *handle, struct Mx3DFEdgeDataHandle *edges, unsigned int numEdges);

/**
 * @brief Get the number of child edges
 * 
 * @param handle populated handle
 * @param value number of child edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_getNumEdges(struct Mx3DFFaceDataHandle *handle, unsigned int *value);

/**
 * @brief Get all parent meshes
 * 
 * @param handle populated handle
 * @param meshes parent meshes
 * @param numMeshes number of meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_getMeshes(struct Mx3DFFaceDataHandle *handle, struct Mx3DFMeshDataHandle **meshes, unsigned int *numMeshes);

/**
 * @brief Set all parent meshes
 * 
 * @param handle populated handle
 * @param meshes parent meshes
 * @param numMeshes number of meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_setMeshes(struct Mx3DFFaceDataHandle *handle, struct Mx3DFMeshDataHandle *meshes, unsigned int numMeshes);

/**
 * @brief Get the number of parent meshes
 * 
 * @param handle populated handle
 * @param value number of parent meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_getNumMeshes(struct Mx3DFFaceDataHandle *handle, unsigned int *value);

/**
 * @brief Test whether has a vertex
 * 
 * @param handle populated handle
 * @param vertex vertex to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_hasVertex(struct Mx3DFFaceDataHandle *handle, struct Mx3DFVertexDataHandle *vertex, bool *isIn);

/**
 * @brief Test whether has an edge
 * 
 * @param handle populated handle
 * @param edge edge to test
 * @param isIn flag signifying whether has an edge
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_hasEdge(struct Mx3DFFaceDataHandle *handle, struct Mx3DFEdgeDataHandle *edge, bool *isIn);

/**
 * @brief Test whether in a mesh
 * 
 * @param handle populated handle
 * @param mesh mesh to test
 * @param isIn flag signifying whether in a mesh
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_inMesh(struct Mx3DFFaceDataHandle *handle, struct Mx3DFMeshDataHandle *mesh, bool *isIn);

/**
 * @brief Test whether in a structure
 * 
 * @param handle populated handle
 * @param structure structure to test
 * @param isIn flag signifying whether in a structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_inStructure(struct Mx3DFFaceDataHandle *handle, struct Mx3DFStructureHandle *structure, bool *isIn);

/**
 * @brief Get the face normal vector
 * 
 * @param handle populated handle
 * @param normal normal vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_getNormal(struct Mx3DFFaceDataHandle *handle, float **normal);

/**
 * @brief Set the face normal vector
 * 
 * @param handle populated handle
 * @param normal normal vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFFaceData_setNormal(struct Mx3DFFaceDataHandle *handle, float *normal);


///////////////////
// Mx3DFMeshData //
///////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_init(struct Mx3DFMeshDataHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_destroy(struct Mx3DFMeshDataHandle *handle);

/**
 * @brief Test whether an instance has a parent structure
 * 
 * @param handle populated handle
 * @param hasStructure flag signifying whether an instance has a parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_hasStructure(struct Mx3DFMeshDataHandle *handle, bool *hasStructure);

/**
 * @brief Get the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_getStructure(struct Mx3DFMeshDataHandle *handle, struct Mx3DFStructureHandle *structure);

/**
 * @brief Set the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_setStructure(struct Mx3DFMeshDataHandle *handle, struct Mx3DFStructureHandle *structure);

/**
 * @brief Get the ID, if any. Unique to its structure and type. -1 if not set.
 * 
 * @param handle populated handle
 * @param value ID, if any; otherwise -1
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_getId(struct Mx3DFMeshDataHandle *handle, int *value);

/**
 * @brief Set the ID
 * 
 * @param handle populated handle
 * @param value ID
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_setId(struct Mx3DFMeshDataHandle *handle, unsigned int value);

/**
 * @brief Get all child vertices
 * 
 * @param handle populated handle
 * @param vertices child vertices
 * @param numVertices number of vertices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_getVertices(struct Mx3DFMeshDataHandle *handle, struct Mx3DFVertexDataHandle **vertices, unsigned int *numVertices);

/**
 * @brief Get the number of child vertices
 * 
 * @param handle populated handle
 * @param value number of child vertices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_getNumVertices(struct Mx3DFMeshDataHandle *handle, unsigned int *value);

/**
 * @brief Get all child edges
 * 
 * @param handle populated handle
 * @param edges child edges
 * @param numEdges number of child edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_getEdges(struct Mx3DFMeshDataHandle *handle, struct Mx3DFEdgeDataHandle **edges, unsigned int *numEdges);

/**
 * @brief Get the number of child edges
 * 
 * @param handle populated handle
 * @param value number of child edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_getNumEdges(struct Mx3DFMeshDataHandle *handle, unsigned int *value);

/**
 * @brief Get all child faces
 * 
 * @param handle populated handle
 * @param faces child faces
 * @param numFaces number of child faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_getFaces(struct Mx3DFMeshDataHandle *handle, struct Mx3DFFaceDataHandle **faces, unsigned int *numFaces);

/**
 * @brief Get all child faces
 * 
 * @param handle populated handle
 * @param faces child faces
 * @param numFaces number of child faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_setFaces(struct Mx3DFMeshDataHandle *handle, struct Mx3DFFaceDataHandle *faces, unsigned int numFaces);

/**
 * @brief Get the number of child faces
 * 
 * @param handle populated handle
 * @param value number of child faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_getNumFaces(struct Mx3DFMeshDataHandle *handle, unsigned int *value);

/**
 * @brief Test whether has a vertex
 * 
 * @param handle populated handle
 * @param vertex vertex to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_hasVertex(struct Mx3DFMeshDataHandle *handle, struct Mx3DFVertexDataHandle *vertex, bool *isIn);

/**
 * @brief Test whether has an edge
 * 
 * @param handle populated handle
 * @param edge edge to test
 * @param isIn flag signifying whether has an edge
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_hasEdge(struct Mx3DFMeshDataHandle *handle, struct Mx3DFEdgeDataHandle *edge, bool *isIn);

/**
 * @brief Test whether has a face
 * 
 * @param handle populated handle
 * @param face face to test
 * @param isIn flag signifying whether has a face
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_hasFace(struct Mx3DFMeshDataHandle *handle, struct Mx3DFFaceDataHandle *face, bool *isIn);

/**
 * @brief Test whether in a structure
 * 
 * @param handle populated handle
 * @param structure structure to test
 * @param isIn flag signifying whether in a structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_inStructure(struct Mx3DFMeshDataHandle *handle, struct Mx3DFStructureHandle *structure, bool *isIn);

/**
 * @brief Get the mesh name
 * 
 * @param handle populated handle
 * @param name mesh name
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_getName(struct Mx3DFMeshDataHandle *handle, char **name, unsigned int *numChars);

/**
 * @brief Set the mesh name
 * 
 * @param handle populated handle
 * @param name mesh name
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_setName(struct Mx3DFMeshDataHandle *handle, const char *name);

/**
 * @brief Test whether has render data
 * 
 * @param handle populated handle
 * @param hasData flag signifying whether has render data
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_hasRenderData(struct Mx3DFMeshDataHandle *handle, bool *hasData);

/**
 * @brief Get render data
 * 
 * @param handle populated handle
 * @param renderData render data
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_getRenderData(struct Mx3DFMeshDataHandle *handle, struct Mx3DFRenderDataHandle *renderData);

/**
 * @brief Set render data
 * 
 * @param handle populated handle
 * @param renderData render data
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_setRenderData(struct Mx3DFMeshDataHandle *handle, struct Mx3DFRenderDataHandle *renderData);

/**
 * @brief Get the centroid of the mesh
 * 
 * @param handle populated handle
 * @param centroid mesh centroid
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_getCentroid(struct Mx3DFMeshDataHandle *handle, float **centroid);

/**
 * @brief Translate the mesh by a displacement
 * 
 * @param handle populated handle
 * @param displacement translation displacement
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_translate(struct Mx3DFMeshDataHandle *handle, float *displacement);

/**
 * @brief Translate the mesh to a position
 * 
 * @param handle populated handle
 * @param displacement translation displacement
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_translateTo(struct Mx3DFMeshDataHandle *handle, float *position);

/**
 * @brief Rotate the mesh about a point
 * 
 * @param handle populated handle
 * @param rotMat rotation matrix
 * @param rotPt rotation point
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_rotateAt(struct Mx3DFMeshDataHandle *handle, float *rotMat, float *rotPot);

/**
 * @brief Rotate the mesh about its centroid
 * 
 * @param handle populated handle
 * @param rotMat rotation matrix
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_rotate(struct Mx3DFMeshDataHandle *handle, float *rotMat);

/**
 * @brief Scale the mesh about a point
 * 
 * @param handle populated handle
 * @param scales scale coefficients
 * @param scalePt scale point
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_scaleFrom(struct Mx3DFMeshDataHandle *handle, float *scales, float *scalePt);

/**
 * @brief Scale the mesh uniformly about a point
 * 
 * @param handle populated handle
 * @param scale scale coefficient
 * @param scalePt scale point
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_scaleFromS(struct Mx3DFMeshDataHandle *handle, float scale, float *scalePt);

/**
 * @brief Scale the structure about its centroid
 * 
 * @param handle populated handle
 * @param scales scale components
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_scale(struct Mx3DFMeshDataHandle *handle, float *scales);

/**
 * @brief Scale the structure uniformly about its centroid
 * 
 * @param handle populated handle
 * @param scale scale coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFMeshData_scaleS(struct Mx3DFMeshDataHandle *handle, float scale);


////////////////////
// Mx3DFStructure //
////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_init(struct Mx3DFStructureHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_destroy(struct Mx3DFStructureHandle *handle);

/**
 * @brief Get the default radius applied to vertices when generating meshes from point clouds
 * 
 * @param handle populated handle
 * @param vRadiusDef default radius
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_getRadiusDef(struct Mx3DFStructureHandle *handle, float *vRadiusDef);

/**
 * @brief Set the default radius applied to vertices when generating meshes from point clouds
 * 
 * @param handle populated handle
 * @param vRadiusDef default radius
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_setRadiusDef(struct Mx3DFStructureHandle *handle, float vRadiusDef);

/**
 * @brief Load from file
 * 
 * @param handle populated handle
 * @param filePath file absolute path
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_fromFile(struct Mx3DFStructureHandle *handle, const char *filePath);

/**
 * @brief Write to file
 * 
 * @param handle populated handle
 * @param format output format of file
 * @param filePath file absolute path
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_toFile(struct Mx3DFStructureHandle *handle, const char *format, const char *filePath);

/**
 * @brief Flush stucture. All scheduled processes are executed. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_flush(struct Mx3DFStructureHandle *handle);

/**
 * @brief Extend a structure
 * 
 * @param handle populated handle
 * @param s stucture to extend with
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_extend(struct Mx3DFStructureHandle *handle, struct Mx3DFStructureHandle *s);

/**
 * @brief Clear all data of the structure
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_clear(struct Mx3DFStructureHandle *handle);

/**
 * @brief Get all constituent vertices
 * 
 * @param handle populated handle
 * @param vertices child vertices
 * @param numVertices number of vertices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_getVertices(struct Mx3DFStructureHandle *handle, struct Mx3DFVertexDataHandle **vertices, unsigned int *numVertices);

/**
 * @brief Get all constituent edges
 * 
 * @param handle populated handle
 * @param edges child edges
 * @param numEdges number of child edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_getEdges(struct Mx3DFStructureHandle *handle, struct Mx3DFEdgeDataHandle **edges, unsigned int *numEdges);

/**
 * @brief Get all constituent faces
 * 
 * @param handle populated handle
 * @param faces child faces
 * @param numFaces number of child faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_getFaces(struct Mx3DFStructureHandle *handle, struct Mx3DFFaceDataHandle **faces, unsigned int *numFaces);

/**
 * @brief Get all constituent meshes
 * 
 * @param handle populated handle
 * @param meshes child meshes
 * @param numMeshes number of meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_getMeshes(struct Mx3DFStructureHandle *handle, struct Mx3DFMeshDataHandle **meshes, unsigned int *numMeshes);

/**
 * @brief Get the number of constituent vertices
 * 
 * @param handle populated handle
 * @param value number of constituent vertices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_getNumVertices(struct Mx3DFStructureHandle *handle, unsigned int *value);

/**
 * @brief Get the number of constituent edges
 * 
 * @param handle populated handle
 * @param value number of constituent edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_getNumEdges(struct Mx3DFStructureHandle *handle, unsigned int *value);

/**
 * @brief Get the number of constituent faces
 * 
 * @param handle populated handle
 * @param value number of constituent faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_getNumFaces(struct Mx3DFStructureHandle *handle, unsigned int *value);

/**
 * @brief Get the number of constituent meshes
 * 
 * @param handle populated handle
 * @param value number of constituent meshes
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_getNumMeshes(struct Mx3DFStructureHandle *handle, unsigned int *value);

/**
 * @brief Test whether a vertex is a constituent
 * 
 * @param handle populated handle
 * @param vertex vertex to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_hasVertex(struct Mx3DFStructureHandle *handle, struct Mx3DFVertexDataHandle *vertex, bool *isIn);

/**
 * @brief Test whether an edge is a constituent
 * 
 * @param handle populated handle
 * @param edge edge to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_hasEdge(struct Mx3DFStructureHandle *handle, struct Mx3DFEdgeDataHandle *edge, bool *isIn);

/**
 * @brief Test whether a face is a constituent
 * 
 * @param handle populated handle
 * @param face face to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_hasFace(struct Mx3DFStructureHandle *handle, struct Mx3DFFaceDataHandle *face, bool *isIn);

/**
 * @brief Test whether a mesh is a constituent
 * 
 * @param handle populated handle
 * @param mesh mesh to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_hasMesh(struct Mx3DFStructureHandle *handle, struct Mx3DFMeshDataHandle *mesh, bool *isIn);

/**
 * @brief Add a vertex
 * 
 * @param handle populated handle
 * @param vertex vertex to add
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_addVertex(struct Mx3DFStructureHandle *handle, struct Mx3DFVertexDataHandle *vertex);

/**
 * @brief Add an edge and all constituent data
 * 
 * @param handle populated handle
 * @param edge edge to add
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_addEdge(struct Mx3DFStructureHandle *handle, struct Mx3DFEdgeDataHandle *edge);

/**
 * @brief Add a face and all constituent data
 * 
 * @param handle populated handle
 * @param face face to add
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_addFace(struct Mx3DFStructureHandle *handle, struct Mx3DFFaceDataHandle *face);

/**
 * @brief Add a mesh and all constituent data
 * 
 * @param handle populated handle
 * @param mesh mesh to add
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_addMesh(struct Mx3DFStructureHandle *handle, struct Mx3DFMeshDataHandle *mesh);

/**
 * @brief Remove a vertex
 * 
 * @param handle populated handle
 * @param vertex vertex to remove
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_removeVertex(struct Mx3DFStructureHandle *handle, struct Mx3DFVertexDataHandle *vertex);

/**
 * @brief Remove a edge and all constituent data
 * 
 * @param handle populated handle
 * @param edge edge to remove
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_removeEdge(struct Mx3DFStructureHandle *handle, struct Mx3DFEdgeDataHandle *edge);

/**
 * @brief Remove a face and all constituent data
 * 
 * @param handle populated handle
 * @param face face to remove
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_removeFace(struct Mx3DFStructureHandle *handle, struct Mx3DFFaceDataHandle *face);

/**
 * @brief Remove a mesh and all constituent data
 * 
 * @param handle populated handle
 * @param mesh mesh to remove
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_removeMesh(struct Mx3DFStructureHandle *handle, struct Mx3DFMeshDataHandle *mesh);

/**
 * @brief Get the centroid of the structure
 * 
 * @param handle populated handle
 * @param centroid structure centroid
 * @return S_OK on success  
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_getCentroid(struct Mx3DFStructureHandle *handle, float **centroid);

/**
 * @brief Translate the structure by a displacement
 * 
 * @param handle populated handle
 * @param displacement translation displacement
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_translate(struct Mx3DFStructureHandle *handle, float *displacement);

/**
 * @brief Translate the structure to a position
 * 
 * @param handle populated handle
 * @param position translation displacement
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_translateTo(struct Mx3DFStructureHandle *handle, float *position);

/**
 * @brief Rotate the structure about a point
 * 
 * @param handle populated handle
 * @param rotMat rotation matrix
 * @param rotPt rotation point
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_rotateAt(struct Mx3DFStructureHandle *handle, float *rotMat, float *rotPot);

/**
 * @brief Rotate the structure about its centroid
 * 
 * @param handle populated handle
 * @param rotMat rotation matrix
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_rotate(struct Mx3DFStructureHandle *handle, float *rotMat);

/**
 * @brief Scale the structure about a point
 * 
 * @param handle populated handle
 * @param scales scale coefficients
 * @param scalePt scale point
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_scaleFrom(struct Mx3DFStructureHandle *handle, float *scales, float *scalePt);

/**
 * @brief Scale the structure uniformly about a point
 * 
 * @param handle populated handle
 * @param scale scale coefficient
 * @param scalePt scale point
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_scaleFromS(struct Mx3DFStructureHandle *handle, float scale, float *scalePt);

/**
 * @brief Scale the structure about its centroid
 * 
 * @param handle populated handle
 * @param scales scale coefficients
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_scale(struct Mx3DFStructureHandle *handle, float *scales);

/**
 * @brief Scale the structure uniformly about its centroid
 * 
 * @param handle populated handle
 * @param scale scale coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxC3DFStructure_scaleS(struct Mx3DFStructureHandle *handle, float scale);


/////////////////
// MxFIOModule //
/////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param moduleName name of module
 * @param toFile callback to export module data
 * @param fromFile callback to import module data
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFIOModule_init(struct MxFIOModuleHandle *handle, const char *moduleName, MxFIOModuleToFileFcn toFile, MxFIOModuleFromFileFcn fromFile);

/**
 * @brief Destroy an instance
 * 
 * @param handle 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFIOModule_destroy(struct MxFIOModuleHandle *handle);

/**
 * @brief Register a module for I/O events
 * 
 */
CAPI_FUNC(HRESULT) MxCFIOModule_registerIOModule(struct MxFIOModuleHandle *handle);

/**
 * @brief User-facing function to load module data from main import. 
 * 
 * Must only be called after main import. 
 * 
 */
CAPI_FUNC(HRESULT) MxCFIOModule_load(struct MxFIOModuleHandle *handle);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Delete a file element and all child elements
 * 
 * @param handle element to delete
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFIO_deleteElement(struct MxIOElementHandle *handle);

/**
 * @brief Get or generate root element from current simulation state
 * 
 * @param handle root element
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFIO_getMxIORootElement(struct MxIOElementHandle *handle);

/**
 * @brief Release current root element
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFIO_releaseMxIORootElement();

/**
 * @brief Test whether imported data is available. 
 * 
 * @param value true when imported data is available
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCFIO_hasImport(bool *value);

/**
 * @brief Map a particle id from currently imported file data to the created particle on import
 * 
 * @param pid particle id according to import file
 * @param mapId particle id according to simulation state
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIO_mapImportParticleId(unsigned int pid, unsigned int *mapId);

/**
 * @brief Map a particle type id from currently imported file data to the created particle type on import
 * 
 * @param ptid particle type id according to import file
 * @param mapId particle type id according to simulation state
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIO_mapImportParticleTypeId(unsigned int ptid, unsigned int *mapId);

/**
 * @brief Load a 3D format file
 * 
 * @param filePath path of file
 * @param strt 3D format data container
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIO_fromFile3DF(const char *filePath, struct Mx3DFStructureHandle *strt);

/**
 * @brief Export engine state to a 3D format file
 * 
 * @param format format of file
 * @param filePath path of file
 * @param pRefinements mesh refinements applied when generating meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIO_toFile3DF(const char *format, const char *filePath, unsigned int pRefinements);

/**
 * @brief Save a simulation to file
 * 
 * @param saveFilePath absolute path to file
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIO_toFile(const char *saveFilePath);

/**
 * @brief Return a simulation state as a JSON string
 * 
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIO_toString(char **str, unsigned int *numChars);

#endif // _WRAPS_C_MXCIO_H_