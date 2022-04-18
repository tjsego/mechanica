/**
 * @file MxCIO.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxIO and associated features
 * @date 2022-04-06
 */

#include "MxCIO.h"

#include "mechanica_c_private.h"

#include <io/mx_io.h>
#include <io/MxIO.h>
#include <io/Mx3DFStructure.h>


//////////////////
// Module casts //
//////////////////


namespace mx { 

void castC(const MxMetaData &obj, struct MxMetaDataHandle *handle) {
    handle->versionMajor = obj.versionMajor;
    handle->versionMinor = obj.versionMinor;
    handle->versionPatch = obj.versionPatch;
}

MxIOElement *castC(struct MxIOElementHandle *handle) {
    return castC<MxIOElement, MxIOElementHandle>(handle);
}

Mx3DFRenderData *castC(struct Mx3DFRenderDataHandle *handle) {
    return castC<Mx3DFRenderData, Mx3DFRenderDataHandle>(handle);
}

Mx3DFVertexData *castC(struct Mx3DFVertexDataHandle *handle) {
    return castC<Mx3DFVertexData, Mx3DFVertexDataHandle>(handle);
}

Mx3DFEdgeData *castC(struct Mx3DFEdgeDataHandle *handle) {
    return castC<Mx3DFEdgeData, Mx3DFEdgeDataHandle>(handle);
}

Mx3DFFaceData *castC(struct Mx3DFFaceDataHandle *handle) {
    return castC<Mx3DFFaceData, Mx3DFFaceDataHandle>(handle);
}

Mx3DFMeshData *castC(struct Mx3DFMeshDataHandle *handle) {
    return castC<Mx3DFMeshData, Mx3DFMeshDataHandle>(handle);
}

Mx3DFStructure *castC(struct Mx3DFStructureHandle *handle) {
    return castC<Mx3DFStructure, Mx3DFStructureHandle>(handle);
}

}

#define MXIOELEMENTHANDLE_GET(handle, varname) \
    MxIOElement *varname = mx::castC<MxIOElement, MxIOElementHandle>(handle); \
    if(!varname) \
        return E_FAIL;

#define MX3DFRENDERDATAHANDLE_GET(handle, varname) \
    Mx3DFRenderData *varname = mx::castC<Mx3DFRenderData, Mx3DFRenderDataHandle>(handle); \
    if(!varname) \
        return E_FAIL;

#define MX3DFVERTEXDATAHANDLE_GET(handle, varname) \
    Mx3DFVertexData *varname = mx::castC<Mx3DFVertexData, Mx3DFVertexDataHandle>(handle); \
    if(!varname) \
        return E_FAIL;

#define MX3DFEDGEDATAHANDLE_GET(handle, varname) \
    Mx3DFEdgeData *varname = mx::castC<Mx3DFEdgeData, Mx3DFEdgeDataHandle>(handle); \
    if(!varname) \
        return E_FAIL;

#define MX3DFFACEDATAHANDLE_GET(handle, varname) \
    Mx3DFFaceData *varname = mx::castC<Mx3DFFaceData, Mx3DFFaceDataHandle>(handle); \
    if(!varname) \
        return E_FAIL;

#define MX3DFMESHDATAHANDLE_GET(handle, varname) \
    Mx3DFMeshData *varname = mx::castC<Mx3DFMeshData, Mx3DFMeshDataHandle>(handle); \
    if(!varname) \
        return E_FAIL;

#define MX3DFSTRUCTUREHANDLE_GET(handle, varname) \
    Mx3DFStructure *varname = mx::castC<Mx3DFStructure, Mx3DFStructureHandle>(handle); \
    if(!varname) \
        return E_FAIL;


//////////////
// Generics //
//////////////


namespace mx { namespace capi {

template <typename O, typename H> 
HRESULT meshObj_hasStructure(H *handle, bool *hasStructure) {
    O *obj = mx::castC(handle);
    MXCPTRCHECK(obj);
    MXCPTRCHECK(hasStructure);
    *hasStructure = obj->structure != NULL;
    return S_OK;
}

template <typename O, typename H> 
HRESULT meshObj_getStructure(H *handle, struct Mx3DFStructureHandle *structure) {
    O *obj = mx::castC(handle);
    MXCPTRCHECK(obj); MXCPTRCHECK(obj->structure);
    MXCPTRCHECK(structure);
    structure->MxObj = (void*)obj->structure;
    return S_OK;
}

template <typename O, typename H> 
HRESULT meshObj_setStructure(H *handle, struct Mx3DFStructureHandle *structure) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(structure);
    O *obj = mx::castC(handle);
    Mx3DFStructure *_structure = mx::castC(structure);
    MXCPTRCHECK(obj); MXCPTRCHECK(_structure);
    obj->structure = _structure;
    return S_OK;
}

template <typename O, typename H>
HRESULT copyMeshObjFromVec(const std::vector<O*> &vec, H **arr, unsigned int *numArr) {
    MXCPTRCHECK(arr);
    MXCPTRCHECK(numArr);
    *numArr = vec.size();
    if(*numArr > 0) {
        H *_arr = (H*)malloc(*numArr * sizeof(H));
        if(!_arr) 
            return E_OUTOFMEMORY;
        for(unsigned int i = 0; i < *numArr; i++) 
            _arr[i].MxObj = (void*)vec[i];
        *arr = _arr;
    }
    return S_OK;
}

template <typename O, typename H>
HRESULT copyMeshObjToVec(std::vector<O*> &vec, H *arr, unsigned int numArr) {
    MXCPTRCHECK(arr);
    vec.clear();
    H _el;
    for(unsigned int i = 0; i < numArr; i++) {
        _el = arr[i];
        if(!_el.MxObj) 
            return E_FAIL;
        vec.push_back((O*)_el.MxObj);
    }
    return S_OK;
}

}}


////////////////////////
// Function factories //
////////////////////////


struct MxCFIOModule : MxFIOModule {
    std::string _moduleName;
    MxFIOModuleToFileFcn _toFile;
    MxFIOModuleFromFileFcn _fromFile;

    std::string moduleName() { return this->_moduleName; }

    HRESULT toFile(const MxMetaData &metaData, MxIOElement *fileElement) {
        MxMetaDataHandle _metaData;
        MxIOElementHandle _fileElement;
        mx::castC(metaData, &_metaData);
        mx::castC(*fileElement, &_fileElement);
        return this->_toFile(_metaData, &_fileElement);
    }

    HRESULT fromFile(const MxMetaData &metaData, const MxIOElement &fileElement) {
        MxMetaDataHandle _metaData;
        MxIOElementHandle _fileElement;
        mx::castC(metaData, &_metaData);
        mx::castC(fileElement, &_fileElement);
        return this->_fromFile(_metaData, _fileElement);
    }
};


//////////////////////
// MxMetaDataHandle //
//////////////////////


HRESULT MxCMetaData_init(struct MxMetaDataHandle *handle) {
    MxMetaData md;
    handle->versionMajor = md.versionMajor;
    handle->versionMinor = md.versionMinor;
    handle->versionPatch = md.versionPatch;
    return S_OK;
}


//////////////////////
// MxFIOStorageKeys //
//////////////////////


HRESULT MxCFIOStorageKeys_init(struct MxFIOStorageKeysHandle *handle) {
    MXCPTRCHECK(handle);

    handle->KEY_TYPE = new char(MxFIO::KEY_TYPE.length());
    std::strcpy(handle->KEY_TYPE, MxFIO::KEY_TYPE.c_str());

    handle->KEY_VALUE = new char(MxFIO::KEY_VALUE.length());
    std::strcpy(handle->KEY_VALUE, MxFIO::KEY_VALUE.c_str());

    handle->KEY_METADATA = new char(MxFIO::KEY_METADATA.length());
    std::strcpy(handle->KEY_METADATA, MxFIO::KEY_METADATA.c_str());

    handle->KEY_SIMULATOR = new char(MxFIO::KEY_SIMULATOR.length());
    std::strcpy(handle->KEY_SIMULATOR, MxFIO::KEY_SIMULATOR.c_str());

    handle->KEY_UNIVERSE = new char(MxFIO::KEY_UNIVERSE.length());
    std::strcpy(handle->KEY_UNIVERSE, MxFIO::KEY_UNIVERSE.c_str());

    handle->KEY_MODULES = new char(MxFIO::KEY_MODULES.length());
    std::strcpy(handle->KEY_MODULES, MxFIO::KEY_MODULES.c_str());

    return S_OK;
}

HRESULT MxCFIOStorageKeys_destroy(struct MxFIOStorageKeysHandle *handle) {
    delete handle->KEY_TYPE;
    delete handle->KEY_VALUE;
    delete handle->KEY_METADATA;
    delete handle->KEY_SIMULATOR;
    delete handle->KEY_UNIVERSE;
    delete handle->KEY_MODULES;

    handle->KEY_TYPE = NULL;
    handle->KEY_VALUE = NULL;
    handle->KEY_METADATA = NULL;
    handle->KEY_SIMULATOR = NULL;
    handle->KEY_UNIVERSE = NULL;
    handle->KEY_MODULES = NULL;
    return S_OK;
}


/////////////////////
// Mx3DFRenderData //
/////////////////////


HRESULT MxC3DFRenderData_init(struct Mx3DFRenderDataHandle *handle) {
    Mx3DFRenderData *rd = new Mx3DFRenderData();
    handle->MxObj = (void*)rd;
    return S_OK;
}

HRESULT MxC3DFRenderData_destroy(struct Mx3DFRenderDataHandle *handle) {
    return mx::capi::destroyHandle<Mx3DFRenderData, Mx3DFRenderDataHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxC3DFRenderData_getColor(struct Mx3DFRenderDataHandle *handle, float **color) {
    MX3DFRENDERDATAHANDLE_GET(handle, rd);
    if(!color) 
        return E_FAIL;
    MXVECTOR3_COPYFROM(rd->color, (*color));
    return S_OK;
}

HRESULT MxC3DFRenderData_setColor(struct Mx3DFRenderDataHandle *handle, float *color) {
    MX3DFRENDERDATAHANDLE_GET(handle, rd);
    if(!color) 
        return E_FAIL;
    MXVECTOR3_COPYTO(color, rd->color);
    return S_OK;
}


/////////////////
// MxIOElement //
/////////////////


HRESULT MxCIOElement_init(struct MxIOElementHandle *handle) {
    MxIOElement *ioel = new MxIOElement();
    handle->MxObj = (void*)ioel;
    return S_OK;
}

HRESULT MxCIOElement_destroy(struct MxIOElementHandle *handle) {
    return mx::capi::destroyHandle<MxIOElement, MxIOElementHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCIOElement_getType(struct MxIOElementHandle *handle, char **type, unsigned int *numChars) {
    MXIOELEMENTHANDLE_GET(handle, ioel);
    MXCPTRCHECK(numChars);
    mx::capi::str2Char(ioel->type, type, numChars);
    return S_OK;
}

HRESULT MxCIOElement_setType(struct MxIOElementHandle *handle, const char *type) {
    MXIOELEMENTHANDLE_GET(handle, ioel);
    ioel->type = type;
    return S_OK;
}

HRESULT MxCIOElement_getValue(struct MxIOElementHandle *handle, char **value, unsigned int *numChars) {
    MXIOELEMENTHANDLE_GET(handle, ioel);
    MXCPTRCHECK(value);
    MXCPTRCHECK(numChars);
    return mx::capi::str2Char(ioel->value, value, numChars);
}

HRESULT MxCIOElement_setValue(struct MxIOElementHandle *handle, const char *value) {
    MXIOELEMENTHANDLE_GET(handle, ioel);
    MXCPTRCHECK(value);
    ioel->value = value;
    return S_OK;
}

HRESULT MxCIOElement_hasParent(struct MxIOElementHandle *handle, bool *hasParent) {
    MXIOELEMENTHANDLE_GET(handle, ioel);
    MXCPTRCHECK(hasParent);
    *hasParent = ioel->parent != NULL;
    return S_OK;
}

HRESULT MxCIOElement_getParent(struct MxIOElementHandle *handle, struct MxIOElementHandle *parent) {
    MXIOELEMENTHANDLE_GET(handle, ioel);
    MXCPTRCHECK(ioel->parent);
    MXCPTRCHECK(parent);
    parent->MxObj = (void*)ioel->parent;
    return S_OK;
}

HRESULT MxCIOElement_setParent(struct MxIOElementHandle *handle, struct MxIOElementHandle *parent) {
    MXIOELEMENTHANDLE_GET(handle, ioel);
    MXIOELEMENTHANDLE_GET(parent, pioel);
    ioel->parent = pioel;
    return S_OK;
}

HRESULT MxCIOElement_getNumChildren(struct MxIOElementHandle *handle, unsigned int *numChildren) {
    MXIOELEMENTHANDLE_GET(handle, ioel);
    MXCPTRCHECK(numChildren);
    *numChildren = ioel->children.size();
    return S_OK;
}

HRESULT MxCIOElement_getKeys(struct MxIOElementHandle *handle, char ***keys) {
    MXIOELEMENTHANDLE_GET(handle, ioel);
    if(!keys) 
        return E_FAIL;
    auto numChildren = ioel->children.size();
    if(numChildren > 0) {
        char **_keys = (char**)malloc(numChildren * sizeof(char*));
        if(!_keys) 
            return E_OUTOFMEMORY;
        unsigned int i = 0;
        for(auto &itr : ioel->children) {
            char *_c = new char[itr.first.size() + 1];
            std::strcpy(_c, itr.first.c_str());
            _keys[i] = _c;
            i++;
        }
        *keys = _keys;
    }
    return S_OK;
}

HRESULT MxCIOElement_getChild(struct MxIOElementHandle *handle, const char *key, struct MxIOElementHandle *child) {
    MXIOELEMENTHANDLE_GET(handle, ioel);
    MXCPTRCHECK(child);
    auto itr = ioel->children.find(key);
    if(itr == ioel->children.end()) 
        return E_FAIL;
    child->MxObj = (void*)itr->second;
    return S_OK;
}

HRESULT MxCIOElement_setChild(struct MxIOElementHandle *handle, const char *key, struct MxIOElementHandle *child) {
    MXIOELEMENTHANDLE_GET(handle, ioel);
    MXCPTRCHECK(key);
    MXIOELEMENTHANDLE_GET(child, cioel);
    ioel->children.insert({key, cioel});
    return S_OK;
}


/////////////////////
// Mx3DFVertexData //
/////////////////////


HRESULT MxC3DFVertexData_init(struct Mx3DFVertexDataHandle *handle, float *position) {
    if(!position) 
        return E_FAIL;
    Mx3DFVertexData *vert = new Mx3DFVertexData(MxVector3f::from(position));
    handle->MxObj = (void*)vert;
    return S_OK;
}

HRESULT MxC3DFVertexData_destroy(struct Mx3DFVertexDataHandle *handle) {
    return mx::capi::destroyHandle<Mx3DFVertexData, Mx3DFVertexDataHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxC3DFVertexData_hasStructure(struct Mx3DFVertexDataHandle *handle, bool *hasStructure) {
    return mx::capi::meshObj_hasStructure<Mx3DFVertexData, Mx3DFVertexDataHandle>(handle, hasStructure);
}

HRESULT MxC3DFVertexData_getStructure(struct Mx3DFVertexDataHandle *handle, struct Mx3DFStructureHandle *structure) {
    return mx::capi::meshObj_getStructure<Mx3DFVertexData, Mx3DFVertexDataHandle>(handle, structure);
}

HRESULT MxC3DFVertexData_setStructure(struct Mx3DFVertexDataHandle *handle, struct Mx3DFStructureHandle *structure) {
    return mx::capi::meshObj_setStructure<Mx3DFVertexData, Mx3DFVertexDataHandle>(handle, structure);
}

HRESULT MxC3DFVertexData_getPosition(struct Mx3DFVertexDataHandle *handle, float **position) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    if(!position) 
        return E_FAIL;
    MXVECTOR3_COPYFROM(vert->position, (*position));
    return S_OK;
}

HRESULT MxC3DFVertexData_setPosition(struct Mx3DFVertexDataHandle *handle, float *position) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    if(!position) 
        return E_FAIL;
    MXVECTOR3_COPYTO(position, vert->position);
    return S_OK;
}

HRESULT MxC3DFVertexData_getId(struct Mx3DFVertexDataHandle *handle, int *value) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    MXCPTRCHECK(value);
    *value = vert->id;
    return S_OK;
}

HRESULT MxC3DFVertexData_setId(struct Mx3DFVertexDataHandle *handle, unsigned int value) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    vert->id = value;
    return S_OK;
}

HRESULT MxC3DFVertexData_getEdges(struct Mx3DFVertexDataHandle *handle, struct Mx3DFEdgeDataHandle **edges, unsigned int *numEdges) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    return mx::capi::copyMeshObjFromVec(vert->edges, edges, numEdges);
}

HRESULT MxC3DFVertexData_setEdges(struct Mx3DFVertexDataHandle *handle, struct Mx3DFEdgeDataHandle *edges, unsigned int numEdges) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    return mx::capi::copyMeshObjToVec(vert->edges, edges, numEdges);
}

HRESULT MxC3DFVertexData_getFaces(struct Mx3DFVertexDataHandle *handle, struct Mx3DFFaceDataHandle **faces, unsigned int *numFaces) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    return mx::capi::copyMeshObjFromVec(vert->getFaces(), faces, numFaces);
}

HRESULT MxC3DFVertexData_getMeshes(struct Mx3DFVertexDataHandle *handle, struct Mx3DFMeshDataHandle **meshes, unsigned int *numMeshes) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    return mx::capi::copyMeshObjFromVec(vert->getMeshes(), meshes, numMeshes);
}

HRESULT MxC3DFVertexData_getNumEdges(struct Mx3DFVertexDataHandle *handle, unsigned int *value) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    MXCPTRCHECK(value);
    *value = vert->getNumEdges();
    return S_OK;
}

HRESULT MxC3DFVertexData_getNumFaces(struct Mx3DFVertexDataHandle *handle, unsigned int *value) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    MXCPTRCHECK(value);
    *value = vert->getNumFaces();
    return S_OK;
}

HRESULT MxC3DFVertexData_getNumMeshes(struct Mx3DFVertexDataHandle *handle, unsigned int *value) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    MXCPTRCHECK(value);
    *value = vert->getNumMeshes();
    return S_OK;
}

HRESULT MxC3DFVertexData_inEdge(struct Mx3DFVertexDataHandle *handle, struct Mx3DFEdgeDataHandle *edge, bool *isIn) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    MX3DFEDGEDATAHANDLE_GET(edge, _edge);
    MXCPTRCHECK(isIn);
    *isIn = vert->in(_edge);
    return S_OK;
}

HRESULT MxC3DFVertexData_inFace(struct Mx3DFVertexDataHandle *handle, struct Mx3DFFaceDataHandle *face, bool *isIn) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    MX3DFFACEDATAHANDLE_GET(face, _face);
    MXCPTRCHECK(isIn);
    *isIn = vert->in(_face);
    return S_OK;
}

HRESULT MxC3DFVertexData_inMesh(struct Mx3DFVertexDataHandle *handle, struct Mx3DFMeshDataHandle *mesh, bool *isIn) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    MX3DFMESHDATAHANDLE_GET(mesh, _mesh);
    MXCPTRCHECK(isIn);
    *isIn = vert->in(_mesh);
    return S_OK;
}

HRESULT MxC3DFVertexData_inStructure(struct Mx3DFVertexDataHandle *handle, struct Mx3DFStructureHandle *structure, bool *isIn) {
    MX3DFVERTEXDATAHANDLE_GET(handle, vert);
    MX3DFSTRUCTUREHANDLE_GET(structure, _structure);
    MXCPTRCHECK(isIn);
    *isIn = vert->in(_structure);
    return S_OK;
}


///////////////////
// Mx3DFEdgeData //
///////////////////


HRESULT MxC3DFEdgeData_init(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFVertexDataHandle *va, struct Mx3DFVertexDataHandle *vb) {
    MXCPTRCHECK(handle);
    MX3DFVERTEXDATAHANDLE_GET(va, _va);
    MX3DFVERTEXDATAHANDLE_GET(vb, _vb);
    Mx3DFEdgeData *edge = new Mx3DFEdgeData(_va, _vb);
    handle->MxObj = (void*)edge;
    return S_OK;
}

HRESULT MxC3DFEdgeData_destroy(struct Mx3DFEdgeDataHandle *handle) {
    return mx::capi::destroyHandle<Mx3DFEdgeData, Mx3DFEdgeDataHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxC3DFEdgeData_hasStructure(struct Mx3DFEdgeDataHandle *handle, bool *hasStructure) {
    return mx::capi::meshObj_hasStructure<Mx3DFEdgeData, Mx3DFEdgeDataHandle>(handle, hasStructure);
}

HRESULT MxC3DFEdgeData_getStructure(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFStructureHandle *structure) {
    return mx::capi::meshObj_getStructure<Mx3DFEdgeData, Mx3DFEdgeDataHandle>(handle, structure);
}

HRESULT MxC3DFEdgeData_setStructure(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFStructureHandle *structure) {
    return mx::capi::meshObj_setStructure<Mx3DFEdgeData, Mx3DFEdgeDataHandle>(handle, structure);
}

HRESULT MxC3DFEdgeData_getId(struct Mx3DFEdgeDataHandle *handle, int *value) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    MXCPTRCHECK(value);
    *value = edge->id;
    return S_OK;
}

HRESULT MxC3DFEdgeData_setId(struct Mx3DFEdgeDataHandle *handle, unsigned int value) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    edge->id = value;
    return S_OK;
}

HRESULT MxC3DFEdgeData_getVertices(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFVertexDataHandle *va, struct Mx3DFVertexDataHandle *vb) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    MXCPTRCHECK(va);
    MXCPTRCHECK(vb);
    va->MxObj = (void*)edge->va;
    vb->MxObj = (void*)edge->vb;
    return S_OK;
}

HRESULT MxC3DFEdgeData_setVertices(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFVertexDataHandle *va, struct Mx3DFVertexDataHandle *vb) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    MX3DFVERTEXDATAHANDLE_GET(va, _va);
    MX3DFVERTEXDATAHANDLE_GET(vb, _vb);
    edge->va = _va;
    edge->vb = _vb;
    return S_OK;
}

HRESULT MxC3DFEdgeData_getFaces(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFFaceDataHandle **faces, unsigned int *numFaces) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    return mx::capi::copyMeshObjFromVec(edge->getFaces(), faces, numFaces);
}

HRESULT MxC3DFEdgeData_setFaces(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFFaceDataHandle *faces, unsigned int numFaces) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    return mx::capi::copyMeshObjToVec(edge->faces, faces, numFaces);
}

HRESULT MxC3DFEdgeData_getMeshes(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFMeshDataHandle **meshes, unsigned int *numMeshes) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    return mx::capi::copyMeshObjFromVec(edge->getMeshes(), meshes, numMeshes);
}

HRESULT MxC3DFEdgeData_getNumFaces(struct Mx3DFEdgeDataHandle *handle, unsigned int *value) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    MXCPTRCHECK(value);
    *value = edge->getNumFaces();
    return S_OK;
}

HRESULT MxC3DFEdgeData_getNumMeshes(struct Mx3DFEdgeDataHandle *handle, unsigned int *value) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    MXCPTRCHECK(value);
    *value = edge->getNumMeshes();
    return S_OK;
}

HRESULT MxC3DFEdgeData_hasVertex(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFVertexDataHandle *vertex, bool *isIn) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    MX3DFVERTEXDATAHANDLE_GET(vertex, _vertex);
    MXCPTRCHECK(isIn);
    *isIn = edge->has(_vertex);
    return S_OK;
}

HRESULT MxC3DFEdgeData_inFace(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFFaceDataHandle *face, bool *isIn) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    MX3DFFACEDATAHANDLE_GET(face, _face);
    MXCPTRCHECK(isIn);
    *isIn = edge->in(_face);
    return S_OK;
}

HRESULT MxC3DFEdgeData_inMesh(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFMeshDataHandle *mesh, bool *isIn) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    MX3DFMESHDATAHANDLE_GET(mesh, _mesh);
    MXCPTRCHECK(isIn);
    *isIn = edge->in(_mesh);
    return S_OK;
}

HRESULT MxC3DFEdgeData_inStructure(struct Mx3DFEdgeDataHandle *handle, struct Mx3DFStructureHandle *structure, bool *isIn) {
    MX3DFEDGEDATAHANDLE_GET(handle, edge);
    MX3DFSTRUCTUREHANDLE_GET(structure, _structure);
    MXCPTRCHECK(isIn);
    *isIn = edge->in(_structure);
    return S_OK;
}


///////////////////
// Mx3DFFaceData //
///////////////////


HRESULT MxC3DFFaceData_init(struct Mx3DFFaceDataHandle *handle) {
    Mx3DFFaceData *face = new Mx3DFFaceData();
    handle->MxObj = (void*)face;
    return S_OK;
}

HRESULT MxC3DFFaceData_destroy(struct Mx3DFFaceDataHandle *handle) {
    return mx::capi::destroyHandle<Mx3DFFaceData, Mx3DFFaceDataHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxC3DFFaceData_hasStructure(struct Mx3DFFaceDataHandle *handle, bool *hasStructure) {
    return mx::capi::meshObj_hasStructure<Mx3DFFaceData, Mx3DFFaceDataHandle>(handle, hasStructure);
}

HRESULT MxC3DFFaceData_getStructure(struct Mx3DFFaceDataHandle *handle, struct Mx3DFStructureHandle *structure) {
    return mx::capi::meshObj_getStructure<Mx3DFFaceData, Mx3DFFaceDataHandle>(handle, structure);
}

HRESULT MxC3DFFaceData_setStructure(struct Mx3DFFaceDataHandle *handle, struct Mx3DFStructureHandle *structure) {
    return mx::capi::meshObj_setStructure<Mx3DFFaceData, Mx3DFFaceDataHandle>(handle, structure);
}

HRESULT MxC3DFFaceData_getId(struct Mx3DFFaceDataHandle *handle, int *value) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    MXCPTRCHECK(value);
    *value = face->id;
    return S_OK;
}

HRESULT MxC3DFFaceData_setId(struct Mx3DFFaceDataHandle *handle, unsigned int value) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    face->id = value;
    return S_OK;
}

HRESULT MxC3DFFaceData_getVertices(struct Mx3DFFaceDataHandle *handle, struct Mx3DFVertexDataHandle **vertices, unsigned int *numVertices) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    return mx::capi::copyMeshObjFromVec(face->getVertices(), vertices, numVertices);
}

HRESULT MxC3DFFaceData_getNumVertices(struct Mx3DFFaceDataHandle *handle, unsigned int *value) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    MXCPTRCHECK(value);
    *value = face->getNumVertices();
    return S_OK;
}

HRESULT MxC3DFFaceData_getEdges(struct Mx3DFFaceDataHandle *handle, struct Mx3DFEdgeDataHandle **edges, unsigned int *numEdges) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    return mx::capi::copyMeshObjFromVec(face->edges, edges, numEdges);
}

HRESULT MxC3DFFaceData_setEdges(struct Mx3DFFaceDataHandle *handle, struct Mx3DFEdgeDataHandle *edges, unsigned int numEdges) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    if(!edges) 
        return E_FAIL;
    return mx::capi::copyMeshObjToVec(face->edges, edges, numEdges);
}

HRESULT MxC3DFFaceData_getNumEdges(struct Mx3DFFaceDataHandle *handle, unsigned int *value) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    MXCPTRCHECK(value);
    *value = face->getNumEdges();
    return S_OK;
}

HRESULT MxC3DFFaceData_getMeshes(struct Mx3DFFaceDataHandle *handle, struct Mx3DFMeshDataHandle **meshes, unsigned int *numMeshes) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    return mx::capi::copyMeshObjFromVec(face->meshes, meshes, numMeshes);
}

HRESULT MxC3DFFaceData_setMeshes(struct Mx3DFFaceDataHandle *handle, struct Mx3DFMeshDataHandle *meshes, unsigned int numMeshes) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    if(!meshes) 
        return E_FAIL;
    return mx::capi::copyMeshObjToVec(face->meshes, meshes, numMeshes);
}

HRESULT MxC3DFFaceData_getNumMeshes(struct Mx3DFFaceDataHandle *handle, unsigned int *value) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    MXCPTRCHECK(value);
    *value = face->getNumMeshes();
    return S_OK;
}

HRESULT MxC3DFFaceData_hasVertex(struct Mx3DFFaceDataHandle *handle, struct Mx3DFVertexDataHandle *vertex, bool *isIn) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    MX3DFVERTEXDATAHANDLE_GET(vertex, _vertex);
    MXCPTRCHECK(isIn);
    *isIn = face->has(_vertex);
    return S_OK;
}

HRESULT MxC3DFFaceData_hasEdge(struct Mx3DFFaceDataHandle *handle, struct Mx3DFEdgeDataHandle *edge, bool *isIn) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    MX3DFEDGEDATAHANDLE_GET(edge, _edge);
    MXCPTRCHECK(isIn);
    *isIn = face->has(_edge);
    return S_OK;
}

HRESULT MxC3DFFaceData_inMesh(struct Mx3DFFaceDataHandle *handle, struct Mx3DFMeshDataHandle *mesh, bool *isIn) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    MX3DFMESHDATAHANDLE_GET(mesh, _mesh);
    MXCPTRCHECK(isIn);
    *isIn = face->in(_mesh);
    return S_OK;
}

HRESULT MxC3DFFaceData_inStructure(struct Mx3DFFaceDataHandle *handle, struct Mx3DFStructureHandle *structure, bool *isIn) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    MX3DFSTRUCTUREHANDLE_GET(structure, _structure);
    MXCPTRCHECK(isIn);
    *isIn = face->in(_structure);
    return S_OK;
}

HRESULT MxC3DFFaceData_getNormal(struct Mx3DFFaceDataHandle *handle, float **normal) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    MXCPTRCHECK(normal);
    MXVECTOR3_COPYFROM(face->normal, (*normal));
    return S_OK;
}

HRESULT MxC3DFFaceData_setNormal(struct Mx3DFFaceDataHandle *handle, float *normal) {
    MX3DFFACEDATAHANDLE_GET(handle, face);
    MXCPTRCHECK(normal);
    MXVECTOR3_COPYTO(normal, face->normal);
    return S_OK;
}


///////////////////
// Mx3DFMeshData //
///////////////////


HRESULT MxC3DFMeshData_init(struct Mx3DFMeshDataHandle *handle) {
    Mx3DFMeshData *mesh = new Mx3DFMeshData();
    handle->MxObj = (void*)mesh;
    return S_OK;
}

HRESULT MxC3DFMeshData_destroy(struct Mx3DFMeshDataHandle *handle) {
    return mx::capi::destroyHandle<Mx3DFMeshData, Mx3DFMeshDataHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxC3DFMeshData_hasStructure(struct Mx3DFMeshDataHandle *handle, bool *hasStructure) {
    return mx::capi::meshObj_hasStructure<Mx3DFMeshData, Mx3DFMeshDataHandle>(handle, hasStructure);
}

HRESULT MxC3DFMeshData_getStructure(struct Mx3DFMeshDataHandle *handle, struct Mx3DFStructureHandle *structure) {
    return mx::capi::meshObj_getStructure<Mx3DFMeshData, Mx3DFMeshDataHandle>(handle, structure);
}

HRESULT MxC3DFMeshData_setStructure(struct Mx3DFMeshDataHandle *handle, struct Mx3DFStructureHandle *structure) {
    return mx::capi::meshObj_setStructure<Mx3DFMeshData, Mx3DFMeshDataHandle>(handle, structure);
}

HRESULT MxC3DFMeshData_getId(struct Mx3DFMeshDataHandle *handle, int *value) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MXCPTRCHECK(value);
    *value = mesh->id;
    return S_OK;
}

HRESULT MxC3DFMeshData_setId(struct Mx3DFMeshDataHandle *handle, unsigned int value) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    mesh->id = value;
    return S_OK;
}

HRESULT MxC3DFMeshData_getVertices(struct Mx3DFMeshDataHandle *handle, struct Mx3DFVertexDataHandle **vertices, unsigned int *numVertices) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    return mx::capi::copyMeshObjFromVec(mesh->getVertices(), vertices, numVertices);
}

HRESULT MxC3DFMeshData_getNumVertices(struct Mx3DFMeshDataHandle *handle, unsigned int *value) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MXCPTRCHECK(value);
    *value = mesh->getNumVertices();
    return S_OK;
}

HRESULT MxC3DFMeshData_getEdges(struct Mx3DFMeshDataHandle *handle, struct Mx3DFEdgeDataHandle **edges, unsigned int *numEdges) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    return mx::capi::copyMeshObjFromVec(mesh->getEdges(), edges, numEdges);
}

HRESULT MxC3DFMeshData_getNumEdges(struct Mx3DFMeshDataHandle *handle, unsigned int *value) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MXCPTRCHECK(value);
    *value = mesh->getNumEdges();
    return S_OK;
}

HRESULT MxC3DFMeshData_getFaces(struct Mx3DFMeshDataHandle *handle, struct Mx3DFFaceDataHandle **faces, unsigned int *numFaces) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    return mx::capi::copyMeshObjFromVec(mesh->getFaces(), faces, numFaces);
}

HRESULT MxC3DFMeshData_setFaces(struct Mx3DFMeshDataHandle *handle, struct Mx3DFFaceDataHandle *faces, unsigned int numFaces) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!faces)
        return E_FAIL;
    return mx::capi::copyMeshObjToVec(mesh->faces, faces, numFaces);
}

HRESULT MxC3DFMeshData_getNumFaces(struct Mx3DFMeshDataHandle *handle, unsigned int *value) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MXCPTRCHECK(value);
    *value = mesh->getNumFaces();
    return S_OK;
}

HRESULT MxC3DFMeshData_hasVertex(struct Mx3DFMeshDataHandle *handle, struct Mx3DFVertexDataHandle *vertex, bool *isIn) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MX3DFVERTEXDATAHANDLE_GET(vertex, _vertex);
    MXCPTRCHECK(isIn);
    *isIn = mesh->has(_vertex);
    return S_OK;
}

HRESULT MxC3DFMeshData_hasEdge(struct Mx3DFMeshDataHandle *handle, struct Mx3DFEdgeDataHandle *edge, bool *isIn) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MX3DFEDGEDATAHANDLE_GET(edge, _edge);
    MXCPTRCHECK(isIn);
    *isIn = mesh->has(_edge);
    return S_OK;
}

HRESULT MxC3DFMeshData_hasFace(struct Mx3DFMeshDataHandle *handle, struct Mx3DFFaceDataHandle *face, bool *isIn) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MX3DFFACEDATAHANDLE_GET(face, _face);
    MXCPTRCHECK(isIn);
    *isIn = mesh->has(_face);
    return S_OK;
}

HRESULT MxC3DFMeshData_inStructure(struct Mx3DFMeshDataHandle *handle, struct Mx3DFStructureHandle *structure, bool *isIn) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MX3DFSTRUCTUREHANDLE_GET(structure, _structure);
    MXCPTRCHECK(isIn);
    *isIn = mesh->in(_structure);
    return S_OK;
}

HRESULT MxC3DFMeshData_getName(struct Mx3DFMeshDataHandle *handle, char **name, unsigned int *numChars) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MXCPTRCHECK(name);
    MXCPTRCHECK(numChars);
    char *_name = new char[mesh->name.size() + 1];
    std::strcpy(_name, mesh->name.c_str());
    *numChars = mesh->name.size() + 1;
    return S_OK;
}

HRESULT MxC3DFMeshData_setName(struct Mx3DFMeshDataHandle *handle, const char *name) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MXCPTRCHECK(name);
    mesh->name = name;
    return S_OK;
}

HRESULT MxC3DFMeshData_hasRenderData(struct Mx3DFMeshDataHandle *handle, bool *hasData) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MXCPTRCHECK(hasData);
    *hasData = mesh->renderData != NULL;
    return S_OK;
}

HRESULT MxC3DFMeshData_getRenderData(struct Mx3DFMeshDataHandle *handle, struct Mx3DFRenderDataHandle *renderData) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MXCPTRCHECK(mesh->renderData);
    MXCPTRCHECK(renderData);
    renderData->MxObj = (void*)mesh->renderData;
    return S_OK;
}

HRESULT MxC3DFMeshData_setRenderData(struct Mx3DFMeshDataHandle *handle, struct Mx3DFRenderDataHandle *renderData) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    MX3DFRENDERDATAHANDLE_GET(renderData, _renderData);
    mesh->renderData = _renderData;
    return S_OK;
}

HRESULT MxC3DFMeshData_getCentroid(struct Mx3DFMeshDataHandle *handle, float **centroid) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!centroid) 
        return E_FAIL;
    auto _centroid = mesh->getCentroid();
    MXVECTOR3_COPYFROM(_centroid, (*centroid));
    return S_OK;
}

HRESULT MxC3DFMeshData_translate(struct Mx3DFMeshDataHandle *handle, float *displacement) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!displacement) 
        return E_FAIL;
    return mesh->translate(MxVector3f::from(displacement));
}

HRESULT MxC3DFMeshData_translateTo(struct Mx3DFMeshDataHandle *handle, float *position) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!position) 
        return E_FAIL;
    return mesh->translateTo(MxVector3f::from(position));
}

HRESULT MxC3DFMeshData_rotateAt(struct Mx3DFMeshDataHandle *handle, float *rotMat, float *rotPot) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!rotMat || !rotPot) 
        return E_FAIL;
    MxMatrix3f _rotMat;
    MXMATRIX3_COPYTO(rotMat, _rotMat);
    return mesh->rotateAt(_rotMat, MxVector3f::from(rotPot));
}

HRESULT MxC3DFMeshData_rotate(struct Mx3DFMeshDataHandle *handle, float *rotMat) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!rotMat) 
        return E_FAIL;
    MxMatrix3f _rotMat;
    MXMATRIX3_COPYTO(rotMat, _rotMat);
    return mesh->rotate(_rotMat);
}

HRESULT MxC3DFMeshData_scaleFrom(struct Mx3DFMeshDataHandle *handle, float *scales, float *scalePt) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!scales || !scalePt) 
        return E_FAIL;
    return mesh->scaleFrom(MxVector3f::from(scales), MxVector3f::from(scalePt));
}

HRESULT MxC3DFMeshData_scaleFromS(struct Mx3DFMeshDataHandle *handle, float scale, float *scalePt) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!scalePt) 
        return E_FAIL;
    return mesh->scaleFrom(scale, MxVector3f::from(scalePt));
}

HRESULT MxC3DFMeshData_scale(struct Mx3DFMeshDataHandle *handle, float *scales) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!scales) 
        return E_FAIL;
    return mesh->scale(MxVector3f::from(scales));
}

HRESULT MxC3DFMeshData_scaleS(struct Mx3DFMeshDataHandle *handle, float scale) {
    MX3DFMESHDATAHANDLE_GET(handle, mesh);
    return mesh->scale(scale);
}


////////////////////
// Mx3DFStructure //
////////////////////


HRESULT MxC3DFStructure_init(struct Mx3DFStructureHandle *handle) {
    Mx3DFStructure *strt = new Mx3DFStructure();
    handle->MxObj = (void*)strt;
    return S_OK;
}

HRESULT MxC3DFStructure_destroy(struct Mx3DFStructureHandle *handle) {
    return mx::capi::destroyHandle<Mx3DFStructure, Mx3DFStructureHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxC3DFStructure_getRadiusDef(struct Mx3DFStructureHandle *handle, float *vRadiusDef) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MXCPTRCHECK(vRadiusDef);
    *vRadiusDef = strt->vRadiusDef;
    return S_OK;
}

HRESULT MxC3DFStructure_setRadiusDef(struct Mx3DFStructureHandle *handle, float vRadiusDef) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    strt->vRadiusDef = vRadiusDef;
    return S_OK;
}

HRESULT MxC3DFStructure_fromFile(struct Mx3DFStructureHandle *handle, const char *filePath) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    return strt->fromFile(filePath);
}

HRESULT MxC3DFStructure_toFile(struct Mx3DFStructureHandle *handle, const char *format, const char *filePath) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    return strt->toFile(format, filePath);
}

HRESULT MxC3DFStructure_flush(struct Mx3DFStructureHandle *handle) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    return strt->flush();
}

HRESULT MxC3DFStructure_extend(struct Mx3DFStructureHandle *handle, struct Mx3DFStructureHandle *s) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFSTRUCTUREHANDLE_GET(s, _s);
    return strt->extend(*_s);
}

HRESULT MxC3DFStructure_clear(struct Mx3DFStructureHandle *handle) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    return strt->clear();
}

HRESULT MxC3DFStructure_getVertices(struct Mx3DFStructureHandle *handle, struct Mx3DFVertexDataHandle **vertices, unsigned int *numVertices) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    return mx::capi::copyMeshObjFromVec(strt->getVertices(), vertices, numVertices);
}

HRESULT MxC3DFStructure_getEdges(struct Mx3DFStructureHandle *handle, struct Mx3DFEdgeDataHandle **edges, unsigned int *numEdges) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    return mx::capi::copyMeshObjFromVec(strt->getEdges(), edges, numEdges);
}

HRESULT MxC3DFStructure_getFaces(struct Mx3DFStructureHandle *handle, struct Mx3DFFaceDataHandle **faces, unsigned int *numFaces) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    return mx::capi::copyMeshObjFromVec(strt->getFaces(), faces, numFaces);
}

HRESULT MxC3DFStructure_getMeshes(struct Mx3DFStructureHandle *handle, struct Mx3DFMeshDataHandle **meshes, unsigned int *numMeshes) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    return mx::capi::copyMeshObjFromVec(strt->getMeshes(), meshes, numMeshes);
}

HRESULT MxC3DFStructure_getNumVertices(struct Mx3DFStructureHandle *handle, unsigned int *value) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MXCPTRCHECK(value);
    *value = strt->getNumVertices();
    return S_OK;
}

HRESULT MxC3DFStructure_getNumEdges(struct Mx3DFStructureHandle *handle, unsigned int *value) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MXCPTRCHECK(value);
    *value = strt->getNumEdges();
    return S_OK;
}

HRESULT MxC3DFStructure_getNumFaces(struct Mx3DFStructureHandle *handle, unsigned int *value) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MXCPTRCHECK(value);
    *value = strt->getNumFaces();
    return S_OK;
}

HRESULT MxC3DFStructure_getNumMeshes(struct Mx3DFStructureHandle *handle, unsigned int *value) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MXCPTRCHECK(value);
    *value = strt->getNumMeshes();
    return S_OK;
}

HRESULT MxC3DFStructure_hasVertex(struct Mx3DFStructureHandle *handle, struct Mx3DFVertexDataHandle *vertex, bool *isIn) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFVERTEXDATAHANDLE_GET(vertex, _vertex);
    MXCPTRCHECK(isIn);
    *isIn = strt->has(_vertex);
    return S_OK;
}

HRESULT MxC3DFStructure_hasEdge(struct Mx3DFStructureHandle *handle, struct Mx3DFEdgeDataHandle *edge, bool *isIn) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFEDGEDATAHANDLE_GET(edge, _edge);
    MXCPTRCHECK(isIn);
    *isIn = strt->has(_edge);
    return S_OK;
}

HRESULT MxC3DFStructure_hasFace(struct Mx3DFStructureHandle *handle, struct Mx3DFFaceDataHandle *face, bool *isIn) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFFACEDATAHANDLE_GET(face, _face);
    MXCPTRCHECK(isIn);
    *isIn = strt->has(_face);
    return S_OK;
}

HRESULT MxC3DFStructure_hasMesh(struct Mx3DFStructureHandle *handle, struct Mx3DFMeshDataHandle *mesh, bool *isIn) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFMESHDATAHANDLE_GET(mesh, _mesh);
    MXCPTRCHECK(isIn);
    *isIn = strt->has(_mesh);
    return S_OK;
}

HRESULT MxC3DFStructure_addVertex(struct Mx3DFStructureHandle *handle, struct Mx3DFVertexDataHandle *vertex) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFVERTEXDATAHANDLE_GET(vertex, _vertex);
    strt->add(_vertex);
    return S_OK;
}

HRESULT MxC3DFStructure_addEdge(struct Mx3DFStructureHandle *handle, struct Mx3DFEdgeDataHandle *edge) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFEDGEDATAHANDLE_GET(edge, _edge);
    strt->add(_edge);
    return S_OK;
}

HRESULT MxC3DFStructure_addFace(struct Mx3DFStructureHandle *handle, struct Mx3DFFaceDataHandle *face) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFFACEDATAHANDLE_GET(face, _face);
    strt->add(_face);
    return S_OK;
}

HRESULT MxC3DFStructure_addMesh(struct Mx3DFStructureHandle *handle, struct Mx3DFMeshDataHandle *mesh) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFMESHDATAHANDLE_GET(mesh, _mesh);
    strt->add(_mesh);
    return S_OK;
}

HRESULT MxC3DFStructure_removeVertex(struct Mx3DFStructureHandle *handle, struct Mx3DFVertexDataHandle *vertex) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFVERTEXDATAHANDLE_GET(vertex, _vertex);
    strt->remove(_vertex);
    return S_OK;
}

HRESULT MxC3DFStructure_removeEdge(struct Mx3DFStructureHandle *handle, struct Mx3DFEdgeDataHandle *edge) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFEDGEDATAHANDLE_GET(edge, _edge);
    strt->remove(_edge);
    return S_OK;
}

HRESULT MxC3DFStructure_removeFace(struct Mx3DFStructureHandle *handle, struct Mx3DFFaceDataHandle *face) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFFACEDATAHANDLE_GET(face, _face);
    strt->remove(_face);
    return S_OK;
}

HRESULT MxC3DFStructure_removeMesh(struct Mx3DFStructureHandle *handle, struct Mx3DFMeshDataHandle *mesh) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    MX3DFMESHDATAHANDLE_GET(mesh, _mesh);
    strt->remove(_mesh);
    return S_OK;
}

HRESULT MxC3DFStructure_getCentroid(struct Mx3DFStructureHandle *handle, float **centroid) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!centroid) 
        return E_FAIL;
    auto _centroid = strt->getCentroid();
    MXVECTOR3_COPYFROM(_centroid, (*centroid));
    return S_OK;
}

HRESULT MxC3DFStructure_translate(struct Mx3DFStructureHandle *handle, float *displacement) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!displacement) 
        return E_FAIL;
    return strt->translate(MxVector3f::from(displacement));;
}

HRESULT MxC3DFStructure_translateTo(struct Mx3DFStructureHandle *handle, float *position) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!position) 
        return E_FAIL;
    return strt->translateTo(MxVector3f::from(position));
}

HRESULT MxC3DFStructure_rotateAt(struct Mx3DFStructureHandle *handle, float *rotMat, float *rotPot) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!rotMat || !rotPot) 
        return E_FAIL;
    MxMatrix3f _rotMat;
    MXMATRIX3_COPYTO(rotMat, _rotMat);
    return strt->rotateAt(_rotMat, MxVector3f::from(rotPot));
}

HRESULT MxC3DFStructure_rotate(struct Mx3DFStructureHandle *handle, float *rotMat) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!rotMat) 
        return E_FAIL;
    MxMatrix3f _rotMat;
    MXMATRIX3_COPYTO(rotMat, _rotMat);
    return strt->rotate(_rotMat);
}

HRESULT MxC3DFStructure_scaleFrom(struct Mx3DFStructureHandle *handle, float *scales, float *scalePt) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!scales || !scalePt) 
        return E_FAIL;
    return strt->scaleFrom(MxVector3f::from(scales), MxVector3f::from(scalePt));
}

HRESULT MxC3DFStructure_scaleFromS(struct Mx3DFStructureHandle *handle, float scale, float *scalePt) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!scalePt) 
        return E_FAIL;
    return strt->scaleFrom(scale, MxVector3f::from(scalePt));
}

HRESULT MxC3DFStructure_scale(struct Mx3DFStructureHandle *handle, float *scales) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!scales) 
        return E_FAIL;
    return strt->scale(MxVector3f::from(scales));
}

HRESULT MxC3DFStructure_scaleS(struct Mx3DFStructureHandle *handle, float scale) {
    MX3DFSTRUCTUREHANDLE_GET(handle, strt);
    return strt->scale(scale);
}


/////////////////
// MxFIOModule //
/////////////////


HRESULT MxCFIOModule_init(struct MxFIOModuleHandle *handle, const char *moduleName, MxFIOModuleToFileFcn toFile, MxFIOModuleFromFileFcn fromFile) {
    MxCFIOModule *cmodule = new MxCFIOModule();
    cmodule->_moduleName = moduleName;
    cmodule->_toFile = toFile;
    cmodule->_fromFile = fromFile;
    handle->MxObj = (void*)cmodule;
    return S_OK;
}

HRESULT MxCFIOModule_destroy(struct MxFIOModuleHandle *handle) {
    return mx::capi::destroyHandle<MxCFIOModule, MxFIOModuleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCFIOModule_registerIOModule(struct MxFIOModuleHandle *handle) {
    MXCPTRCHECK(handle); MXCPTRCHECK(handle->MxObj);
    MxCFIOModule *cmodule = (MxCFIOModule*)handle->MxObj;
    cmodule->registerIOModule();
    return S_OK;
}

HRESULT MxCFIOModule_load(struct MxFIOModuleHandle *handle) {
    MXCPTRCHECK(handle); MXCPTRCHECK(handle->MxObj);
    MxCFIOModule *cmodule = (MxCFIOModule*)handle->MxObj;
    cmodule->load();
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT MxCFIO_getMxIORootElement(struct MxIOElementHandle *handle) {
    MxIOElement *rootElement = MxFIO::currentRootElement == NULL ? MxFIO::generateMxIORootElement() : MxFIO::currentRootElement;
    if(!rootElement) 
        return E_FAIL;
    handle->MxObj = (void*)rootElement;
    return S_OK;
}

HRESULT MxCFIO_releaseMxIORootElement() {
    return MxFIO::releaseMxIORootElement();
}

HRESULT MxCFIO_hasImport(bool *value) {
    MXCPTRCHECK(value);
    *value = MxFIO::currentRootElement != NULL;
    return S_OK;
}

HRESULT MxCIO_mapImportParticleId(unsigned int pid, unsigned int *mapId) {
    MXCPTRCHECK(MxFIO::importSummary);
    MXCPTRCHECK(mapId);
    
    auto itr = MxFIO::importSummary->particleIdMap.find(pid);
    if(itr == MxFIO::importSummary->particleIdMap.end()) 
        return E_FAIL;
    *mapId = itr->second;
    return S_OK;
}

HRESULT MxCIO_mapImportParticleTypeId(unsigned int ptid, unsigned int *mapId) {
    MXCPTRCHECK(MxFIO::importSummary);
    MXCPTRCHECK(mapId);
    
    auto itr = MxFIO::importSummary->particleTypeIdMap.find(ptid);
    if(itr == MxFIO::importSummary->particleTypeIdMap.end()) 
        return E_FAIL;
    *mapId = itr->second;
    return S_OK;
}

HRESULT MxCIO_fromFile3DF(const char *filePath, struct Mx3DFStructureHandle *strt) {
    MXCPTRCHECK(filePath);
    Mx3DFStructure *_strt = MxIO::fromFile3DF(filePath);
    MXCPTRCHECK(_strt);
    MXCPTRCHECK(strt);
    strt->MxObj = (void*)_strt;
    return S_OK;
}

HRESULT MxCIO_toFile3DF(const char *format, const char *filePath, unsigned int pRefinements) {
    MXCPTRCHECK(format);
    MXCPTRCHECK(filePath);
    return MxIO::toFile3DF(format, filePath, pRefinements);
}

HRESULT MxCIO_toFile(const char *saveFilePath) {
    MXCPTRCHECK(saveFilePath);
    return MxIO::toFile(saveFilePath);
}

HRESULT MxCIO_toString(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(MxIO::toString(), str, numChars);
}
