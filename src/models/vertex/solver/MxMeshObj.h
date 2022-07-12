/**
 * @file MxMeshObj.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the base object of a Mechanica mesh
 * @date 2022-04-26
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_MXMESHOBJ_H_
#define MODELS_VERTEX_SOLVER_MXMESHOBJ_H_

#include <platform.h>

#include <vector>

class MxMesh;

struct MxMeshObj { 

    enum Type : unsigned int {
        NONE        = 0, 
        VERTEX      = 1, 
        SURFACE     = 2, 
        BODY        = 3, 
        STRUCTURE   = 4
    };

    /** The mesh of this object, if any */
    MxMesh *mesh;

    /** Object id; unique by type in a mesh */
    int objId;

    /** Object actors */
    std::vector<struct MxMeshObjActor*> actors;

    MxMeshObj();

    virtual MxMeshObj::Type objType() = 0;

    /** Current parent objects. */
    virtual std::vector<MxMeshObj*> parents() = 0;

    /** Current child objects. Child objects require this object as part of their definition. */
    virtual std::vector<MxMeshObj*> children() = 0;

    virtual HRESULT addChild(MxMeshObj *obj) = 0;

    virtual HRESULT addParent(MxMeshObj *obj) = 0;

    virtual HRESULT removeChild(MxMeshObj *obj) = 0;

    virtual HRESULT removeParent(MxMeshObj *obj) = 0;

    /** Validate state of object for deployment in a mesh */
    virtual bool validate() = 0;

    /** Test whether this object is in another object */
    bool in(MxMeshObj *obj);

    /** Test whether this object has another object */
    bool has(MxMeshObj *obj);

};


struct MxMeshObjActor { 

    virtual HRESULT energy(MxMeshObj *source, MxMeshObj *target, float &e) = 0;

    virtual HRESULT force(MxMeshObj *source, MxMeshObj *target, float *f) = 0;

};


struct MxMeshObjType {

    int id = -1;

    /** Object type actors */
    std::vector<MxMeshObjActor*> actors;

    virtual MxMeshObj::Type objType() = 0;

};

#endif // MODELS_VERTEX_SOLVER_MXMESHOBJ_H_