/**
 * @file MxMeshStructure.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh structure class
 * @date 2022-04-26
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_MXMESHSTRUCTURE_H_
#define MODELS_VERTEX_SOLVER_MXMESHSTRUCTURE_H_

#include <mx_port.h>

#include "mx_mesh.h"

class MxMeshVertex;
class MxMeshSurface;
class MxMeshBody;

struct MxMeshStructureType;

class CAPI_EXPORT MxMeshStructure : public MxMeshObj {

    std::vector<MxMeshStructure*> structures_parent;
    std::vector<MxMeshStructure*> structures_child;
    std::vector<MxMeshBody*> bodies;

public:

    unsigned int typeId;

    MxMeshStructure() : MxMeshObj() {};

    MxMeshObj::Type objType() { return MxMeshObj::Type::STRUCTURE; }

    std::vector<MxMeshObj*> parents();

    std::vector<MxMeshObj*> children() { return mx::models::vertex::vectorToBase(structures_child); }

    HRESULT addChild(MxMeshObj *obj);

    HRESULT addParent(MxMeshObj *obj);

    HRESULT removeChild(MxMeshObj *obj);

    HRESULT removeParent(MxMeshObj *obj);

    bool validate() { return true; }

    MxMeshStructureType *type();

    std::vector<MxMeshStructure*> getStructures() { return structures_parent; }

    std::vector<MxMeshBody*> getBodies();

    std::vector<MxMeshSurface*> getSurfaces();

    std::vector<MxMeshVertex*> getVertices();

};


struct CAPI_EXPORT MxMeshStructureType : MxMeshObjType {

    MxMeshObj::Type objType() { return MxMeshObj::Type::STRUCTURE; }
    
};

#endif // MODELS_VERTEX_SOLVER_MXMESHSTRUCTURE_H_