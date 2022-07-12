/**
 * @file MxMeshBind.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh binding
 * @date 2022-06-15
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_MXMESHBIND_H_
#define MODELS_VERTEX_SOLVER_MXMESHBIND_H_

#include "MxMeshObj.h"
#include "MxMeshVertex.h"
#include "MxMeshSurface.h"
#include "MxMeshBody.h"
#include "MxMeshStructure.h"


struct CAPI_EXPORT MxMeshBind {

    static HRESULT structure(MxMeshStructureType *structureType, MxMeshObjActor *actor);

    static HRESULT structure(MxMeshStructure *structure, MxMeshObjActor *actor);

    static HRESULT body(MxMeshBodyType *bodyType, MxMeshObjActor *actor);

    static HRESULT body(MxMeshBody *body, MxMeshObjActor *actor);

    static HRESULT surface(MxMeshSurfaceType *surfaceType, MxMeshObjActor *actor);

    static HRESULT surface(MxMeshSurface *surface, MxMeshObjActor *actor);

};

#endif // MODELS_VERTEX_SOLVER_MXMESHBIND_H_