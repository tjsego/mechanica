/**
 * @file MxMeshBind.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh binding
 * @date 2022-06-15
 * 
 */

#include "MxMeshBind.h"


HRESULT MxMeshBind::structure(MxMeshStructureType *structureType, MxMeshObjActor *actor) {
    structureType->actors.push_back(actor);
    return S_OK;
}

HRESULT MxMeshBind::structure(MxMeshStructure *structure, MxMeshObjActor *actor) {
    structure->actors.push_back(actor);
    return S_OK;
}

HRESULT MxMeshBind::body(MxMeshBodyType *bodyType, MxMeshObjActor *actor) {
    bodyType->actors.push_back(actor);
    return S_OK;
}

HRESULT MxMeshBind::body(MxMeshBody *body, MxMeshObjActor *actor) {
    body->actors.push_back(actor);
    return S_OK;
}

HRESULT MxMeshBind::surface(MxMeshSurfaceType *surfaceType, MxMeshObjActor *actor) { 
    surfaceType->actors.push_back(actor);
    return S_OK;
}

HRESULT MxMeshBind::surface(MxMeshSurface *surface, MxMeshObjActor *actor) { 
    surface->actors.push_back(actor);
    return S_OK;
}
