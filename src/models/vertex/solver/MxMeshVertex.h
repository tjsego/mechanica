/**
 * @file MxMeshVertex.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh vertex class
 * @date 2022-04-26
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_MXMESHVERTEX_H_
#define MODELS_VERTEX_SOLVER_MXMESHVERTEX_H_

#include <mx_port.h>

#include <MxParticle.h>
#include <rendering/MxStyle.hpp>

#include "MxMeshObj.h"

#include <io/Mx3DFVertexData.h>

#include <vector>


class MxMeshSurface;
class MxMeshBody;
class MxMeshStructure;
class MxMesh;

struct CAPI_EXPORT MxMeshParticleType : MxParticleType { 

    MxMeshParticleType() : MxParticleType(true) {
        std::memcpy(this->name, "MxMeshParticleType", sizeof("MxMeshParticleType"));
        style->setVisible(false);
        registerType();
    };

};

CAPI_FUNC(MxMeshParticleType*) MxMeshParticleType_get();


/**
 * @brief The mesh vertex is a volume of a mesh centered at a point in a space.
 * 
 */
class CAPI_EXPORT MxMeshVertex : public MxMeshObj {

    /** Particle id. -1 if not assigned */
    int pid;

    /** Connected surfaces */
    std::vector<MxMeshSurface*> surfaces;

public:

    MxMeshVertex();
    MxMeshVertex(const unsigned int &_pid);
    MxMeshVertex(const MxVector3f &position);
    MxMeshVertex(Mx3DFVertexData *vdata);

    MxMeshObj::Type objType() { return MxMeshObj::Type::VERTEX; }

    std::vector<MxMeshObj*> parents() { return std::vector<MxMeshObj*>(); }

    std::vector<MxMeshObj*> children();

    HRESULT addChild(MxMeshObj *obj);

    HRESULT addParent(MxMeshObj *obj) { return E_FAIL; }

    HRESULT removeChild(MxMeshObj *obj);

    HRESULT removeParent(MxMeshObj *obj) { return E_FAIL; }

    bool validate() { return true; }

    std::vector<MxMeshStructure*> getStructures();

    std::vector<MxMeshBody*> getBodies();

    std::vector<MxMeshSurface*> getSurfaces() { return surfaces; }

    MxMeshSurface *findSurface(const MxVector3f &dir);
    MxMeshBody *findBody(const MxVector3f &dir);

    std::vector<MxMeshVertex*> neighborVertices();

    std::vector<MxMeshSurface*> sharedSurfaces(MxMeshVertex *other);

    float getVolume();
    float getMass();

    HRESULT positionChanged();
    HRESULT updateProperties();

    MxParticleHandle *particle();

    MxVector3f getPosition();

    HRESULT setPosition(const MxVector3f &pos);


    friend MxMeshSurface;
    friend MxMeshBody;
    friend MxMesh;

};

#endif // MODELS_VERTEX_SOLVER_MXMESHVERTEX_H_