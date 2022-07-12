/**
 * @file MxMeshBody.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh body class
 * @date 2022-04-26
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_MXMESHBODY_H_
#define MODELS_VERTEX_SOLVER_MXMESHBODY_H_

#include <mx_port.h>

#include <state/MxStateVector.h>

#include "mx_mesh.h"

class MxMeshVertex;
class MxMeshSurface;
class MxMeshStructure;
class MxMesh;

struct MxMeshBodyType;

/**
 * @brief The mesh body is a volume-enclosing object of mesh surfaces. 
 * 
 * The mesh body consists of at least four mesh surfaces. 
 * 
 * The mesh body can have a state vector, which represents a uniform amount of substance 
 * enclosed in the volume of the body. 
 * 
 */
class CAPI_EXPORT MxMeshBody : public MxMeshObj { 

    std::vector<MxMeshSurface*> surfaces;

    std::vector<MxMeshStructure*> structures;

    /** current centroid */
    MxVector3f centroid;

    /** current surface area */
    float area;

    /** current volume */
    float volume;

    /** mass density */
    float density;

    void _updateInternal();

public:

    unsigned int typeId;

    MxStateVector *species;

    MxMeshBody();
    MxMeshBody(std::vector<MxMeshSurface*> _surfaces);

    MxMeshObj::Type objType() { return MxMeshObj::Type::BODY; }

    std::vector<MxMeshObj*> parents();

    std::vector<MxMeshObj*> children();

    HRESULT addChild(MxMeshObj *obj);

    HRESULT addParent(MxMeshObj *obj);

    HRESULT removeChild(MxMeshObj *obj);

    HRESULT removeParent(MxMeshObj *obj);

    bool validate();

    HRESULT positionChanged();

    MxMeshBodyType *type();

    std::vector<MxMeshStructure*> getStructures();
    std::vector<MxMeshSurface*> getSurfaces() { return surfaces; }
    std::vector<MxMeshVertex*> getVertices();

    MxMeshVertex *findVertex(const MxVector3f &dir);
    MxMeshSurface *findSurface(const MxVector3f &dir);

    std::vector<MxMeshBody*> neighborBodies();
    std::vector<MxMeshSurface*> neighborSurfaces(MxMeshSurface *s);

    float getDensity() const { return density; }
    void setDensity(const float &_density) { density = _density; }

    MxVector3f getCentroid() const { return centroid; }
    MxVector3f getVelocity();
    float getArea() const { return area; }
    float getVolume() const { return volume; }
    float getMass() const { return volume * density; }

    float getVertexArea(MxMeshVertex *v);
    float getVertexVolume(MxMeshVertex *v);
    float getVertexMass(MxMeshVertex *v) { return getVertexVolume(v) * density; }

    float contactArea(MxMeshBody *other);

    
    friend MxMesh;
    friend MxMeshBodyType;

};


struct CAPI_EXPORT MxMeshBodyType : MxMeshObjType {

    float density;

    MxMeshObj::Type objType() { return MxMeshObj::Type::BODY; }

    MxMeshBody *operator() (std::vector<MxMeshSurface*> surfaces);
};

#endif // MODELS_VERTEX_SOLVER_MXMESHBODY_H_