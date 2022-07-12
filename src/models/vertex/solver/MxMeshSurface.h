/**
 * @file MxMeshSurface.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh surface class
 * @date 2022-04-26
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_MXMESHSURFACE_H_
#define MODELS_VERTEX_SOLVER_MXMESHSURFACE_H_

#include <mx_port.h>

#include <state/MxStateVector.h>
#include <rendering/MxStyle.hpp>

#include "mx_mesh.h"

#include <io/Mx3DFFaceData.h>

class MxMeshVertex;
class MxMeshBody;
class MxMeshStructure;
class MxMesh;

struct MxMeshSurfaceType;

/**
 * @brief The mesh surface is an area-enclosed object of implicit mesh edges defined by mesh vertices. 
 * 
 * The mesh surface consists of at least three mesh vertices. 
 * 
 * The mesh surface is always flat. 
 * 
 * The mesh surface can have a state vector, which represents a uniform amount of substance 
 * attached to the surface. 
 * 
 */
class CAPI_EXPORT MxMeshSurface : public MxMeshObj {

    /** Connected body, if any, where the surface normal is outward-facing */
    MxMeshBody *b1;
    
    /** Connected body, if any, where the surface normal is inward-facing */
    MxMeshBody *b2;

    std::vector<MxMeshVertex*> vertices;

    MxVector3f normal;

    MxVector3f centroid;

    float area;

    /** Volume contributed by this surface to its child bodies */
    float _volumeContr;

public:

    unsigned int typeId;

    MxStateVector *species;

    MxStyle *style;

    MxMeshSurface();
    MxMeshSurface(std::vector<MxMeshVertex*> _vertices);
    MxMeshSurface(Mx3DFFaceData *face);

    MxMeshObj::Type objType() { return MxMeshObj::Type::SURFACE; }

    std::vector<MxMeshObj*> parents() { return mx::models::vertex::vectorToBase(vertices); }

    std::vector<MxMeshObj*> children();

    HRESULT addChild(MxMeshObj *obj);

    HRESULT addParent(MxMeshObj *obj);

    HRESULT removeChild(MxMeshObj *obj);

    HRESULT removeParent(MxMeshObj *obj);

    bool validate();

    HRESULT refreshBodies();

    MxMeshSurfaceType *type();

    std::vector<MxMeshStructure*> getStructures();

    std::vector<MxMeshBody*> getBodies();

    std::vector<MxMeshVertex*> getVertices() { return vertices; }

    MxMeshVertex *findVertex(const MxVector3f &dir);
    MxMeshBody *findBody(const MxVector3f &dir);

    std::vector<MxMeshVertex*> neighborVertices(MxMeshVertex *v);

    std::vector<MxMeshSurface*> neighborSurfaces();

    std::vector<unsigned int> contiguousEdgeLabels(MxMeshSurface *other);

    unsigned int numSharedContiguousEdges(MxMeshSurface *other);

    MxVector3f getNormal() { return normal; }

    MxVector3f getCentroid() { return centroid; }

    float getArea() { return area; }

    float volumeSense(MxMeshBody *body);
    float getVolumeContr(MxMeshBody *body) { return _volumeContr * volumeSense(body); }

    float getVertexArea(MxMeshVertex *v);

    MxVector3f triangleNormal(const unsigned int &idx);

    HRESULT positionChanged();


    friend MxMeshBody;
    friend MxMesh;

};


struct CAPI_EXPORT MxMeshSurfaceType : MxMeshObjType {

    MxStyle *style;

    MxMeshSurfaceType() : MxMeshObjType() {
        style = NULL;
    }

    MxMeshObj::Type objType() { return MxMeshObj::Type::SURFACE; }

    MxMeshSurface *operator() (std::vector<MxMeshVertex*> _vertices);

    MxMeshSurface *operator() (const std::vector<MxVector3f> &_positions);

    MxMeshSurface *operator() (Mx3DFFaceData *face);

};

#endif // MODELS_VERTEX_SOLVER_MXMESHSURFACE_H_