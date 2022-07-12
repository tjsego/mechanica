/**
 * @file MxMesh.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh class
 * @date 2022-04-26
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_MXMESH_H_
#define MODELS_VERTEX_SOLVER_MXMESH_H_

#define MXMESHINV_INCR 100

#include "MxMeshVertex.h"
#include "MxMeshSurface.h"
#include "MxMeshBody.h"
#include "MxMeshStructure.h"

#include <set>
#include <vector>

struct MxMeshRenderer;
struct MxMeshSolver;

class CAPI_EXPORT MxMesh { 

    std::vector<MxMeshVertex*> vertices;
    std::vector<MxMeshSurface*> surfaces;
    std::vector<MxMeshBody*> bodies;
    std::vector<MxMeshStructure*> structures;

    std::set<unsigned int> vertexIdsAvail, surfaceIdsAvail, bodyIdsAvail, structureIdsAvail;
    bool isDirty;
    MxMeshSolver *_solver = NULL;

public:

    HRESULT add(MxMeshVertex *obj);
    HRESULT add(MxMeshSurface *obj);
    HRESULT add(MxMeshBody *obj);
    HRESULT add(MxMeshStructure *obj);
    HRESULT removeObj(MxMeshObj *obj);

    MxMeshVertex *findVertex(const MxVector3f &pos, const float &tol = 0.0001);

    MxMeshVertex *getVertex(const unsigned int &idx);
    MxMeshSurface *getSurface(const unsigned int &idx);
    MxMeshBody *getBody(const unsigned int &idx);
    MxMeshStructure *getStructure(const unsigned int &idx);

    unsigned int numVertices() const { return vertices.size() - vertexIdsAvail.size(); }
    unsigned int numSurfaces() const { return surfaces.size() - surfaceIdsAvail.size(); }
    unsigned int numBodies() const { return bodies.size() - bodyIdsAvail.size(); }
    unsigned int numStructures() const { return structures.size() - structureIdsAvail.size(); }

    unsigned int sizeVertices() const { return vertices.size(); }
    unsigned int sizeSurfaces() const { return surfaces.size(); }
    unsigned int sizeBodies() const { return bodies.size(); }
    unsigned int sizeStructures() const { return structures.size(); }

    /** Validate state of the mesh */
    bool validate();

    /** Manually notify that the mesh has been changed */
    HRESULT makeDirty();

    /** Check whether two vertices are connected */
    bool connected(MxMeshVertex *v1, MxMeshVertex *v2);

    /** Check whether two surfaces are connected */
    bool connected(MxMeshSurface *s1, MxMeshSurface *s2);

    /** Check whether two bodies are connected */
    bool connected(MxMeshBody *b1, MxMeshBody *b2);

    // Mesh editing

    /** Remove a vertex from the mesh; all connected surfaces and bodies are also removed */
    HRESULT remove(MxMeshVertex *v);

    /** Remove a surface from the mesh; all connected bodies are also removed */
    HRESULT remove(MxMeshSurface *s);

    /** Remove a body from the mesh */
    HRESULT remove(MxMeshBody *b);

    /** Inserts a vertex between two vertices */
    HRESULT insert(MxMeshVertex *toInsert, MxMeshVertex *v1, MxMeshVertex *v2);
    
    /** Replace a surface with a vertex */
    HRESULT replace(MxMeshVertex *toInsert, MxMeshSurface *toReplace);

    /** Replace a vertex with a surface. Vertices are created for the surface along every destroyed edge. */
    MxMeshSurface *replace(MxMeshSurfaceType *toInsert, MxMeshVertex *toReplace, std::vector<float> lenCfs);

    /** Merge two vertices. 
     * 
     * Vertices must be adjacent and on the same surface. 
    */
    HRESULT merge(MxMeshVertex *toKeep, MxMeshVertex *toRemove, const float &lenCf=0.5f);

    /** Merge two surfaces. 
     * 
     * Surfaces must have the same number of vertices. Vertices are paired by nearest distance.
    */
    HRESULT merge(MxMeshSurface *toKeep, MxMeshSurface *toRemove, const std::vector<float> &lenCfs);

    /** Create a surface from two vertices of a surface in the mesh and a position */
    MxMeshSurface *extend(MxMeshSurface *base, const unsigned int &vertIdxStart, const MxVector3f &pos);

    /** Create a body from a surface in the mesh and a position */
    MxMeshBody *extend(MxMeshSurface *base, MxMeshBodyType *btype, const MxVector3f &pos);

    /** Create a surface from two vertices of a surface in a mesh by extruding along the normal of the surface
     * 
     * todo: add support for extruding at an angle w.r.t. the center of the edge and centroid of the base surface
    */
    MxMeshSurface *extrude(MxMeshSurface *base, const unsigned int &vertIdxStart, const float &normLen);

    /** Create a body from a surface in a mesh by extruding along the outward-facing normal of the surface
     * 
     * todo: add support for extruding at an angle
    */
    MxMeshBody *extrude(MxMeshSurface *base, MxMeshBodyType *btype, const float &normLen);

    friend MxMeshRenderer;
    friend MxMeshSolver;

};

#endif // MODELS_VERTEX_SOLVER_MXMESH_H_