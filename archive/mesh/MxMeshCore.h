/*
 * MxMeshCore.h
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#ifndef SRC_MXMESHCORE_H_
#define SRC_MXMESHCORE_H_

#include "mechanica_private.h"
#include <vector>
#include <array>
#include <algorithm>
#include <set>
#include <Magnum/Magnum.h>
#include <types/mx_types.h>
#include <MxParticle.h>
#include <iostream>


template<typename Container, typename Value>
bool contains(const Container& container, const Value& value) {
    return std::find(container.begin(), container.end(), value) != container.end();
}

template<typename Container, typename Value>
void remove(Container& cont, const Value& val) {
    auto it = std::find(cont.begin(), cont.end(), val);
    if(it != cont.end()) {
        cont.erase(it);
    }
}


using Color4 = Magnum::Color4;


using namespace Magnum;

struct MxVertex;
typedef MxVertex* VertexPtr;
typedef const MxVertex *CVertexPtr;

typedef std::array<VertexPtr, 3> VertexIndices;

struct MxPolygon;
typedef MxPolygon *PolygonPtr;
typedef const MxPolygon *CPolygonPtr;


struct MxPartialPolygon;
typedef MxPartialPolygon *PPolygonPtr;
typedef const MxPartialPolygon *CPPolygonPtr;

typedef std::vector<PPolygonPtr> PartialPolygons;

struct MxCell;
typedef MxCell *CellPtr;
typedef const MxCell *CCellPtr;

struct MxMesh;
typedef MxMesh *MeshPtr;
typedef const MxMesh *CMeshPtr;

struct MxEdge;
typedef MxEdge *EdgePtr;
typedef const MxEdge *CEdgePtr;

struct MxVertex;
typedef MxVertex *VertexPtr;
typedef const MxVertex *CVertexPtr;

// triangle container in the main mesh
typedef std::vector<PolygonPtr> TriangleContainer;

// triangle container for the vertices
typedef std::vector<PolygonPtr> Triangles;


struct MxMeshNode {
    MxMeshNode(MeshPtr m, uint _id) : id{_id}, mesh{m} {};

    const uint id;

    MeshPtr mesh;

    friend struct MxVertex;
};

struct MxVertex {
    /**
     * The Mechanica vertex does not represent a point mass as in a traditional
     * particle based approach. Rather, the vertex here represents a region of space,
     * hence, we need to calculate the mass as a weighted sum of all the neighboring
     * triangles.
     *
     * TODO: We presently compute the mass in the MxTriangle::positionsChanged method.
     * This approach is not very cache friendly, will come up with a more optimal
     * solution in a later release.
     */
    float mass = 0;

    float area = 0;

    MxVertex() {}

    MxVertex(float mass, float area, const MxVector3f &pos);

    MxVector3f position;

    MxVector3f force;


    float attr = 0;

    int edgeCount() const;


    uint id{0};

    void positionsChanged() {
        mass = 0;
        area = 0;
    }

private:


    /**
     * Get the mesh pointer.
     */
    MeshPtr mesh();


    //EdgePtr edges[4] = {nullptr};

    uint _edgeCount = 0;

    friend struct MxCell;

    friend HRESULT connectEdgeVertices(EdgePtr edge, VertexPtr,
            VertexPtr);
};

std::ostream& operator<<(std::ostream& os, CVertexPtr v);



struct MxVertexAttribute {

    enum Id : ushort {Position, Normal};

    // the offset in the given vertex buffer memory block of where to write
    // the attribute
    ushort offset;

    // the id of this attribute. The id is simply an identifier that means something
    // to the MxMesh. This id does not have anything to do with the shader location id.
    // we keep separate ids because the MxMesh will often have many more attributes than
    // a renderer will want to display at any one time. Renderers typically display only
    // a subset of the available attributes. Attributes are things like scalar fields
    // attached to vertices, vertex position, normal, velocity, acceleration, etc...
    Id id;
};



#ifndef NDEBUG
#define checkVec(vec) \
    if(std::isnan(vec[0]) || std::isnan(vec[1]) || std::isnan(vec[1])) { \
        std::cout << "Vector with NaN values" << std::endl; \
        assert(0); \
    }
#else
#define checkVec(vec)
#endif




#endif /* SRC_MXMESHCORE_H_ */
