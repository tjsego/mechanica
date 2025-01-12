/*
 * MxMesh.cpp
 *
 *  Created on: Jan 31, 2017
 *      Author: andy
 */


#include "MxDebug.h"
#include "MxMesh.h"
#include "MeshRelationships.h"
#include <Magnum/Math/Math.h>
#include <Corrade/Containers/Optional.h>
#include <mx_error.h>

#include <deque>
#include <limits>
#include <cmath>


int MxMesh::findVertex(const MxVector3f& pos, double tolerance) {
    for (int i = 1; i < vertices.size(); ++i) {
        float dist = (vertices[i]->position - pos).dot();
        if (dist <= tolerance) {
            return i;
        }
    }
    return -1;
}

VertexPtr MxMesh::createVertex(const MxVector3f& pos) {

    VertexPtr retval = nullptr;
    retval = new MxVertex{0., 0., pos};


    retval->id = ++vertexId;
    vertices.push_back(retval);
    return retval;
}

// todo: specialize alloc for MxVertex
template<>
MxVertex *MxMesh::alloc() {
    auto v = new MxVertex();
    vertices.push_back(v);
    return v;
}

// todo: specialize alloc for MxEdge
template<>
MxEdge *MxMesh::alloc() {
    MxEdge *e = new MxEdge(edges.size());
    edges.push_back(e);
    return e;
}

CellPtr MxMesh::createCell(MxCellType *type, const std::string& name) {
    CellPtr cell = new MxCell{(uint)cells.size(), type, this, nullptr, name};
    cells.push_back(cell);
    return cell;
}

void MxMesh::vertexAtributes(const std::vector<MxVertexAttribute>& attributes,
        uint vertexCount, uint stride, void* buffer) {
}


void MxMesh::dump(uint what) {
    for(int i = 0; i < vertices.size(); ++i) {
        std::cout << "[" << i << "]" << vertices[i]->position << std::endl;
    }
}

std::tuple<MxVector3f, MxVector3f> MxMesh::extents() const {

    auto min = MxVector3f{std::numeric_limits<float>::max()};
    auto max = MxVector3f{std::numeric_limits<float>::min()};


    for(auto& v : vertices) {
        for(int i = 0; i < 3; ++i) {min[i] = (v->position[i] < min[i] ? v->position[i] : min[i]);}
        for(int i = 0; i < 3; ++i) {max[i] = (v->position[i] > max[i] ? v->position[i] : max[i]);}
    }

    return std::make_tuple(min, max);
}

struct UniverseCellType : MxCellType {
    virtual Magnum::Color4 color(struct MxCell *cell) {
        return Color4{0., 0., 0., 0.};
    }

    UniverseCellType() : MxCellType{"UniverseCellType"} {};
};

UniverseCellType universeCellType;

MxMesh::MxMesh() /*: meshOperations(this, 0, 1.5)*/
{
    _rootCell = new MxCell{(uint)cells.size(), &universeCellType, this, nullptr, "RootCell"};
    cells.push_back(_rootCell);
}



HRESULT MxMesh::deleteVertex(VertexPtr v) {

    //meshOperations.removeDependentOperations(v);

    remove(vertices, v);
#ifndef NDEBUG
    for(PolygonPtr tri : polygons) {
        assert(!incidentPolygonVertex(tri, v));
    }
#endif
    delete v;
    return S_OK;
}

HRESULT MxMesh::deletePolygon(PolygonPtr tri) {

    //meshOperations.removeDependentOperations(tri);

    for(EdgePtr e : tri->edges) {
        remove(edges, e);
    }

    remove(polygons, tri);
    delete tri;

    assert(!contains(polygons, tri));

#ifndef NDEBUG
    for(CellPtr cell : cells) {
        assert(!connectedCellPolygonPointers(cell, tri));
    }

#endif
    return S_OK;
}


int test(const std::vector<std::string*> &stuff) {

    for(int i = 0; i < stuff.size(); ++i) {
        std::string *s = stuff[i];

        s->append("foo");
    }

    //stuff.push_back("");
    return 5;
}



bool MxMesh::valid(PolygonPtr p) {

    if(std::find(polygons.begin(), polygons.end(), p) == polygons.end()) {
        return false;
    }


    return
        p == p->partialPolygons[0].polygon &&
        p == p->partialPolygons[1].polygon &&
        valid(p->vertices[0]) &&
        valid(p->vertices[1]) &&
        valid(p->vertices[2]);
}

bool MxMesh::valid(CellPtr c) {
    if(std::find(cells.begin(), cells.end(), c) == cells.end()) {
        return false;
    }

    for(PPolygonPtr p : c->surface) {
        if(!valid(p)) {
            return false;
        }
    }

    return true;
}

bool MxMesh::valid(VertexPtr v) {
    return std::find(vertices.begin(), vertices.end(), v) != vertices.end();
}

bool MxMesh::valid(PPolygonPtr p) {
    return p && valid(p->polygon);
}

MxMesh::~MxMesh() {
    for(auto c : cells) {
        delete c;
    }
    for(auto p : vertices) {
        delete p;
    }
    for(auto t : polygons) {
        delete t;
    }
}

HRESULT MxMesh::applyMeshOperations() {
    HRESULT result = E_FAIL;

    /*
    if((result = meshOperations.positionsChanged(polygons.begin(), polygons.end())) != S_OK) {
        return result;
    }

    for(VertexPtr vert : vertices) {
        meshOperations.valenceChanged(vert);
    }

    if((result = meshOperations.apply()) != S_OK) {
        return result;
    }
    */

    return positionsChanged();
}

PolygonPtr MxMesh::createPolygon(MxPolygonType* type,  const std::vector<VertexPtr> &vertices) {

    PolygonPtr poly = new MxPolygon{(uint)polygons.size(), type};

    polygons.push_back(poly);

    VERIFY(connectPolygonVertices(this, poly, vertices));

    return poly;
}


HRESULT MxMesh::positionsChanged()
{
    HRESULT result;

    for(int i = 0; i < vertices.size(); ++i) {
        VertexPtr v = vertices[i];
        v->mass = 0;
        v->area = 0;
        v->force = {0.f, 0.f, 0.f};
    }

    for(PolygonPtr tri : polygons) {

        if((result = tri->positionsChanged() != S_OK)) {
            return result;
        }
    }

    for(CellPtr cell : cells) {
        if((result = cell->positionsChanged() != S_OK)) {
            return result;
        }
    }

    return S_OK;
}

EdgePtr MxMesh::findEdge(CVertexPtr a, CVertexPtr b) const
{
    for(EdgePtr edge : edges) {
        if(edge->matches(a, b)) {
            return edge;
        }
    }
    return nullptr;
}


EdgePtr MxMesh::createEdge(VertexPtr a, VertexPtr b)
{
    EdgePtr e = alloc<MxEdge>();
    VERIFY(connectEdgeVertices(e, a, b));
    return e;
}


HRESULT MxMesh::setPositions(uint32_t len, const MxVector3f* positions)
{
    HRESULT result;



    if(positions) {
        for(int i = 0; i < vertices.size(); ++i) {
            VertexPtr v = vertices[i];
            v->mass = 0;
            v->area = 0;
            v->position = positions[i];
        }
    }
    else {
        for(int i = 0; i < vertices.size(); ++i) {
            VertexPtr v = vertices[i];
            v->positionsChanged();
        }
    }

    for(PolygonPtr poly : polygons) {
        if((result = poly->positionsChanged() != S_OK)) {
            return result;
        }
    }

    for(CellPtr cell : cells) {
        if((result = cell->positionsChanged() != S_OK)) {
            return result;
        }
    }

    return S_OK;
}

HRESULT MxMesh::selectObject(MxMesh_TYPESEL type, uint index) {
    switch (type)
    {
    case MxMesh_TYPESEL::MxMesh_TYPEEDGE:
        return selectEdge(index);
    case MxMesh_TYPESEL::MxMesh_TYPEPOLYGON:
        return selectPolygon(index);
    case MxMesh_TYPESEL::MxMesh_TYPENONE:
        return deselectObject();
    default:
        break;
    }
    return E_FAIL;
}

HRESULT MxMesh::selectEdge(uint index) {

    if(index >= edges.size()) return mx_error(E_FAIL, "index out of range");

    *_selectedObject = edges[index];
    return S_OK;

}

HRESULT MxMesh::selectPolygon(uint index) {

    if(index >= polygons.size()) return mx_error(E_FAIL, "index out of range");

    *_selectedObject = polygons[index];
    return S_OK;
}

HRESULT MxMesh::deselectObject() {
    *_selectedObject = MxMesh::NoneSelectedType();
    return S_OK;
}

PolygonPtr MxMesh::createPolygon(MxPolygonType* type)
{
    PolygonPtr poly = new MxPolygon{(uint)polygons.size(), type};

    polygons.push_back(poly);

    return poly;
}
