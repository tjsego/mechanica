/*
 * FlipEdge.cpp
 *
 *  Created on: Jul 24, 2018
 *      Author: andy
 */

#include "MeshOperations.h"
#include "MxPolygon.h"
#include "MeshRelationships.h"
#include "MxMesh.h"
#include <mx_error.h>
#include <MxLogger.h>

HRESULT Mx_FlipEdge(MeshPtr mesh, EdgePtr edge) {

    Log(LOG_INFORMATION) << "applyT1Edge2Transition(edge=" << edge << ")";

    if(edge->polygonCount() != 2) {
        return mx_error(E_FAIL, "edge polygon count must be 2");
    }

    if(edge->polygons[0]->size() <= 3 || edge->polygons[1]->size() <= 3) {
        return mx_error(E_FAIL, "can't collapse edge that's connected to polygons with less than 3 sides");

    }

    PolygonPtr p1 = nullptr, p2 = nullptr, p3 = nullptr, p4 = nullptr;

    // top and bottom vertices
    VertexPtr v1 = edge->vertices[0];
    VertexPtr v2 = edge->vertices[1];

    // identify the two side polygons as p4 on the left, and p2 on the right, check
    // vertex winding. Choose p2 to have v2 after v1, and p4 to have v1 after v2
    {
        // TODO use next vert instead of looking at all vertices.
        int v1Index_0 = edge->polygons[0]->vertexIndex(v1);
        int v2Index_0 = edge->polygons[0]->vertexIndex(v2);

#ifndef NDEBUG
        int v1Index_1 = edge->polygons[1]->vertexIndex(v1);
        int v2Index_1 = edge->polygons[1]->vertexIndex(v2);
#endif
        assert(v1Index_0 >= 0 && v2Index_0 >= 0 && v1Index_0 != v2Index_0);

        if(((v1Index_0 + 1) % edge->polygons[0]->size()) == v2Index_0) {
            // found p2 CCW
            p2 = edge->polygons[0];
            p4 = edge->polygons[1];
            assert((v2Index_1 + 1) % edge->polygons[1]->size() == v1Index_1);
        }
        else {
            // found p4
            p4 = edge->polygons[0];
            p2 = edge->polygons[1];
            assert((v1Index_1 + 1) % edge->polygons[1]->size() == v2Index_1);
        }
    }

    assert(p4 && p2);

    EdgePtr e1 = nullptr, e2 = nullptr, e3 = nullptr, e4 = nullptr;


    Log(LOG_INFORMATION) << "poly p2: " << p2;
    Log(LOG_INFORMATION) << "poly p4: " << p4;

    Log(LOG_INFORMATION) << "disconnectPolygonEdgeVertex(p2, edge, v1, &e1, &e2)";
    VERIFY(disconnectPolygonEdgeVertex(p2, edge, v1, &e1, &e2));


    Log(LOG_INFORMATION) << "poly p2: " << p2;
    Log(LOG_INFORMATION) << "poly p4: " << p4;

    Log(LOG_INFORMATION) << "disconnectPolygonEdgeVertex(p4, edge, v2, &e3, &e4)";
    VERIFY(disconnectPolygonEdgeVertex(p4, edge, v2, &e3, &e4));

    assert(edge->polygonCount() == 0);

    Log(LOG_INFORMATION) << "e1:" << e1;
    Log(LOG_INFORMATION) << "e2:" << e2;
    Log(LOG_INFORMATION) << "e3:" << e3;
    Log(LOG_INFORMATION) << "e4:" << e4;

    Log(LOG_INFORMATION) << "poly p2: " << p2;
    Log(LOG_INFORMATION) << "poly p4: " << p4;

    assert(connectedEdgeVertex(e1, v1));
    assert(connectedEdgeVertex(e2, v2));
    assert(connectedEdgeVertex(e3, v2));
    assert(connectedEdgeVertex(e4, v1));

    for(PolygonPtr p : e1->polygons) {
        if(contains(p->edges, e4)) {
            p1 = p;
            break;
        }
    }

    for(PolygonPtr p : e2->polygons) {
        if(contains(p->edges, e3)) {
            p3 = p;
            break;
        }
    }

    assert(p1 && p3);
    assert(p1 != p2 && p1 != p3 && p1 != p4);
    assert(p2 != p1 && p2 != p3 && p2 != p4);
    assert(p3 != p1 && p3 != p2 && p3 != p4);
    assert(p4 != p1 && p4 != p2 && p1 != p3);

    // original edge vector.
    MxVector3f edgeVec = v1->position - v2->position;
    float halfLen = edgeVec.length() / 2;

    // center position of the polygons that will get a new edge connecting them.
    MxVector3f centroid = (p2->centroid + p4->centroid) / 2;

    v2->position = centroid + (p2->centroid - centroid).normalized() * halfLen;
    v1->position = centroid + (p4->centroid - centroid).normalized() * halfLen;

    Log(LOG_INFORMATION) << "poly p1: " << p1;
    Log(LOG_INFORMATION) << "poly p2: " << p2;
    Log(LOG_INFORMATION) << "poly p3: " << p3;
    Log(LOG_INFORMATION) << "poly p4: " << p4;

    Log(LOG_INFORMATION) << "insertPolygonEdge(p1, edge)";
    VERIFY(insertPolygonEdge(p1, edge));

    Log(LOG_INFORMATION) << "poly p1: " << p1;
    Log(LOG_INFORMATION) << "poly p2: " << p2;
    Log(LOG_INFORMATION) << "poly p3: " << p3;
    Log(LOG_INFORMATION) << "poly p4: " << p4;

    Log(LOG_INFORMATION) << "insertPolygonEdge(p3, edge)";
    VERIFY(insertPolygonEdge(p3, edge));

    Log(LOG_INFORMATION) << "poly p1: " << p1;
    Log(LOG_INFORMATION) << "poly p2: " << p2;
    Log(LOG_INFORMATION) << "poly p3: " << p3;
    Log(LOG_INFORMATION) << "poly p4: " << p4;

    assert(connectedEdgeVertex(e1, v1));
    assert(connectedEdgeVertex(e2, v2));
    assert(connectedEdgeVertex(e3, v2));
    assert(connectedEdgeVertex(e4, v1));

    Log(LOG_INFORMATION) << "reconnecting edge vertices...";

    // reconnect the two diagonal edges, the other two edges, e2 and e4 stay
    // connected to their same vertices.
    VERIFY(reconnectEdgeVertex(e1, v2, v1));
    VERIFY(reconnectEdgeVertex(e3, v1, v2));

    Log(LOG_INFORMATION) << "poly p1: " << p1;
    Log(LOG_INFORMATION) << "poly p2: " << p2;
    Log(LOG_INFORMATION) << "poly p3: " << p3;
    Log(LOG_INFORMATION) << "poly p4: " << p4;

    assert(p1->size() >= 0);
    assert(p2->size() >= 0);
    assert(p3->size() >= 0);
    assert(p4->size() >= 0);

    assert(p1->checkEdges());
    assert(p2->checkEdges());
    assert(p3->checkEdges());
    assert(p4->checkEdges());

    for(CellPtr cell : mesh->cells) {
        cell->topologyChanged();
    }

    mesh->setPositions(0, 0);

    VERIFY(mesh->positionsChanged());

    return S_OK;
}


