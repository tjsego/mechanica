/*
 * MxTriangle.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include "MxTriangle.h"
#include "MxCell.h"


int MxTriangle::matchVertexIndices(const std::array<VertexPtr, 3> &indices) {
    typedef std::array<VertexPtr, 3> vertind;

    if (vertices == indices ||
        vertices == vertind{{indices[1], indices[2], indices[0]}} ||
        vertices == vertind{{indices[2], indices[0], indices[1]}}) {
        return 1;
    }

    if (vertices == vertind{{indices[2], indices[1], indices[0]}} ||
        vertices == vertind{{indices[1], indices[0], indices[2]}} ||
        vertices == vertind{{indices[0], indices[2], indices[1]}}) {
        return -1;
    }
    return 0;
}


MxTriangle::MxTriangle(MxTriangleType* type,
		const std::array<VertexPtr, 3>& verts,
		const std::array<CellPtr, 2>& cells,
		const std::array<MxPartialTriangleType*, 2>& partTriTypes,
		FacetPtr facet) :
			MxObject{type}, vertices{verts}, cells{cells},
			partialTriangles{{{partTriTypes[0], this}, {partTriTypes[1], this}}},
			facet{facet} {
    for(VertexPtr vert : verts) {
        assert(!contains(vert->triangles, this));
        vert->triangles.push_back(this);
    }
    positionsChanged();
}

HRESULT MxTriangle::positionsChanged() {

	const Vector3& v1 = vertices[0]->position;
	const Vector3& v2 = vertices[1]->position;
	const Vector3& v3 = vertices[2]->position;

	// the aspect ratio
    float a = (v1 - v2).length();
    float b = (v2 - v3).length();
    float c = (v3 - v1).length();
    float s = (a + b + c) / 2.0;
    aspectRatio = (a * b * c) / (8.0 * (s - a) * (s - b) * (s - c));

    // A surface normal for a triangle can be calculated by taking the vector cross product
    // of two edges of that triangle. The order of the vertices used in the calculation will
    // affect the direction of the normal (in or out of the face w.r.t. winding).

    // So for a triangle p1, p2, p3, if the vector U = p2 - p1 and the
    // vector V = p3 - p1 then the normal N = U x V and can be calculated by:

    // Nx = UyVz - UzVy
    // Ny = UzVx - UxVz
    // Nz = UxVy - UyVx
    // non-normalized normal vector
    Vector3 abnormal = Math::cross(v2 - v1, v3 - v1);
    area = abnormal.length();
    normal = abnormal / area;

    // average position of 3 position vectors
    centroid = (v1 + v2 + v3) / 3;

    return S_OK;
}