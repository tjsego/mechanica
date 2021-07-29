/*
 * MxMeshCore.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include <MxPolygon.h>
#include "MxMeshCore.h"
#include "MxMesh.h"
#include "MxDebug.h"


MxVertex::MxVertex(float mass, float area, const MxVector3f &pos) :
        /*CObject{0, MxVertex_Type}*/
        mass{mass}, area{area}, position{pos} {
};



std::ostream& operator <<(std::ostream& os, CVertexPtr v)
{
    os << "{id:" << v->id << ", pos:" << v->position << "}";
    return os;
}



int MxVertex::edgeCount() const
{
    return _edgeCount;
}


