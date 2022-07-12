/**
 * @file MxMeshObj.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the base object of a Mechanica mesh
 * @date 2022-26-04
 * 
 */

#include "MxMeshObj.h"


MxMeshObj::MxMeshObj() : 
    mesh{NULL}, 
    objId{-1}
{}

bool MxMeshObj::in(MxMeshObj *obj) {
    if(!obj || objType() > obj->objType()) 
        return false;

    for(auto &p : obj->parents()) 
        if(p == this || in(p)) 
            return true;

    return false;
}

bool MxMeshObj::has(MxMeshObj *obj) {
    return obj && obj->in(this);
}
