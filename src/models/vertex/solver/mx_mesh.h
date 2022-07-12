/**
 * @file mx_mesh.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic utilities for the Mechanica mesh
 * @date 2022-06-15
 * 
 */

#ifndef MODELS_VERTEX_SOLVER_MX_MESH_H_
#define MODELS_VERTEX_SOLVER_MX_MESH_H_

#include "MxMeshObj.h"


namespace mx { namespace models { namespace vertex {

MX_ALWAYS_INLINE bool check(MxMeshObj *obj, const MxMeshObj::Type &typeEnum) {
    return obj->objType() == typeEnum;
}

template<typename T>
std::vector<MxMeshObj*> vectorToBase(const std::vector<T*> &implVec) {
    return std::vector<MxMeshObj*>(implVec.begin(), implVec.end());
}

template<typename T> 
std::vector<T*> vectorToDerived(const std::vector<MxMeshObj*> &baseVec) {
    std::vector<T*> result(baseVec.size(), 0);
    for(unsigned int i = 0; i < baseVec.size(); i++) 
        result[i] = dynamic_cast<T*>(baseVec[i]);
    return result;
}

}}}

#endif // MODELS_VERTEX_SOLVER_MX_MESH_H_
