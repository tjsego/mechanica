/**
 * @file mechanica_c_private.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines private support methods for the Mechanica C API
 * @date 2022-03-28
 */

#ifndef _WRAPS_C_MECHANICA_C_PRIVATE_H_
#define _WRAPS_C_MECHANICA_C_PRIVATE_H_

#include <string>
#include <vector>

#include <types/mx_types.h>

#define MXCPTRCHECK(varname) if(!varname) return E_FAIL;

// Convenience macros for array operations

#define MXVECTOR2_COPYFROM(vec, arr) arr[0] = vec.x(); arr[1] = vec.y();
#define MXVECTOR2_COPYTO(arr, vec)   vec.x() = arr[0]; vec.y() = arr[1];
#define MXVECTOR3_COPYFROM(vec, arr) arr[0] = vec.x(); arr[1] = vec.y(); arr[2] = vec.z();
#define MXVECTOR3_COPYTO(arr, vec)   vec.x() = arr[0]; vec.y() = arr[1]; vec.z() = arr[2];
#define MXVECTOR4_COPYFROM(vec, arr) arr[0] = vec.x(); arr[1] = vec.y(); arr[2] = vec.z();; arr[3] = vec.w();
#define MXVECTOR4_COPYTO(arr, vec)   vec.x() = arr[0]; vec.y() = arr[1]; vec.z() = arr[2]; vec.w() = arr[3];
#define MXMATRIX2_COPYFROM(mat, arr) arr[0] = mat[0][0]; arr[1] = mat[0][1]; \
                                     arr[2] = mat[1][0]; arr[3] = mat[1][1];
#define MXMATRIX2_COPYTO(arr, mat)   mat[0][0] = arr[0]; mat[0][1] = arr[1]; \
                                     mat[1][0] = arr[2]; mat[1][1] = arr[2];
#define MXMATRIX3_COPYFROM(mat, arr) arr[0] = mat[0][0]; arr[1] = mat[0][1]; arr[2] = mat[0][2]; \
                                     arr[3] = mat[1][0]; arr[4] = mat[1][1]; arr[5] = mat[1][2]; \
                                     arr[6] = mat[2][0]; arr[7] = mat[2][1]; arr[8] = mat[2][2];
#define MXMATRIX3_COPYTO(arr, mat)   mat[0][0] = arr[0]; mat[0][1] = arr[1]; mat[0][2] = arr[2]; \
                                     mat[1][0] = arr[3]; mat[1][1] = arr[4]; mat[1][2] = arr[5]; \
                                     mat[2][0] = arr[6]; mat[2][1] = arr[7]; mat[2][2] = arr[8];
#define MXMATRIX4_COPYFROM(mat, arr) arr[0]  = mat[0][0]; arr[1]  = mat[0][1]; arr[2]  = mat[0][2]; arr[3]  = mat[0][3]; \
                                     arr[4]  = mat[1][0]; arr[5]  = mat[1][1]; arr[6]  = mat[1][2]; arr[7]  = mat[1][3]; \
                                     arr[8]  = mat[2][0]; arr[9]  = mat[2][1]; arr[10] = mat[2][2]; arr[11] = mat[2][3]; \
                                     arr[12] = mat[3][0]; arr[13] = mat[3][1]; arr[14] = mat[3][2]; arr[15] = mat[3][3];
#define MXMATRIX4_COPYTO(arr, mat)   mat[0][0] = arr[0];  mat[0][1] = arr[1];  mat[0][2] = arr[2];  mat[0][3] = arr[3];  \
                                     mat[1][0] = arr[4];  mat[1][1] = arr[5];  mat[1][2] = arr[6];  mat[1][3] = arr[7];  \
                                     mat[2][0] = arr[8];  mat[2][1] = arr[9];  mat[2][2] = arr[10]; mat[2][3] = arr[11]; \
                                     mat[3][0] = arr[12]; mat[3][1] = arr[13]; mat[3][2] = arr[14]; mat[3][3] = arr[15];


// Standard template handle casts

namespace mx {

template <typename O, typename H>
O *castC(H *h) {
    if(!h || !h->MxObj) 
        return NULL;
    return (O*)h->MxObj;
}

template <typename O, typename H>
HRESULT castC(O &obj, H *handle) {
    MXCPTRCHECK(handle);
    handle->MxObj = (void*)&obj;
    return S_OK;
}


namespace capi {

HRESULT str2Char(const std::string s, char **c, unsigned int *n);

std::vector<std::string> charA2StrV(const char **c, const unsigned int &n);

template <typename O, typename H>
bool destroyHandle(H *h) {
    if(!h || !h->MxObj) {
        delete (O*)h->MxObj;
        h->MxObj = NULL;
        return true;
    }
    return false;
}

template<typename T> 
HRESULT copyVecVecs2_2Arr(const std::vector<mx::type::MxVector2<T> > &vecsV, T **vecsA) {
    unsigned int n = 2;
    MXCPTRCHECK(vecsA);
    auto nr_parts = vecsV.size();
    T *_vecsA = (T*)malloc(n * nr_parts * sizeof(T));
    if(!_vecsA) 
        return E_OUTOFMEMORY;
    for(unsigned int i = 0; i < nr_parts; i++) {
        auto _aV = vecsV[i];
        MXVECTOR2_COPYFROM(_aV, (_vecsA + n * i));
    }
    *vecsA = _vecsA;
    return S_OK;
}

template<typename T> 
HRESULT copyVecVecs3_2Arr(const std::vector<mx::type::MxVector3<T> > &vecsV, T **vecsA) {
    unsigned int n = 3;
    MXCPTRCHECK(vecsA);
    auto nr_parts = vecsV.size();
    T *_vecsA = (T*)malloc(n * nr_parts * sizeof(T));
    if(!_vecsA) 
        return E_OUTOFMEMORY;
    for(unsigned int i = 0; i < nr_parts; i++) {
        auto _aV = vecsV[i];
        MXVECTOR3_COPYFROM(_aV, (_vecsA + n * i));
    }
    *vecsA = _vecsA;
    return S_OK;
}

template<typename T> 
HRESULT copyVecVecs4_2Arr(const std::vector<mx::type::MxVector4<T> > &vecsV, T **vecsA) {
    unsigned int n = 4;
    MXCPTRCHECK(vecsA);
    auto nr_parts = vecsV.size();
    T *_vecsA = (T*)malloc(n * nr_parts * sizeof(T));
    if(!_vecsA) 
        return E_OUTOFMEMORY;
    for(unsigned int i = 0; i < nr_parts; i++) {
        auto _aV = vecsV[i];
        MXVECTOR4_COPYFROM(_aV, (_vecsA + n * i));
    }
    *vecsA = _vecsA;
    return S_OK;
}

}

}

#endif // _WRAPS_C_MECHANICA_C_PRIVATE_H_