/**
 * @file Mx3DFRenderData.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format rendering data
 * @date 2021-12-20
 * 
 */


#ifndef SRC_MX_IO_MX3DFRENDERDATA_H_
#define SRC_MX_IO_MX3DFRENDERDATA_H_


#include <mechanica_private.h>


struct CAPI_EXPORT Mx3DFRenderData {

    MxVector3f color = {0.f, 0.f, 0.f};

};


#endif // SRC_MX_IO_MX3DFRENDERDATA_H_
