/**
 * @file Mx3DFIO.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format import/export interface
 * @date 2021-12-13
 * 
 */

#ifndef SRC_MX_IO_MX3DFIO_H_
#define SRC_MX_IO_MX3DFIO_H_

#include <mechanica_private.h>

#include "Mx3DFStructure.h"


struct CAPI_EXPORT Mx3DFIO {
    /**
     * @brief Load a 3D format file
     * 
     * @param filePath path of file
     * @return Mx3DFStructure* 3D format data container
     */
    static Mx3DFStructure *fromFile(const std::string &filePath);

    /**
     * @brief Export engine state to a 3D format file
     * 
     * @param format format of file
     * @param filePath path of file
     * @param pRefinements mesh refinements applied when generating meshes
     * @return HRESULT 
     */
    static HRESULT toFile(const std::string &format, const std::string &filePath, const unsigned int &pRefinements=0);
};


#endif // SRC_MX_IO_MX3DFIO_H_
