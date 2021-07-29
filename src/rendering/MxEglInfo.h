/*
 * MxEglInfo.h
 *
 *  Created on: Jun 21, 2021
 *      Author: t.j.
 */

#ifndef SRC_RENDERING_MXEGLINFO_H_
#define SRC_RENDERING_MXEGLINFO_H_

#include "MxGlInfo.h"

class CAPI_EXPORT MxEGLInfo {

public:

    MxEGLInfo() {};
    ~MxEGLInfo() {};

    static const std::string getInfo();

};

#endif // SRC_RENDERING_MXEGLINFO_H_
