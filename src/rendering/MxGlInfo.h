/*
 * MxGlInfo.h
 *
 *  Created on: Apr 22, 2020
 *      Author: andy
 */

#ifndef SRC_RENDERING_MXGLINFO_H_
#define SRC_RENDERING_MXGLINFO_H_

#include "Mechanica.h"

#include <string>
#include <unordered_map>
#include <vector>

class CAPI_EXPORT MxGLInfo {

public:

    MxGLInfo() {};
    ~MxGLInfo() {};

    static const std::unordered_map<std::string, std::string> getInfo();
    static const std::vector<std::string> getExtensionsInfo();

};

std::unordered_map<std::string, std::string> Mx_GlInfo();

std::string Mx_EglInfo();

std::string gl_info();

#endif /* SRC_RENDERING_MXGLINFO_H_ */
