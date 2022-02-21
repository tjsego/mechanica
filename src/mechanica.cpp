/**
 * mechanica.cpp
 *
 * Initialize the mechanica module, python init functions.
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#include "mechanica_private.h"
#include <MxSimulator.h>
#include <mx_error.h>
#include <MxLogger.h>

#include <Magnum/GL/Context.h>
#include <string>

static std::string version_str() {
    #if MX_VERSION_DEV
        std::string dev = "-dev" + std::to_string(MX_VERSION_DEV);
    #else
        std::string dev = "";
    #endif

    std::string version = std::string(MX_VERSION) + dev;
    return version;
}

static std::string systemNameStr() {
    return std::string(MX_SYSTEM_NAME);
}

static std::string systemVersionStr() {
    return std::string(MX_SYSTEM_VERSION);
}

static std::string compilerIdStr() {
    return std::string(MX_COMPILER_ID);
}

static std::string compilerVersionStr() {
    return std::string(MX_COMPILER_VERSION);
}

/**
 * Initialize the entire runtime.
 */
CAPI_FUNC(HRESULT) Mx_Initialize(int args) {

    Log(LOG_TRACE);
    
    // GL symbols are globals in each shared library address space,
    // if the app already initialized gl, we need to get the symbols here
    if(Magnum::GL::Context::hasCurrent() && !glCreateProgram) {
        flextGLInit(Magnum::GL::Context::current());
    }


    return modules_init();
}
