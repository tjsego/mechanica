/**
 * @file MxCStateVector.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxStateVector
 * @date 2022-04-01
 */

#include "MxCStateVector.h"

#include "mechanica_c_private.h"

#include "MxCSpecies.h"

#include <state/MxStateVector.h>


namespace mx { 

MxStateVector *castC(struct MxStateVectorHandle *handle) {
    return castC<MxStateVector, MxStateVectorHandle>(handle);
}

}

#define MXSTATEVECTOR_GET(handle) \
    MxStateVector *svec = mx::castC<MxStateVector, MxStateVectorHandle>(handle); \
    MXCPTRCHECK(svec);


///////////////////
// MxStateVector //
///////////////////


HRESULT MxCStateVector_destroy(struct MxStateVectorHandle *handle) {
    return mx::capi::destroyHandle<MxStateVector, MxStateVectorHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCStateVector_getSize(struct MxStateVectorHandle *handle, unsigned int *size) {
    MXSTATEVECTOR_GET(handle);
    MXCPTRCHECK(size);
    *size = svec->size;
    return S_OK;
}

HRESULT MxCStateVector_getSpecies(struct MxStateVectorHandle *handle, struct MxSpeciesListHandle *slist) {
    MXSTATEVECTOR_GET(handle);
    MXCPTRCHECK(slist);
    slist->MxObj = (void*)svec->species;
    return S_OK;
}

HRESULT MxCStateVector_reset(struct MxStateVectorHandle *handle) {
    MXSTATEVECTOR_GET(handle);
    svec->reset();
    return S_OK;
}

HRESULT MxCStateVector_getStr(struct MxStateVectorHandle *handle, char **str, unsigned int *numChars) {
    MXSTATEVECTOR_GET(handle);
    return mx::capi::str2Char(svec->str(), str, numChars);
}

HRESULT MxCStateVector_getItem(struct MxStateVectorHandle *handle, int i, float *value) {
    MXSTATEVECTOR_GET(handle);
    MXCPTRCHECK(value);
    float *_value = svec->item(i);
    MXCPTRCHECK(_value);
    *value = *_value;
    return S_OK;
}

HRESULT MxCStateVector_setItem(struct MxStateVectorHandle *handle, int i, float value) {
    MXSTATEVECTOR_GET(handle);
    svec->setItem(i, value);
    return S_OK;
}

HRESULT MxCStateVector_toString(struct MxStateVectorHandle *handle, char **str, unsigned int *numChars) {
    MXSTATEVECTOR_GET(handle);
    return mx::capi::str2Char(svec->toString(), str, numChars);
}

HRESULT MxCStateVector_fromString(struct MxStateVectorHandle *handle, const char *str) {
    MxStateVector *svec = MxStateVector::fromString(str);
    MXCPTRCHECK(svec);
    handle->MxObj = (void*)svec;
    return S_OK;
}
