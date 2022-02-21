/**
 * @file mx_io.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica import/export interface
 * @date 2021-12-21
 * 
 */

#include <MxParticle.h>

#include "mx_io.h"

#include <limits>


#define MXIOEASYTOFILE(dataElement, typeName) \
    fileElement->value = std::to_string(dataElement); \
    fileElement->type = typeName; \
    return S_OK;

#define MXIOFINDSAFE(fileElement, itrName, keyName, valObj) \
    auto itrName = fileElement.children.find(keyName); \
    if(itrName == fileElement.children.end()) \
        return E_FAIL; \
    valObj = itrName->second;


namespace mx { namespace io {

// built-in types


// char

template <>
HRESULT toFile(const char &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MXIOEASYTOFILE(dataElement, "char");
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, char *dataElement) {
    *dataElement = fileElement.value[0];
    return S_OK;
}

// signed char

template <>
HRESULT toFile(const signed char &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MXIOEASYTOFILE(dataElement, "signed_char");
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, signed char *dataElement) {
    *dataElement = fileElement.value[0];
    return S_OK;
}

// unsigned char

template <>
HRESULT toFile(const unsigned char &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MXIOEASYTOFILE(dataElement, "unsigned_char");
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, unsigned char *dataElement) {
    *dataElement = fileElement.value[0];
    return S_OK;
}

// short

template <>
HRESULT toFile(const short &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    if(toFile((int)dataElement, metaData, fileElement) != S_OK) 
        return E_FAIL;
    fileElement->type = "short";
    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, short *dataElement) {
    int i;

    if(fromFile(fileElement, metaData, &i) != S_OK) 
        return E_FAIL;

    if(std::abs(i) >= std::numeric_limits<short>::max()) {
        mx_exp(std::range_error("Value exceeds numerical limits"));
        return E_FAIL;
    }

    *dataElement = i;

    return S_OK;
}

// unsigned short

template <>
HRESULT toFile(const unsigned short &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    if(toFile((unsigned int)dataElement, metaData, fileElement) != S_OK) 
        return E_FAIL;
    fileElement->type = "unsigned_short";
    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, unsigned short *dataElement) {
    unsigned int i;
    
    if(fromFile(fileElement, metaData, &i) != S_OK) 
        return E_FAIL;

    if(i >= std::numeric_limits<unsigned short>::max()) {
        mx_exp(std::range_error("Value exceeds numerical limits"));
        return E_FAIL;
    }

    *dataElement = i;

    return S_OK;
}

// int

template <>
HRESULT toFile(const int &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MXIOEASYTOFILE(dataElement, "int");
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, int *dataElement) {
    *dataElement = std::stoi(fileElement.value);
    return S_OK;
}

// unsigned int

template <>
HRESULT toFile(const unsigned int &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MXIOEASYTOFILE(dataElement, "unsigned_int");
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, unsigned int *dataElement) {
    *dataElement = std::stoul(fileElement.value);
    return S_OK;
}

// bool

template <>
HRESULT toFile(const bool &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MXIOEASYTOFILE(dataElement, "bool");
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, bool *dataElement) {
    unsigned int i;
    if(fromFile(fileElement, metaData, &i) != S_OK) 
        return E_FAIL;
    *dataElement = i;
    return S_OK;
}

// long

template <>
HRESULT toFile(const long &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MXIOEASYTOFILE(dataElement, "long");
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, long *dataElement) {
    *dataElement = std::stol(fileElement.value);
    return S_OK;
}

// unsigned long

template <>
HRESULT toFile(const unsigned long &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MXIOEASYTOFILE(dataElement, "unsigned_long");
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, unsigned long *dataElement) {
    *dataElement = std::stoul(fileElement.value);
    return S_OK;
}

// long long

template <>
HRESULT toFile(const long long &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MXIOEASYTOFILE(dataElement, "long_long");
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, long long *dataElement) {
    *dataElement = std::stoll(fileElement.value);
    return S_OK;
}

// unsigned long long

template <>
HRESULT toFile(const unsigned long long &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MXIOEASYTOFILE(dataElement, "unsigned_long_long");
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, unsigned long long *dataElement) {
    *dataElement = std::stoull(fileElement.value);
    return S_OK;
}

// float

template <>
HRESULT toFile(const float &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MXIOEASYTOFILE(dataElement, "float");
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, float *dataElement) {
    *dataElement = std::stof(fileElement.value);
    return S_OK;
}

// double

template <>
HRESULT toFile(const double &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MXIOEASYTOFILE(dataElement, "double");
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, double *dataElement) {
    *dataElement = std::stod(fileElement.value);
    return S_OK;
}

// string

template <>
HRESULT toFile(const std::string &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    fileElement->value = dataElement;
    fileElement->type = "string";
    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, std::string *dataElement) {
    *dataElement = std::string(fileElement.value);
    return S_OK;
}

// Containers


// Mechanica types


// MxMetaData

template <>
HRESULT toFile(const MxMetaData &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MxIOElement *vMajfe = new MxIOElement();
    MxIOElement *vMinfe = new MxIOElement();
    MxIOElement *vPatfe = new MxIOElement();

    if(toFile(dataElement.versionMajor, metaData, vMajfe) != S_OK) 
        return E_FAIL;
    if(toFile(dataElement.versionMinor, metaData, vMinfe) != S_OK) 
        return E_FAIL;
    if(toFile(dataElement.versionPatch, metaData, vPatfe) != S_OK) 
        return E_FAIL;
    
    vMajfe->parent = fileElement;
    vMinfe->parent = fileElement;
    vPatfe->parent = fileElement;
    fileElement->children["versionMajor"] = vMajfe;
    fileElement->children["versionMinor"] = vMinfe;
    fileElement->children["versionPatch"] = vPatfe;

    fileElement->type = "MetaData";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxMetaData *dataElement) { 

    MxIOElement *vMajfe, *vMinfe, *vPatfe;
    MXIOFINDSAFE(fileElement, vMajfeItr, "versionMajor", vMajfe);
    MXIOFINDSAFE(fileElement, vMinfeItr, "versionMinor", vMinfe);
    MXIOFINDSAFE(fileElement, vPatfeItr, "versionPatch", vPatfe);

    unsigned int vMajde, vMinde, vPatde;

    if(fromFile(*vMajfe, metaData, &vMajde) != S_OK) 
        return E_FAIL;
    if(fromFile(*vMinfe, metaData, &vMinde) != S_OK) 
        return E_FAIL;
    if(fromFile(*vPatfe, metaData, &vPatde) != S_OK) 
        return E_FAIL;

    dataElement->versionMajor = vMajde;
    dataElement->versionMinor = vMinde;
    dataElement->versionPatch = vPatde;

    return S_OK;
}

}};
