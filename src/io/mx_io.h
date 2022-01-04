/**
 * @file mx_io.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica import/export interface
 * @date 2021-12-13
 * 
 */

#ifndef SRC_IO_MX_IO_H_
#define SRC_IO_MX_IO_H_

#include <mx_port.h>
#include <mx_config.h>
#include "../types/mx_types.h"

#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <string>


using MxIOChildMap = std::unordered_map<std::string, struct MxIOElement*>;

/**
 * @brief Intermediate I/O class for reading/writing 
 * Mechanica objects to/from file/string. 
 * 
 */
struct MxIOElement {
    std::string type;
    std::string value;
    MxIOElement *parent = NULL;
    MxIOChildMap children;
};

/**
 * @brief Mechanica meta data. 
 * 
 * An instance is always stored in any object export. 
 * 
 */
struct MxMetaData {
    unsigned int versionMajor = MX_VERSION_MAJOR;
    unsigned int versionMinor = MX_VERSION_MINOR;
    unsigned int versionPatch = MX_VERSION_PATCH;
};

namespace mx { namespace io {

/**
 * @brief Convert an object to an intermediate I/O object
 * 
 * @tparam T type of object to convert
 * @param dataElement object to convert
 * @param metaData meta data of target installation
 * @param fileElement resulting I/O object
 * @return HRESULT 
 */
template <typename T>
HRESULT toFile(const T &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

/**
 * @brief Instantiate an object from an intermediate I/O object
 * 
 * @tparam T type of object to instantiate
 * @param fileElement source I/O object
 * @param metaData meta data of exporting installation
 * @param dataElement resulting object
 * @return HRESULT 
 */
template <typename T>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, T *dataElement);


// Mechanica types


// MxMetaData

template <>
HRESULT toFile(const MxMetaData &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxMetaData *dataElement);

// mx::type::MxVector2<T>

template <typename T>
HRESULT toFile(const mx::type::MxVector2<T> &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    fileElement->type = "MxVector2";
    MxIOElement *xfe = new MxIOElement();
    MxIOElement *yfe = new MxIOElement();

    if(toFile(dataElement.x(), metaData, xfe) != S_OK || toFile(dataElement.y(), metaData, yfe) != S_OK) 
        return E_FAIL;
    
    xfe->parent = fileElement;
    yfe->parent = fileElement;
    fileElement->children["x"] = xfe;
    fileElement->children["y"] = yfe;

    return S_OK;
}

template <typename T>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, mx::type::MxVector2<T> *dataElement) {

    T de;
    auto feItr = fileElement.children.find("x");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
        return E_FAIL;
    
    (*dataElement)[0] = de;

    feItr = fileElement.children.find("y");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
        return E_FAIL;
    
    (*dataElement)[1] = de;
    
    return S_OK;
}

// mx::type::MxVector3<T>

template <typename T>
HRESULT toFile(const mx::type::MxVector3<T> &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    fileElement->type = "MxVector3";
    MxIOElement *xfe = new MxIOElement();
    MxIOElement *yfe = new MxIOElement();
    MxIOElement *zfe = new MxIOElement();

    if(toFile(dataElement.x(), metaData, xfe) != S_OK || 
       toFile(dataElement.y(), metaData, yfe) != S_OK || 
       toFile(dataElement.z(), metaData, zfe) != S_OK) 
        return E_FAIL;
    
    xfe->parent = fileElement;
    yfe->parent = fileElement;
    zfe->parent = fileElement;
    fileElement->children["x"] = xfe;
    fileElement->children["y"] = yfe;
    fileElement->children["z"] = zfe;

    return S_OK;
}

template <typename T>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, mx::type::MxVector3<T> *dataElement) {

    T de;
    auto feItr = fileElement.children.find("x");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
        return E_FAIL;
    
    (*dataElement)[0] = de;

    feItr = fileElement.children.find("y");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
        return E_FAIL;
    
    (*dataElement)[1] = de;

    feItr = fileElement.children.find("z");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
        return E_FAIL;
    
    (*dataElement)[2] = de;
    
    return S_OK;
}

// mx::type::MxVector4<T>

template <typename T>
HRESULT toFile(const mx::type::MxVector4<T> &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    fileElement->type = "MxVector4";
    MxIOElement *xfe = new MxIOElement();
    MxIOElement *yfe = new MxIOElement();
    MxIOElement *zfe = new MxIOElement();
    MxIOElement *wfe = new MxIOElement();

    if(toFile(dataElement.x(), metaData, xfe) != S_OK || 
       toFile(dataElement.y(), metaData, yfe) != S_OK || 
       toFile(dataElement.z(), metaData, zfe) != S_OK || 
       toFile(dataElement.w(), metaData, zfe) != S_OK) 
        return E_FAIL;
    
    xfe->parent = fileElement;
    yfe->parent = fileElement;
    zfe->parent = fileElement;
    wfe->parent = fileElement;
    fileElement->children["x"] = xfe;
    fileElement->children["y"] = yfe;
    fileElement->children["z"] = zfe;
    fileElement->children["w"] = wfe;

    return S_OK;
}

template <typename T>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, mx::type::MxVector4<T> *dataElement) {

    T de;
    auto feItr = fileElement.children.find("x");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
        return E_FAIL;
    
    (*dataElement)[0] = de;

    feItr = fileElement.children.find("y");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
        return E_FAIL;
    
    (*dataElement)[1] = de;

    feItr = fileElement.children.find("z");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
        return E_FAIL;
    
    (*dataElement)[2] = de;

    feItr = fileElement.children.find("w");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
        return E_FAIL;
    
    (*dataElement)[3] = de;
    
    return S_OK;
}

// mx::type::MxMatrix3<T>

template <typename T>
HRESULT toFile(const mx::type::MxMatrix3<T> &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    fileElement->type = "MxMatrix3";

    for(unsigned int i = 0; i < 3; i++) {
        for (unsigned int j = 0; j < 3; j++) { 
            std::string key = std::to_string(i) + std::to_string(j);
            MxIOElement *fe = new MxIOElement();
            if(toFile(dataElement[i][j], metaData, fe) != S_OK) 
                return E_FAIL;
            fe->parent = fileElement;
            fileElement->children[key] = fe;
        }
    }

    return S_OK;
}

template <typename T>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, mx::type::MxMatrix3<T> *dataElement) {

    T de;

    for(unsigned int i = 0; i < 3; i++) {
        for(unsigned int j = 0; j < 3; j++) { 
            std::string key = std::to_string(i) + std::to_string(j);
            auto feItr = fileElement.children.find(key);
            if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
                return E_FAIL;
            (*dataElement)[i][j] = de;
        }
    }

    return S_OK;
}

// mx::type::MxMatrix4<T>

template <typename T>
HRESULT toFile(const mx::type::MxMatrix4<T> &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    fileElement->type = "MxMatrix4";

    for(unsigned int i = 0; i < 4; i++) {
        for (unsigned int j = 0; j < 4; j++) { 
            std::string key = std::to_string(i) + std::to_string(j);
            MxIOElement *fe = new MxIOElement();
            if(toFile(dataElement[i][j], metaData, fe) != S_OK) 
                return E_FAIL;
            fe->parent = fileElement;
            fileElement->children[key] = fe;
        }
    }

    return S_OK;
}

template <typename T>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, mx::type::MxMatrix4<T> *dataElement) {

    T de;

    for(unsigned int i = 0; i < 4; i++) {
        for(unsigned int j = 0; j < 4; j++) { 
            std::string key = std::to_string(i) + std::to_string(j);
            auto feItr = fileElement.children.find(key);
            if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
                return E_FAIL;
            (*dataElement)[i][j] = de;
        }
    }

    return S_OK;
}

// mx::type::MxQuaternion<T>

template <typename T>
HRESULT toFile(const mx::type::MxQuaternion<T> &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    fileElement->type = "MxQuaternion";
    MxIOElement *vfe = new MxIOElement();
    MxIOElement *sfe = new MxIOElement();
    
    if(toFile(dataElement.vector(), metaData, vfe) != S_OK || toFile(dataElement.scalar(), metaData, sfe) != S_OK) 
        return E_FAIL;

    vfe->parent = fileElement;
    sfe->parent = fileElement;
    fileElement->children["vector"] = vfe;
    fileElement->children["scalar"] = sfe;

    return S_OK;
}

template <typename T>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, mx::type::MxQuaternion<T> *dataElement) { 

    std::vector<T> vde;
    T sde;

    auto feItr = fileElement.children.find("vector");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &vde) != S_OK) 
        return E_FAIL;
    dataElement->vector() = vde;

    feItr = fileElement.children.find("scalar");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &sde) != S_OK) 
        return E_FAIL;
    dataElement->scalar() = sde;

    return S_OK;
}


// Built-in implementations


// char

template <>
HRESULT toFile(const char &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, char *dataElement);

// signed char

template <>
HRESULT toFile(const signed char &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, signed char *dataElement);

// unsigned char

template <>
HRESULT toFile(const unsigned char &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, unsigned char *dataElement);

// short

template <>
HRESULT toFile(const short &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, short *dataElement);

// unsigned short

template <>
HRESULT toFile(const unsigned short &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, unsigned short *dataElement);

// int

template <>
HRESULT toFile(const int &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, int *dataElement);

// unsigned int

template <>
HRESULT toFile(const unsigned int &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, unsigned int *dataElement);

// bool

template <>
HRESULT toFile(const bool &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, bool *dataElement);

// long

template <>
HRESULT toFile(const long &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, long *dataElement);

// unsigned long

template <>
HRESULT toFile(const unsigned long &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, unsigned long *dataElement);

// long long

template <>
HRESULT toFile(const long long &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, long long *dataElement);

// unsigned long long

template <>
HRESULT toFile(const unsigned long long &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, unsigned long long *dataElement);

// float

template <>
HRESULT toFile(const float &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, float *dataElement);

// double

template <>
HRESULT toFile(const double &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, double *dataElement);

// string

template <>
HRESULT toFile(const std::string &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, std::string *dataElement);

// Containers

// set

template <typename T>
HRESULT toFile(const std::set<T> &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    fileElement->type = "set";
    fileElement->children.reserve(dataElement.size());
    unsigned int i = 0;
    for(auto de : dataElement) {
        MxIOElement *fe = new MxIOElement();
        if(toFile(de, metaData, fe) != S_OK) 
            return E_FAIL;
        
        fe->parent = fileElement;
        fileElement->children[std::to_string(i)] = fe;
        i++;
    }
    return S_OK;
}

template <typename T>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, std::set<T> *dataElement) {
    unsigned int numEls = fileElement.children.size();
    for(unsigned int i = 0; i < numEls; i++) {
        T de;
        auto itr = fileElement.children.find(std::to_string(i));
        if(itr == fileElement.children.end()) 
            return E_FAIL;
        if(fromFile(*itr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        dataElement->insert(de);
    }
    return S_OK;
}

// unordered_set

template <typename T>
HRESULT toFile(const std::unordered_set<T> &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    fileElement->type = "unordered_set";
    fileElement->children.reserve(dataElement.size());
    unsigned int i = 0;
    for(auto de : dataElement) {
        MxIOElement *fe = new MxIOElement();
        if(toFile(de, metaData, fe) != S_OK) 
            return E_FAIL;
        
        fe->parent = fileElement;
        fileElement->children[std::to_string(i)] = fe;
        i++;
    }
    return S_OK;
}

template <typename T>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, std::unordered_set<T> *dataElement) {
    unsigned int numEls = fileElement.children.size();
    for(unsigned int i = 0; i < numEls; i++) {
        T de;
        auto itr = fileElement.children.find(std::to_string(i));
        if(itr == fileElement.children.end()) 
            return E_FAIL;
        if(fromFile(*itr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        dataElement->insert(de);
    }
    return S_OK;
}

// vector

template <typename T>
HRESULT toFile(const std::vector<T> &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    fileElement->type = "vector";
    fileElement->children.reserve(dataElement.size());
    for(unsigned int i = 0; i < dataElement.size(); i++) {
        MxIOElement *fe = new MxIOElement();
        if(toFile(dataElement[i], metaData, fe) != S_OK) 
            return E_FAIL;
        
        fe->parent = fileElement;
        fileElement->children[std::to_string(i)] = fe;
    }
    return S_OK;
}

template <typename T>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, std::vector<T> *dataElement) {
    unsigned int numEls = fileElement.children.size();
    dataElement->reserve(numEls);
    for(unsigned int i = 0; i < numEls; i++) {
        T de;
        auto itr = fileElement.children.find(std::to_string(i));
        if(itr == fileElement.children.end()) 
            return E_FAIL;
        if(fromFile(*itr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        dataElement->push_back(de);
    }
    return S_OK;
}

// map

template <typename S, typename T>
HRESULT toFile(const std::map<S, T> &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    fileElement->type = "map";
    
    std::vector<S> keysde;
    std::vector<T> valsde;

    for(typename std::map<S, T>::iterator de = dataElement.begin(); de != dataElement.end(); de++) {
        keysde.push_back(de->first);
        valsde.push_back(de->second);
    }

    MxIOElement *keysfe = new MxIOElement();
    MxIOElement *valsfe = new MxIOElement();
    if(toFile(keysde, metaData, keysfe) != S_OK || toFile(valsde, metaData, valsfe) != S_OK) 
        return E_FAIL;
    
    keysfe->parent = fileElement;
    valsfe->parent = fileElement;
    fileElement->children["keys"] = keysfe;
    fileElement->children["values"] = valsfe;

    return S_OK;
}

template <typename S, typename T>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, std::map<S, T> *dataElement) {
    
    MxIOElement *keysfe, *valsfe;
    MXIOFINDSAFE(fileElement, keysfeItr, "keys", keysfe);
    MXIOFINDSAFE(fileElement, valsfeItr, "values", valsfe);

    std::vector<S> keysde;
    std::vector<T> valsde;
    if(fromFile(*keysfe, metaData, &keysde) != S_OK || fromFile(*valsfe, metaData, &valsde) != S_OK) 
        return E_FAIL;
    
    for(unsigned int i = 0; i < keysde.size(); i++) {
        (*dataElement)[keysde[i]] = valsde[i];
    }

    return S_OK;
}

// unordered_map

template <typename S, typename T>
HRESULT toFile(const std::unordered_map<S, T> &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    fileElement->type = "unordered_map";
    
    std::vector<S> keysde;
    std::vector<T> valsde;
    
    for(typename std::map<S, T>::iterator de = dataElement.begin(); de != dataElement.end(); de++) {
        keysde.push_back(de->first);
        valsde.push_back(de->second);
    }

    MxIOElement *keysfe = new MxIOElement();
    MxIOElement *valsfe = new MxIOElement();
    if(toFile(keysde, metaData, keysfe) != S_OK || toFile(valsde, metaData, valsfe) != S_OK) 
        return E_FAIL;
    
    keysfe->parent = fileElement;
    valsfe->parent = fileElement;
    fileElement->children["keys"] = keysfe;
    fileElement->children["values"] = valsfe;

    return S_OK;
}

template <typename S, typename T>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, std::unordered_map<S, T> *dataElement) {
    
    MxIOElement *keysfe, *valsfe;
    MXIOFINDSAFE(fileElement, keysfeItr, "keys", keysfe);
    MXIOFINDSAFE(fileElement, valsfeItr, "values", valsfe);

    std::vector<S> keysde;
    std::vector<T> valsde;
    if(fromFile(*keysfe, metaData, &keysde) != S_OK || fromFile(*valsfe, metaData, &valsde) != S_OK) 
        return E_FAIL;
    
    for(unsigned int i = 0; i < keysde.size(); i++) {
        (*dataElement)[keysde[i]] = valsde[i];
    }

    return S_OK;
}

}};

#endif // SRC_IO_MX_IO_H_
