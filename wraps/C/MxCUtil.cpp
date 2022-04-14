/**
 * @file MxCUtil.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxUtil
 * @date 2022-04-04
 */

#include "MxCUtil.h"

#include "mechanica_c_private.h"

#include <MxUtil.h>


//////////////////
// MxPointsType //
//////////////////


HRESULT MxCPointsType_init(struct MxPointsTypeHandle *handle) {
    MXCPTRCHECK(handle);
    handle->Sphere = (unsigned int)MxPointsType::Sphere;
    handle->SolidSphere = (unsigned int)MxPointsType::SolidSphere;
    handle->Disk = (unsigned int)MxPointsType::Disk;
    handle->SolidCube = (unsigned int)MxPointsType::SolidCube;
    handle->Cube = (unsigned int)MxPointsType::Cube;
    handle->Ring = (unsigned int)MxPointsType::Ring;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT MxCGetSeed(unsigned int *seed) {
    MXCPTRCHECK(seed);
    *seed = MxGetSeed();
    return S_OK;
}

HRESULT MxCSetSeed(unsigned int seed) {
    return MxSetSeed(&seed);
}

HRESULT MxCColor3_Names(char ***names, unsigned int *numNames) {
    MXCPTRCHECK(names);
    MXCPTRCHECK(numNames);
    
    std::vector<std::string> _namesV = MxColor3_Names();
    *numNames = _namesV.size();
    if(*numNames > 0) {
        char **_names = (char**)malloc(*numNames * sizeof(char*));
        for(unsigned int i = 0; i < *numNames; i++) {
            std::string _s = _namesV[i];
            char *_c = new char[_s.size() + 1];
            std::strcpy(_c, _s.c_str());
            _names[i] = _c;
        }
        *names = _names;
    }
    return S_OK;
}

HRESULT MxCRandomPoint(unsigned int kind, float dr, float phi0, float phi1, float *x, float *y, float *z) {
    MXCPTRCHECK(x); MXCPTRCHECK(y); MXCPTRCHECK(z);
    auto p = MxRandomPoint((MxPointsType)kind, dr, phi0, phi1);
    *x = p.x(); *y = p.y(); *z = p.z();
    return S_OK;
}

HRESULT MxCRandomPoints(unsigned int kind, int n, float dr, float phi0, float phi1, float **x) {
    auto pv = MxRandomPoints((MxPointsType)kind, n, dr, phi0, phi1);
    return pv.size() > 0 ? mx::capi::copyVecVecs3_2Arr(pv, x) : S_OK;
}

HRESULT MxCPoints(unsigned int kind, int n, float **x) {
    auto pv = MxPoints((MxPointsType)kind, n);
    return pv.size() > 0 ? mx::capi::copyVecVecs3_2Arr(pv, x) : S_OK;
}

HRESULT MxCFilledCubeUniform(float *corner1, float *corner2, 
                             unsigned int nParticlesX, unsigned int nParticlesY, unsigned int nParticlesZ, 
                             float **x) 
{
    MXCPTRCHECK(corner1);
    MXCPTRCHECK(corner2);
    auto pv = MxFilledCubeUniform(MxVector3f::from(corner1), MxVector3f::from(corner2), nParticlesX, nParticlesY, nParticlesZ);
    return pv.size() > 0 ? mx::capi::copyVecVecs3_2Arr(pv, x) : S_OK;
}

HRESULT MxCFilledCubeRandom(float *corner1, float *corner2, int nParticles, float **x) {
    MXCPTRCHECK(corner1);
    MXCPTRCHECK(corner2);
    auto pv = MxFilledCubeRandom(MxVector3f::from(corner1), MxVector3f::from(corner2), nParticles);
    return pv.size() > 0 ? mx::capi::copyVecVecs3_2Arr(pv, x) : S_OK;
}

HRESULT MxCIcosphere(unsigned int subdivisions, float phi0, float phi1,
                     float **verts, unsigned int *numVerts,
                     int **inds, unsigned int *numInds) 
{
    MXCPTRCHECK(verts);
    MXCPTRCHECK(numVerts);
    MXCPTRCHECK(inds);
    MXCPTRCHECK(numInds);

    std::vector<MxVector3f> _verts;
    std::vector<int> _indsV;
    HRESULT result = Mx_Icosphere(subdivisions, phi0, phi1, _verts, _indsV);
    if(result != S_OK) 
        return result;

    *numInds = _indsV.size();
    *numVerts = _verts.size();
    int *_inds = (int*)malloc(*numInds * sizeof(int));
    if(!_inds) 
        return E_OUTOFMEMORY;
    for(unsigned int i = 0; i < _indsV.size(); i++) 
        _inds[i] = _indsV[i];
    *inds = _inds;
    return mx::capi::copyVecVecs3_2Arr(_verts, verts);
}

HRESULT MxCRandomVector(float mean, float std, float *x, float *y, float *z) {
    MXCPTRCHECK(x); MXCPTRCHECK(y); MXCPTRCHECK(z);
    auto pv = MxRandomVector(mean, std);
    *x = pv.x(); *y = pv.y(); *z = pv.z();
    return S_OK;
}

HRESULT MxCRandomUnitVector(float *x, float *y, float *z) {
    MXCPTRCHECK(x); MXCPTRCHECK(y); MXCPTRCHECK(z);
    auto pv = MxRandomUnitVector();
    *x = pv.x(); *y = pv.y(); *z = pv.z();
    return S_OK;
}

HRESULT MxCGetFeaturesMap(char ***names, bool **flags, unsigned int *numFeatures) {
    MXCPTRCHECK(names);
    MXCPTRCHECK(flags);
    MXCPTRCHECK(numFeatures);
    auto fmap = getFeaturesMap();
    *numFeatures = fmap.size();
    if(*numFeatures > 0) {
        char **_names = (char**)malloc(*numFeatures * sizeof(char*));
        bool *_flags = (bool*)malloc(*numFeatures * sizeof(bool));
        if(!_names || !_flags) 
            return E_OUTOFMEMORY;
        unsigned int i = 0;
        for(auto &fm : fmap) {
            char *_c = new char[fm.first.size() + 1];
            std::strcpy(_c, fm.first.c_str());
            _names[i] = _c;
            _flags[i] = fm.second;
            i++;
        }
        *names = _names;
        *flags = _flags;
    }
    return S_OK;
}

HRESULT MxCWallTime(double *wtime) {
    MXCPTRCHECK(wtime);
    *wtime = MxWallTime();
    return S_OK;
}

HRESULT MxCCPUTime(double *cputime) {
    MXCPTRCHECK(cputime);
    *cputime = MxCPUTime();
    return S_OK;
}
