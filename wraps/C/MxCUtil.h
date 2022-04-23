/**
 * @file MxCUtil.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxUtil
 * @date 2022-04-04
 */

#ifndef _WRAPS_C_MXCUTIL_H_
#define _WRAPS_C_MXCUTIL_H_

#include <mx_port.h>

// Handles

struct CAPI_EXPORT MxPointsTypeHandle {
    unsigned int Sphere;
    unsigned int SolidSphere;
    unsigned int Disk;
    unsigned int SolidCube;
    unsigned int Cube;
    unsigned int Ring;
};


//////////////////
// MxPointsType //
//////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCPointsType_init(struct MxPointsTypeHandle *handle);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Get the current seed for the pseudo-random number generator
 * 
 */
CAPI_FUNC(HRESULT) MxCGetSeed(unsigned int *seed);

/**
 * @brief Set the current seed for the pseudo-random number generator
 * 
 * @param _seed 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSetSeed(unsigned int seed);

/**
 * @brief Get the names of all available colors
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCColor3_Names(char ***names, unsigned int *numNames);

/**
 * @brief Get the coordinates of a random point in a kind of shape. 
 * 
 * Currently supports sphere, disk, solid cube and solid sphere. 
 * 
 * @param kind kind of shape
 * @param dr thickness parameter; only applicable to solid sphere kind
 * @param phi0 angle lower bound; only applicable to solid sphere kind
 * @param phi1 angle upper bound; only applicable to solid sphere kind
 * @param x x-coordinate of random point
 * @param y y-coordinate of random point
 * @param z z-coordinate of random point
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCRandomPoint(unsigned int kind, float dr, float phi0, float phi1, float *x, float *y, float *z);

/**
 * @brief Get the coordinates of random points in a kind of shape. 
 * 
 * Currently supports sphere, disk, solid cube and solid sphere.
 * 
 * @param kind kind of shape
 * @param n number of points
 * @param dr thickness parameter; only applicable to solid sphere kind
 * @param phi0 angle lower bound; only applicable to solid sphere kind
 * @param phi1 angle upper bound; only applicable to solid sphere kind
 * @param x coordinates of random points
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCRandomPoints(unsigned int kind, int n, float dr, float phi0, float phi1, float **x);

/**
 * @brief Get the coordinates of uniform points in a kind of shape. 
 * 
 * Currently supports ring and sphere. 
 * 
 * @param kind kind of shape
 * @param n number of points
 * @param x coordinates of points
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPoints(unsigned int kind, int n, float **x);

/**
 * @brief Get the coordinates of a uniformly filled cube. 
 * 
 * @param corner1 first corner of cube
 * @param corner2 second corner of cube
 * @param nParticlesX number of particles along x-direction of filling axes (>=2)
 * @param nParticlesY number of particles along y-direction of filling axes (>=2)
 * @param nParticlesZ number of particles along z-direction of filling axes (>=2)
 * @param x x-coordinate of points
 * @param y y-coordinate of points
 * @param z z-coordinate of points
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFilledCubeUniform(float *corner1, float *corner2, 
                                        unsigned int nParticlesX, 
                                        unsigned int nParticlesY, 
                                        unsigned int nParticlesZ, 
                                        float **x);

/**
 * @brief Get the coordinates of a randomly filled cube. 
 * 
 * @param corner1 first corner of cube
 * @param corner2 second corner of cube
 * @param nParticles number of points in the cube
 * @param x x-coordinate of points
 * @param y y-coordinate of points
 * @param z z-coordinate of points
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFilledCubeRandom(float *corner1, float *corner2, int nParticles, float **x);

/**
 * @brief Get the coordinates of an icosphere. 
 * 
 * @param subdivisions number of subdivisions
 * @param phi0 angle lower bound
 * @param phi1 angle upper bound
 * @param verts returned vertices
 * @param numVerts number of vertices
 * @param inds returned indices
 * @param numInds number of indices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCIcosphere(unsigned int subdivisions, 
                                float phi0, 
                                float phi1,
                                float **verts, 
                                unsigned int *numVerts,
                                int **inds, 
                                unsigned int *numInds);

/**
 * @brief Generates a randomly oriented vector with random magnitude 
 * with given mean and standard deviation according to a normal 
 * distribution.
 * 
 * @param mean magnitude mean
 * @param std magnitude standard deviation
 * @param x x-component of vector
 * @param y y-component of vector
 * @param z z-component of vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCRandomVector(float mean, float std, float *x, float *y, float *z);

/**
 * @brief Generates a randomly oriented unit vector.
 * 
 * @param x x-component of vector
 * @param y y-component of vector
 * @param z z-component of vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCRandomUnitVector(float *x, float *y, float *z);

/**
 * @brief Get the compiler features names and flags
 * 
 * @param names feature names
 * @param flags feature flags
 * @param numFeatures number of features
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCGetFeaturesMap(char ***names, bool **flags, unsigned int *numFeatures);

/**
 * @brief Get the current wall time
 * 
 * @param wtime wall time
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCWallTime(double *wtime);

/**
 * @brief Get the current CPU time
 * 
 * @param cputime CPU time
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCPUTime(double *cputime);

#endif // _WRAPS_C_MXCUTIL_H_