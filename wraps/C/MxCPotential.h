/**
 * @file MxCPotential.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxPotential
 * @date 2022-03-30
 */

#ifndef _WRAPS_C_MXCPOTENTIAL_H_
#define _WRAPS_C_MXCPOTENTIAL_H_

#include <mx_port.h>

typedef void (*MxPotentialEval_ByParticleHandleFcn)(struct MxPotentialHandle *p, 
                                                    struct MxParticleHandleHandle *part_i, 
                                                    float *dx, 
                                                    float r2, 
                                                    float *e, 
                                                    float *f);
typedef void (*MxPotentialEval_ByParticlesHandleFcn)(struct MxPotentialHandle *p, 
                                                     struct MxParticleHandleHandle *part_i, 
                                                     struct MxParticleHandleHandle *part_j, 
                                                     float *dx, 
                                                     float r2, 
                                                     float *e, 
                                                     float *f);
typedef void (*MxPotentialEval_ByParticles3HandleFcn)(struct MxPotentialHandle *p, 
                                                      struct MxParticleHandleHandle *part_i, 
                                                      struct MxParticleHandleHandle *part_j, 
                                                      struct MxParticleHandleHandle *part_k, 
                                                      float ctheta, 
                                                      float *e, 
                                                      float *fi, 
                                                      float *fk);
typedef void (*MxPotentialEval_ByParticles4HandleFcn)(struct MxPotentialHandle *p, 
                                                      struct MxParticleHandleHandle *part_i, 
                                                      struct MxParticleHandleHandle *part_j, 
                                                      struct MxParticleHandleHandle *part_k, 
                                                      struct MxParticleHandleHandle *part_l, 
                                                      float cphi, 
                                                      float *e, 
                                                      float *fi, 
                                                      float *fl);
typedef void (*MxPotentialClearHandleFcn)(struct MxPotentialHandle *p);

// Handles

struct CAPI_EXPORT PotentialFlagsHandle {
    unsigned int POTENTIAL_NONE;
    unsigned int POTENTIAL_LJ126;
    unsigned int POTENTIAL_EWALD;
    unsigned int POTENTIAL_COULOMB;
    unsigned int POTENTIAL_SINGLE;
    unsigned int POTENTIAL_R2;
    unsigned int POTENTIAL_R;
    unsigned int POTENTIAL_ANGLE;
    unsigned int POTENTIAL_HARMONIC;
    unsigned int POTENTIAL_DIHEDRAL;
    unsigned int POTENTIAL_SWITCH;
    unsigned int POTENTIAL_REACTIVE;
    unsigned int POTENTIAL_SCALED;
    unsigned int POTENTIAL_SHIFTED;
    unsigned int POTENTIAL_BOUND;
    unsigned int POTENTIAL_SUM;
    unsigned int POTENTIAL_PERIODIC;
    unsigned int POTENTIAL_COULOMBR;
};

struct CAPI_EXPORT PotentialKindHandle {
    unsigned int POTENTIAL_KIND_POTENTIAL;
    unsigned int POTENTIAL_KIND_DPD;
    unsigned int POTENTIAL_KIND_BYPARTICLES;
    unsigned int POTENTIAL_KIND_COMBINATION;
};

/**
 * @brief Handle to a @ref MxPotentialEval_ByParticle instance
 * 
 */
struct CAPI_EXPORT MxPotentialEval_ByParticleHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxPotentialEval_ByParticles instance
 * 
 */
struct CAPI_EXPORT MxPotentialEval_ByParticlesHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxPotentialEval_ByParticles3 instance
 * 
 */
struct CAPI_EXPORT MxPotentialEval_ByParticles3Handle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxPotentialEval_ByParticles4 instance
 * 
 */
struct CAPI_EXPORT MxPotentialEval_ByParticles4Handle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxPotentialClear instance
 * 
 */
struct CAPI_EXPORT MxPotentialClearHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxPotential instance
 * 
 */
struct CAPI_EXPORT MxPotentialHandle {
    void *MxObj;
};



////////////////////
// PotentialFlags //
////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotentialFlags_init(struct PotentialFlagsHandle *handle);


///////////////////
// PotentialKind //
///////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotentialKind_init(struct PotentialKindHandle *handle);


////////////////////////////////
// MxPotentialEval_ByParticle //
////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param fcn evaluation function
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotentialEval_ByParticle_init(struct MxPotentialEval_ByParticleHandle *handle, MxPotentialEval_ByParticleHandleFcn *fcn);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotentialEval_ByParticle_destroy(struct MxPotentialEval_ByParticleHandle *handle);


/////////////////////////////////
// MxPotentialEval_ByParticles //
/////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param fcn evaluation function
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotentialEval_ByParticles_init(struct MxPotentialEval_ByParticlesHandle *handle, MxPotentialEval_ByParticlesHandleFcn *fcn);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotentialEval_ByParticles_destroy(struct MxPotentialEval_ByParticlesHandle *handle);


//////////////////////////////////
// MxPotentialEval_ByParticles3 //
//////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param fcn evaluation function
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotentialEval_ByParticles3_init(struct MxPotentialEval_ByParticles3Handle *handle, MxPotentialEval_ByParticles3HandleFcn *fcn);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotentialEval_ByParticles3_destroy(struct MxPotentialEval_ByParticles3Handle *handle);


//////////////////////////////////
// MxPotentialEval_ByParticles4 //
//////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param fcn evaluation function
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotentialEval_ByParticles4_init(struct MxPotentialEval_ByParticles4Handle *handle, MxPotentialEval_ByParticles4HandleFcn *fcn);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotentialEval_ByParticles4_destroy(struct MxPotentialEval_ByParticles4Handle *handle);


//////////////////////
// MxPotentialClear //
//////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param fcn evaluation function
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotentialClear_init(struct MxPotentialClearHandle *handle, MxPotentialClearHandleFcn *fcn);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotentialClear_destroy(struct MxPotentialClearHandle *handle);


/////////////////
// MxPotential //
/////////////////


/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_destroy(struct MxPotentialHandle *handle);

/**
 * @brief Get the name of the potential
 * 
 * @param handle populated handle
 * @param name name
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_getName(struct MxPotentialHandle *handle, char **name, unsigned int *numChars);

/**
 * @brief Set the name of the potential
 * 
 * @param handle populated handle
 * @param name name
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_setName(struct MxPotentialHandle *handle, const char *name);

/**
 * @brief Get the flags of the potential
 * 
 * @param handle populated handle
 * @param flags potential flags
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_getFlags(struct MxPotentialHandle *handle, unsigned int *flags);

/**
 * @brief Set the flags of the potential
 * 
 * @param handle populated handle
 * @param flags potential flags
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_setFlags(struct MxPotentialHandle *handle, unsigned int flags);

/**
 * @brief Get the kind of the potential
 * 
 * @param handle populated handle
 * @param kind potential kind enum
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_getKind(struct MxPotentialHandle *handle, unsigned int *kind);

/**
 * @brief Evaluate the potential
 * 
 * @param handle populated handle
 * @param r distance
 * @param potE evaluated potential energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_evalR(struct MxPotentialHandle *handle, float r, float *potE);

/**
 * @brief Evaluate the potential for a given scaling distance
 * 
 * @param handle populated handle
 * @param r distance
 * @param r0 scaling distance
 * @param potE evaluated potential energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_evalR0(struct MxPotentialHandle *handle, float r, float r0, float *potE);

/**
 * @brief Evaluate the potential for a given position
 * 
 * @param handle populated handle
 * @param pos position
 * @param potE evaluated potential energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_evalPos(struct MxPotentialHandle *handle, float *pos, float *potE);

/**
 * @brief Evaluate the potential for a given particle and position
 * 
 * @param handle populated handle
 * @param partHandle particle
 * @param pos position
 * @param potE evaluated potential energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_evalPart(struct MxPotentialHandle *handle, struct MxParticleHandleHandle *partHandle, float *pos, float *potE);

/**
 * @brief Evalute the potential for two particles
 * 
 * @param handle populated handle
 * @param phi first particle
 * @param phj second particle
 * @param potE evaluated potential energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_evalParts2(struct MxPotentialHandle *handle, 
                                           struct MxParticleHandleHandle *phi, 
                                           struct MxParticleHandleHandle *phj, 
                                           float *potE);

/**
 * @brief Evalute the potential for three particles
 * 
 * @param handle populated handle
 * @param phi first particle
 * @param phj second particle
 * @param phk third particle
 * @param potE evaluated potential energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_evalParts3(struct MxPotentialHandle *handle, 
                                           struct MxParticleHandleHandle *phi, 
                                           struct MxParticleHandleHandle *phj, 
                                           struct MxParticleHandleHandle *phk, 
                                           float *potE);

/**
 * @brief Evalute the potential for four particles
 * 
 * @param handle populated handle
 * @param phi first particle
 * @param phj second particle
 * @param phk third particle
 * @param phl fourth particle
 * @param potE evaluated potential energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_evalParts4(struct MxPotentialHandle *handle, 
                                           struct MxParticleHandleHandle *phi, 
                                           struct MxParticleHandleHandle *phj, 
                                           struct MxParticleHandleHandle *phk, 
                                           struct MxParticleHandleHandle *phl, 
                                           float *potE);

/**
 * @brief Evaluate the force
 * 
 * @param handle populated handle
 * @param r distance
 * @param force evaluated force
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_fevalR(struct MxPotentialHandle *handle, float r, float *force);

/**
 * @brief Evaluate the force for a given scaling distance
 * 
 * @param handle populated handle
 * @param r distance
 * @param r0 scaling distance
 * @param force evaluated force
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_fevalR0(struct MxPotentialHandle *handle, float r, float r0, float *force);

/**
 * @brief Evaluate the force for a given position
 * 
 * @param handle populated handle
 * @param pos position
 * @param force evaluated force
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_fevalPos(struct MxPotentialHandle *handle, float *pos, float **force);

/**
 * @brief Evaluate the force for a given particle and position
 * 
 * @param handle populated handle
 * @param partHandle particle
 * @param pos position
 * @param force evaluated force
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_fevalPart(struct MxPotentialHandle *handle, struct MxParticleHandleHandle *partHandle, float *pos, float **force);

/**
 * @brief Evalute the force for two particles
 * 
 * @param handle populated handle
 * @param phi first particle
 * @param phj second particle
 * @param force evaluated force
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_fevalParts2(struct MxPotentialHandle *handle, 
                                            struct MxParticleHandleHandle *phi, 
                                            struct MxParticleHandleHandle *phj, 
                                            float **force);

/**
 * @brief Evalute the forces for three particles
 * 
 * @param handle populated handle
 * @param phi first particle
 * @param phj second particle
 * @param phk third particle
 * @param forcei evaluated force on phi
 * @param forcek evaluated force on phk
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_fevalParts3(struct MxPotentialHandle *handle, 
                                            struct MxParticleHandleHandle *phi, 
                                            struct MxParticleHandleHandle *phj, 
                                            struct MxParticleHandleHandle *phk, 
                                            float **forcei, 
                                            float **forcek);

/**
 * @brief Evalute the forces for four particles
 * 
 * @param handle populated handle
 * @param phi first particle
 * @param phj second particle
 * @param phk third particle
 * @param phl fourth particle
 * @param forcei evaluated force on phi
 * @param forcel evaluated force on phl
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_fevalParts4(struct MxPotentialHandle *handle, 
                                            struct MxParticleHandleHandle *phi, 
                                            struct MxParticleHandleHandle *phj, 
                                            struct MxParticleHandleHandle *phk, 
                                            struct MxParticleHandleHandle *phl, 
                                            float **forcei, 
                                            float **forcel);

/**
 * @brief Get the constituent potentials
 * 
 * @param handle populated handle
 * @param chandles constituent potentials
 * @param numPots number of constituent potentials
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_getConstituents(struct MxPotentialHandle *handle, struct MxPotentialHandle ***chandles, unsigned int *numPots);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 */
CAPI_FUNC(HRESULT) MxCPotential_toString(struct MxPotentialHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCPotential_fromString(struct MxPotentialHandle *handle, const char *str);

/**
 * @brief Set the clear function
 * 
 * @param handle populated handle
 * @param fcn clear function
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_setClearFcn(struct MxPotentialHandle *handle, struct MxPotentialClearHandle *fcn);

/**
 * @brief Remove the clear function
 * 
 * @param handle populate handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_removeClearFcn(struct MxPotentialHandle *handle);

/**
 * @brief Test whether a potential has a clear function
 * 
 * @param handle populate handle
 * @param hasClear flag signifying whether a potential has a clear function
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_hasClearFcn(struct MxPotentialHandle *handle, bool *hasClear);

/**
 * @brief Get the minimum distance for which a potential is evaluated
 * 
 * @param handle populated handle
 * @param minR minimum distance
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_getMin(struct MxPotentialHandle *handle, float *minR);

/**
 * @brief Get the maximum distance for which a potential is evaluated
 * 
 * @param handle populated handle
 * @param maxR maximum distance
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_getMax(struct MxPotentialHandle *handle, float *maxR);

/**
 * @brief Get whether the potential is bound
 * 
 * @param handle populated handle
 * @param bound flag signifying whether the potential is bound
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_getBound(struct MxPotentialHandle *handle, bool *bound);

/**
 * @brief Set whether the potential is bound
 * 
 * @param handle populated handle
 * @param bound flag signifying whether the potential is bound
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_setBound(struct MxPotentialHandle *handle, bool bound);

/**
 * @brief Get the equilibrium distance of the potential
 * 
 * @param handle populated handle
 * @param r0 equilibrium distance
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_getR0(struct MxPotentialHandle *handle, float *r0);

/**
 * @brief Set the equilibrium distance of the potential
 * 
 * @param handle populated handle
 * @param r0 equilibrium distance
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_setR0(struct MxPotentialHandle *handle, float r0);

/**
 * @brief Get the equilibrium distance of the potential
 * 
 * @param handle populated handle
 * @param r2 squared equilibrium distance
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_getRSquare(struct MxPotentialHandle *handle, float *r2);

/**
 * @brief Set the equilibrium distance of the potential
 * 
 * @param handle populated handle
 * @param r2 squared equilibrium distance
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_setRSquare(struct MxPotentialHandle *handle, float r2);

/**
 * @brief Get whether the potential is shifted
 * 
 * @param handle populated handle
 * @param shifted flag signifying whether the potential is shifted
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_getShifted(struct MxPotentialHandle *handle, bool *shifted);

/**
 * @brief Set whether the potential is shifted
 * 
 * @param handle populated handle
 * @param shifted flag signifying whether the potential is shifted
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_setShifted(struct MxPotentialHandle *handle, bool shifted);

/**
 * @brief Get whether the potential is periodic
 * 
 * @param handle populated handle
 * @param periodic flag signifying whether the potential is periodic
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_getPeriodic(struct MxPotentialHandle *handle, bool *periodic);

/**
 * @brief Creates a 12-6 Lennard-Jones potential. 
 * 
 * The Lennard Jones potential has the form:
 * 
 * @f[
 * 
 *      \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) 
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param min The smallest radius for which the potential will be constructed.
 * @param max The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001 * (max - min). 
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_lennard_jones_12_6(struct MxPotentialHandle *handle, double min, double max, double A, double B, double *tol);

/**
 * @brief Creates a potential of the sum of a 12-6 Lennard-Jones potential and a shifted Coulomb potential. 
 * 
 * The 12-6 Lennard Jones - Coulomb potential has the form:
 * 
 * @f[
 * 
 *      \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) + q \left( \frac{1}{r} - \frac{1}{max} \right)
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param min The smallest radius for which the potential will be constructed.
 * @param max The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param q The charge scaling of the potential.
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001 * (max - min). 
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_lennard_jones_12_6_coulomb(struct MxPotentialHandle *handle, double min, double max, double A, double B, double q, double *tol);

/**
 * @brief Creates a soft sphere interaction potential. 
 * 
 * The soft sphere is a generalized Lennard-Jones-type potential, but with settable exponents to create a softer interaction.
 * 
 * @param handle Handle to populate
 * @param kappa 
 * @param epsilon 
 * @param r0 
 * @param eta 
 * @param min 
 * @param max 
 * @param tol 
 * @param shift 
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_soft_sphere(struct MxPotentialHandle *handle, double kappa, double epsilon, double r0, int eta, double *min, double *max, double *tol, bool *shift);

/**
 * @brief Creates a real-space Ewald potential. 
 * 
 * The Ewald potential has the form:
 * 
 * @f[
 * 
 *      q \frac{\mathrm{erfc}\, ( \kappa r)}{r}
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param min The smallest radius for which the potential will be constructed.
 * @param max The largest radius for which the potential will be constructed.
 * @param q The charge scaling of the potential.
 * @param kappa The screening distance of the Ewald potential.
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001 * (max - min). 
 * @param periodicOrder Order of lattice periodicity along all periodic dimensions. Defaults to 0. 
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_ewald(struct MxPotentialHandle *handle, double min, double max, double q, double kappa, double *tol, unsigned int *periodicOrder);

/**
 * @brief Creates a Coulomb potential. 
 * 
 * The Coulomb potential has the form:
 * 
 * @f[
 * 
 *      \frac{q}{r}
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param q The charge scaling of the potential. 
 * @param min The smallest radius for which the potential will be constructed. Default is 0.01. 
 * @param max The largest radius for which the potential will be constructed. Default is 2.0. 
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001 * (max - min). 
 * @param periodicOrder Order of lattice periodicity along all periodic dimensions. Defaults to 0. 
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_coulomb(struct MxPotentialHandle *handle, double q, double *min, double *max, double *tol, unsigned int *periodicOrder);

/**
 * @brief Creates a Coulomb reciprocal potential. 
 * 
 * The Coulomb reciprocal potential has the form: 
 * 
 * @f[
 * 
 *      \frac{\pi q}{V} \sum_{||\mathbf{m}|| \neq 0} \frac{1}{||\mathbf{m}||^2} \exp \left( \left( i \mathbf{r}_{jk} - \left( \frac{\pi}{\kappa} \right)^{2} \mathbf{m} \right) \cdot \mathbf{m} \right)
 * 
 * @f]
 * 
 * Here @f$ V @f$ is the volume of the domain and @f$ \mathbf{m} @f$ is a reciprocal vector of the domain. 
 * 
 * @param handle Handle to populate
 * @param q Charge scaling of the potential. 
 * @param kappa Screening distance.
 * @param min Smallest radius for which the potential will be constructed. 
 * @param max Largest radius for which the potential will be constructed. 
 * @param modes Number of Fourier modes along each periodic dimension. Default is 1. 
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_coulombR(struct MxPotentialHandle *handle, double q, double kappa, double min, double max, unsigned int* modes);

/**
 * @brief Creates a harmonic bond potential. 
 * 
 * The harmonic potential has the form: 
 * 
 * @f[
 * 
 *      k \left( r-r_0 \right)^2
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param k The energy of the bond.
 * @param r0 The bond rest length.
 * @param min The smallest radius for which the potential will be constructed. Defaults to @f$ r_0 - r_0 / 2 @f$.
 * @param max The largest radius for which the potential will be constructed. Defaults to @f$ r_0 + r_0 /2 @f$.
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to @f$ 0.01 \abs(max-min) @f$.
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_harmonic(struct MxPotentialHandle *handle, double k, double r0, double *min, double *max, double *tol);

/**
 * @brief Creates a linear potential. 
 * 
 * The linear potential has the form:
 * 
 * @f[
 * 
 *      k r
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param k interaction strength; represents the potential energy peak value.
 * @param min The smallest radius for which the potential will be constructed. Defaults to 0.0.
 * @param max The largest radius for which the potential will be constructed. Defaults to 10.0.
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001.
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_linear(struct MxPotentialHandle *handle, double k, double *min, double *max, double *tol);

/**
 * @brief Creates a harmonic angle potential. 
 * 
 * The harmonic angle potential has the form: 
 * 
 * @f[
 * 
 *      k \left(\theta-\theta_{0} \right)^2
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param k The energy of the angle.
 * @param theta0 The minimum energy angle.
 * @param min The smallest angle for which the potential will be constructed. Defaults to zero. 
 * @param max The largest angle for which the potential will be constructed. Defaults to PI. 
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.005 * (max - min). 
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_harmonic_angle(struct MxPotentialHandle *handle, double k, double theta0, double *min, double *max, double *tol);

/**
 * @brief Creates a harmonic dihedral potential. 
 * 
 * The harmonic dihedral potential has the form:
 * 
 * @f[
 * 
 *      k \left( \theta - \delta \right) ^2
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param k energy of the dihedral.
 * @param delta minimum energy dihedral. 
 * @param min The smallest angle for which the potential will be constructed. Defaults to zero. 
 * @param max The largest angle for which the potential will be constructed. Defaults to PI. 
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.005 * (max - min). 
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_harmonic_dihedral(struct MxPotentialHandle *handle, double k, double delta, double *min, double *max, double *tol);

/**
 * @brief Creates a cosine dihedral potential. 
 * 
 * The cosine dihedral potential has the form:
 * 
 * @f[
 * 
 *      k \left( 1 + \cos( n \theta-\delta ) \right)
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param k energy of the dihedral.
 * @param n multiplicity of the dihedral.
 * @param delta minimum energy dihedral. 
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.01. 
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_cosine_dihedral(struct MxPotentialHandle *handle, double k, int n, double delta, double *tol);

/**
 * @brief Creates a well potential. 
 * 
 * Useful for binding a particle to a region.
 * 
 * The well potential has the form: 
 * 
 * @f[
 * 
 *      \frac{k}{\left(r_0 - r\right)^{n}}
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param k potential prefactor constant, should be decreased for larger n.
 * @param n exponent of the potential, larger n makes a sharper potential.
 * @param r0 The extents of the potential, length units. Represents the maximum extents that a two objects connected with this potential should come apart.
 * @param min The smallest radius for which the potential will be constructed. Defaults to zero.
 * @param max The largest radius for which the potential will be constructed. Defaults to r0.
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.01 * abs(min-max).
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_well(struct MxPotentialHandle *handle, double k, double n, double r0, double *min, double *max, double *tol);

/**
 * @brief Creates a generalized Lennard-Jones potential.
 * 
 * The generalized Lennard-Jones potential has the form:
 * 
 * @f[
 * 
 *      \frac{\epsilon}{n-m} \left[ m \left( \frac{r_0}{r} \right)^n - n \left( \frac{r_0}{r} \right)^m \right]
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param e effective energy of the potential. 
 * @param m order of potential. Defaults to 3
 * @param n order of potential. Defaults to 2*m.
 * @param k mimumum of the potential. Defaults to 1.
 * @param r0 mimumum of the potential. Defaults to 1. 
 * @param min The smallest radius for which the potential will be constructed. Defaults to 0.05 * r0.
 * @param max The largest radius for which the potential will be constructed. Defaults to 5 * r0.
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.01.
 * @param shifted Flag for whether using a shifted potential. Defaults to true. 
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_glj(struct MxPotentialHandle *handle, double e, double *m, double *n, double *k, double *r0, double *min, double *max, double *tol, bool *shifted);

/**
 * @brief Creates a Morse potential. 
 * 
 * The Morse potential has the form:
 * 
 * @f[
 * 
 *      d \left(1 - e^{ -a \left(r - r_0 \right) } \right)
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param d well depth. Defaults to 1.0.
 * @param a potential width. Defaults to 6.0.
 * @param r0 equilibrium distance. Defaults to 0.0. 
 * @param min The smallest radius for which the potential will be constructed. Defaults to 0.0001.
 * @param max The largest radius for which the potential will be constructed. Defaults to 3.0.
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001.
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_morse(struct MxPotentialHandle *handle, double *d, double *a, double *r0, double *min, double *max, double *tol);

/**
 * @brief Creates an overlapping-sphere potential from :cite:`Osborne:2017hk`. 
 * 
 * The overlapping-sphere potential has the form: 
 * 
 * @f[
 *      \mu_{ij} s_{ij}(t) \hat{\mathbf{r}}_{ij} \log \left( 1 + \frac{||\mathbf{r}_{ij}|| - s_{ij}(t)}{s_{ij}(t)} \right) 
 *          \text{ if } ||\mathbf{r}_{ij}|| < s_{ij}(t) ,
 * @f]
 * 
 * @f[
 *      \mu_{ij}\left(||\mathbf{r}_{ij}|| - s_{ij}(t)\right) \hat{\mathbf{r}}_{ij} \exp \left( -k_c \frac{||\mathbf{r}_{ij}|| - s_{ij}(t)}{s_{ij}(t)} \right) 
 *          \text{ if } s_{ij}(t) \leq ||\mathbf{r}_{ij}|| \leq r_{max} ,
 * @f]
 * 
 * @f[
 *      0 \text{ otherwise} .
 * @f]
 * 
 * Osborne refers to @f$ \mu_{ij} @f$ as a "spring constant", this 
 * controls the size of the force, and is the potential energy peak value. 
 * @f$ \hat{\mathbf{r}}_{ij} @f$ is the unit vector from particle 
 * @f$ i @f$ center to particle @f$ j @f$ center, @f$ k_C @f$ is a 
 * parameter that defines decay of the attractive force. Larger values of 
 * @f$ k_C @f$ result in a shaper peaked attraction, and thus a shorter 
 * ranged force. @f$ s_{ij}(t) @f$ is the is the sum of the radii of the 
 * two particles.
 * 
 * @param handle Handle to populate
 * @param mu interaction strength, represents the potential energy peak value. Defaults to 1.0.
 * @param kc decay strength of long range attraction. Larger values make a shorter ranged function. Defaults to 1.0.
 * @param kh Optionally add a harmonic long-range attraction, same as :meth:`glj` function. Defaults to 0.0.
 * @param r0 Optional harmonic rest length, only used if `kh` is non-zero. Defaults to 0.0.
 * @param min The smallest radius for which the potential will be constructed. Defaults to 0.001.
 * @param max The largest radius for which the potential will be constructed. Defaults to 10.0.
 * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001.
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_overlapping_sphere(struct MxPotentialHandle *handle, double *mu, double *kc, double *kh, double *r0, double *min, double *max, double *tol);

/**
 * @brief Creates a power potential. 
 * 
 * The power potential the general form of many of the potential 
 * functions, such as :meth:`linear`, etc. power has the form:
 * 
 * @f[
 * 
 *      k (r-r_0)^{\alpha}
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param k interaction strength, represents the potential energy peak value. Defaults to 1
 * @param r0 potential rest length, zero of the potential, defaults to 0.
 * @param alpha Exponent, defaults to 1.
 * @param min minimal value potential is computed for, defaults to r0 / 2.
 * @param max cutoff distance, defaults to 3 * r0.
 * @param tol Tolerance, defaults to 0.01.
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_power(struct MxPotentialHandle *handle, double *k, double *r0, double *alpha, double *min, double *max, double *tol);

/**
 * @brief Creates a Dissipative Particle Dynamics potential. 
 * 
 * The Dissipative Particle Dynamics force has the form: 
 * 
 * @f[
 * 
 *      \mathbf{F}_{ij} = \mathbf{F}^C_{ij} + \mathbf{F}^D_{ij} + \mathbf{F}^R_{ij}
 * 
 * @f]
 * 
 * The conservative force is: 
 * 
 * @f[
 * 
 *      \mathbf{F}^C_{ij} = \alpha \left(1 - \frac{r_{ij}}{r_c}\right) \mathbf{e}_{ij}
 * 
 * @f]
 * 
 * The dissapative force is:
 * 
 * @f[
 * 
 *      \mathbf{F}^D_{ij} = -\gamma \left(1 - \frac{r_{ij}}{r_c}\right)^{2}(\mathbf{e}_{ij} \cdot \mathbf{v}_{ij}) \mathbf{e}_{ij}
 * 
 * @f]
 * 
 * The random force is: 
 * 
 * @f[
 * 
 *      \mathbf{F}^R_{ij} = \sigma \left(1 - \frac{r_{ij}}{r_c}\right) \xi_{ij}\Delta t^{-1/2}\mathbf{e}_{ij}
 * 
 * @f]
 * 
 * @param handle Handle to populate
 * @param alpha interaction strength of the conservative force. Defaults to 1.0. 
 * @param gamma interaction strength of dissapative force. Defaults to 1.0. 
 * @param sigma strength of random force. Defaults to 1.0. 
 * @param cutoff cutoff distance. Defaults to 1.0. 
 * @param shifted Flag for whether using a shifted potential. Defaults to false. 
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_dpd(struct MxPotentialHandle *handle, double *alpha, double *gamma, double *sigma, double *cutoff, bool *shifted);

/**
 * @brief Creates a custom potential. 
 * 
 * @param handle Handle to populate
 * @param min The smallest radius for which the potential will be constructed.
 * @param max The largest radius for which the potential will be constructed.
 * @param f function returning the value of the potential
 * @param fp function returning the value of first derivative of the potential
 * @param f6p function returning the value of sixth derivative of the potential
 * @param tol Tolerance, defaults to 0.001.
 * @return MxPotential* 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_custom(struct MxPotentialHandle *handle, 
                                              double min, 
                                              double max, 
                                              double (*f)(double), 
                                              double (*fp)(double), 
                                              double (*f6p)(double), 
                                              double *tol, 
                                              unsigned int *flags);

/**
 * @brief Create a potential that uses an evaluation function by particle
 * 
 * @param handle Handle to populate
 * @param fcn evaluation function
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_eval_ByParticle(struct MxPotentialHandle *handle, struct MxPotentialEval_ByParticleHandle *fcn);

/**
 * @brief Create a potential that uses an evaluation function by two particles
 * 
 * @param handle Handle to populate
 * @param fcn evaluation function
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_eval_ByParticles(struct MxPotentialHandle *handle, struct MxPotentialEval_ByParticlesHandle *fcn);

/**
 * @brief Create a potential that uses an evaluation function by three particles
 * 
 * @param handle Handle to populate
 * @param fcn evaluation function
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_eval_ByParticles3(struct MxPotentialHandle *handle, struct MxPotentialEval_ByParticles3Handle *fcn);

/**
 * @brief Create a potential that uses an evaluation function by four particles
 * 
 * @param handle Handle to populate
 * @param fcn evaluation function
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_create_eval_ByParticles4(struct MxPotentialHandle *handle, struct MxPotentialEval_ByParticles4Handle *fcn);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Add two potentials
 * 
 * @param handlei first potential
 * @param handlej second potential
 * @param handleSum resulting potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPotential_add(struct MxPotentialHandle *handlei, struct MxPotentialHandle *handlej, struct MxPotentialHandle *handleSum);

#endif // _WRAPS_C_MXCPOTENTIAL_H_