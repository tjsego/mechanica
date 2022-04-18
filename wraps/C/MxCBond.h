/**
 * @file MxCBond.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxBond
 * @date 2022-04-01
 */

#ifndef _WRAPS_C_MXCBOND_H_
#define _WRAPS_C_MXCBOND_H_

#include <mx_port.h>

#include "MxCParticle.h"
#include "MxCPotential.h"

// Handles

/**
 * @brief Handle to a @ref MxBondHandle instance
 * 
 */
struct CAPI_EXPORT MxBondHandleHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxAngleHandle instance
 * 
 */
struct CAPI_EXPORT MxAngleHandleHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxDihedralHandle instance
 * 
 */
struct CAPI_EXPORT MxDihedralHandleHandle {
    void *MxObj;
};


//////////////////
// MxBondHandle //
//////////////////


/**
 * @brief Construct a new bond handle from an existing bond id
 * 
 * @param handle handle to populate
 * @param id id of existing bond
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_init(struct MxBondHandleHandle *handle, unsigned int id);

/**
 * @brief Construct a new bond handle and underlying bond. 
 * 
 * @param handle handle to populate
 * @param potential bond potential
 * @param i ith particle
 * @param j jth particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_create(struct MxBondHandleHandle *handle, 
                                        struct MxPotentialHandle *potential,
                                        struct MxParticleHandleHandle *i, 
                                        struct MxParticleHandleHandle *j);

/**
 * @brief Get the id. Returns -1 if the underlying bond is invalid
 * 
 * @param handle populated handle
 * @param id bond id
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_getId(struct MxBondHandleHandle *handle, int *id);

/**
 * @brief Get a summary string of the bond
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBondHandle_getStr(struct MxBondHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Check the validity of the handle
 * 
 * @param handle populated handle
 * @param flag true if OK
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_check(struct MxBondHandleHandle *handle, bool *flag);

/**
 * @brief Destroy the bond. 
 * 
 * @param populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_destroy(struct MxBondHandleHandle *handle);

/**
 * @brief Tests whether this bond decays
 * 
 * @param handle populated handle
 * @param flag true when the bond should decay
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_decays(struct MxBondHandleHandle *handle, bool *flag);

/**
 * @brief Get the current energy of the bond
 * 
 * @param handle populated handle
 * @param value current energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_getEnergy(struct MxBondHandleHandle *handle, double *value);

/**
 * @brief Get the ids of the particles of the bond
 * 
 * @param handle populated handle
 * @param parti ith particle; -1 if no particle
 * @param partj jth particle; -1 if no particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_getParts(struct MxBondHandleHandle *handle, int *parti, int *partj);

/**
 * @brief Get the potential of the bond
 * 
 * @param handle populated handle
 * @param potential bond potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_getPotential(struct MxBondHandleHandle *handle, struct MxPotentialHandle *potential);

/**
 * @brief Get the dissociation energy
 * 
 * @param handle populated handle
 * @param value dissociation energy; -1 if not defined
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_getDissociationEnergy(struct MxBondHandleHandle *handle, float *value);

/**
 * @brief Set the dissociation energy
 * 
 * @param handle populated handle
 * @param value dissociation energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_setDissociationEnergy(struct MxBondHandleHandle *handle, float value);

/**
 * @brief Get the half life
 * 
 * @param handle populated handle
 * @param value half life; -1 if not defined
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_getHalfLife(struct MxBondHandleHandle *handle, float *value);

/**
 * @brief Set the half life
 * 
 * @param handle populated handle
 * @param value half life
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_setHalfLife(struct MxBondHandleHandle *handle, float value);

/**
 * @brief Test whether a bond is active
 * 
 * @param handle populated handle
 * @param flag true when active
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_getActive(struct MxBondHandleHandle *handle, bool *flag);

/**
 * @brief Set whether a bond is active
 * 
 * @param handle populated handle
 * @param flag true when active
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_setActive(struct MxBondHandleHandle *handle, bool flag);

/**
 * @brief Get the bond style
 * 
 * @param handle populated handle
 * @param style bond style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_getStyle(struct MxBondHandleHandle *handle, struct MxStyleHandle *style);

/**
 * @brief Set the bond style
 * 
 * @param handle populated handle
 * @param style bond style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_setStyle(struct MxBondHandleHandle *handle, struct MxStyleHandle *style);

/**
 * @brief Get the age of the bond
 * 
 * @param handle populated handle
 * @param value bond age
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_getAge(struct MxBondHandleHandle *handle, double *value);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBondHandle_toString(struct MxBondHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBondHandle_fromString(struct MxBondHandleHandle *handle, const char *str);


///////////////////
// MxAngleHandle //
///////////////////


/**
 * @brief Construct a new bond handle from an existing bond id
 * 
 * @param handle handle to populate
 * @param id id of existing bond
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_init(struct MxAngleHandleHandle *handle, unsigned int id);

/**
 * @brief Construct a new bond handle and underlying bond. 
 * 
 * @param handle handle to populate
 * @param potential bond potential
 * @param i ith particle
 * @param j jth particle
 * @param k kth particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_create(struct MxAngleHandleHandle *handle, 
                                         struct MxPotentialHandle *potential,
                                         struct MxParticleHandleHandle *i, 
                                         struct MxParticleHandleHandle *j, 
                                         struct MxParticleHandleHandle *k);

/**
 * @brief Get a summary string of the bond
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_str(struct MxAngleHandleHandle *handle, char **str, unsigned int *numChars);


/**
 * @brief Check the validity of the handle
 * 
 * @param handle populated handle
 * @param flag true if OK
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_check(struct MxAngleHandleHandle *handle, bool *flag);

/**
 * @brief Destroy the angle. 
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_destroy(struct MxAngleHandleHandle *handle);

/**
 * @brief Tests whether this bond decays
 * 
 * @param handle populated handle
 * @param flag true when the bond should decay
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_decays(struct MxAngleHandleHandle, bool *flag);

/**
 * @brief Get the current energy of the bond
 * 
 * @param handle populated handle
 * @param value current energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_getEnergy(struct MxAngleHandleHandle *handle, double *value);

/**
 * @brief Get the ids of the particles of the bond
 * 
 * @param handle populated handle
 * @param parti ith particle; -1 if no particle
 * @param partj jth particle; -1 if no particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_getParts(struct MxAngleHandleHandle *handle, int *parti, int *partj);

/**
 * @brief Get the potential of the bond
 * 
 * @param handle populated handle
 * @param potential bond potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_getPotential(struct MxAngleHandleHandle *handle, struct MxPotentialHandle *potential);

/**
 * @brief Get the dissociation energy
 * 
 * @param handle populated handle
 * @param value dissociation energy; -1 if not defined
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_getDissociationEnergy(struct MxAngleHandleHandle *handle, float *value);

/**
 * @brief Set the dissociation energy
 * 
 * @param handle populated handle
 * @param value dissociation energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_setDissociationEnergy(struct MxAngleHandleHandle *handle, float value);

/**
 * @brief Get the half life
 * 
 * @param handle populated handle
 * @param value half life; -1 if not defined
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_getHalfLife(struct MxAngleHandleHandle *handle, float *value);

/**
 * @brief Set the half life
 * 
 * @param handle populated handle
 * @param value half life
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_setHalfLife(struct MxAngleHandleHandle *handle, float value);

/**
 * @brief Test whether a bond is active
 * 
 * @param handle populated handle
 * @param flag true when active
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_getActive(struct MxAngleHandleHandle *handle, bool *flag);

/**
 * @brief Set whether a bond is active
 * 
 * @param handle populated handle
 * @param flag true when active
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_setActive(struct MxAngleHandleHandle *handle, bool flag);

/**
 * @brief Get the bond style
 * 
 * @param handle populated handle
 * @param style bond style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_getStyle(struct MxAngleHandleHandle *handle, struct MxStyleHandle *style);

/**
 * @brief Set the bond style
 * 
 * @param handle populated handle
 * @param style bond style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_setStyle(struct MxAngleHandleHandle *handle, struct MxStyleHandle *style);

/**
 * @brief Get the age of the bond
 * 
 * @param handle populated handle
 * @param value bond age
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_getAge(struct MxAngleHandleHandle *handle, double *value);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_toString(struct MxAngleHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_fromString(struct MxAngleHandleHandle *handle, const char *str);


//////////////////////
// MxDihedralHandle //
//////////////////////


/**
 * @brief Construct a new bond handle from an existing bond id
 * 
 * @param handle handle to populate
 * @param id id of existing bond
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_init(struct MxDihedralHandleHandle *handle, unsigned int id);

/**
 * @brief Construct a new bond handle and underlying bond. 
 * 
 * @param handle handle to populate
 * @param potential bond potential
 * @param i ith particle
 * @param j jth particle
 * @param k kth particle
 * @param l lth particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_create(struct MxDihedralHandleHandle *handle, 
                                            struct MxPotentialHandle *potential,
                                            struct MxParticleHandleHandle *i, 
                                            struct MxParticleHandleHandle *j, 
                                            struct MxParticleHandleHandle *k, 
                                            struct MxParticleHandleHandle *l);

/**
 * @brief Get a summary string of the bond
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_str(struct MxDihedralHandleHandle *handle, char **str, unsigned int *numChars);


/**
 * @brief Check the validity of the handle
 * 
 * @param handle populated handle
 * @param flag true if OK
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_check(struct MxDihedralHandleHandle *handle, bool *flag);

/**
 * @brief Destroy the angle. 
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_destroy(struct MxDihedralHandleHandle *handle);

/**
 * @brief Tests whether this bond decays
 * 
 * @param handle populated handle
 * @param flag true when the bond should decay
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_decays(struct MxDihedralHandleHandle, bool *flag);

/**
 * @brief Get the current energy of the bond
 * 
 * @param handle populated handle
 * @param value current energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_getEnergy(struct MxDihedralHandleHandle *handle, double *value);

/**
 * @brief Get the ids of the particles of the bond
 * 
 * @param handle populated handle
 * @param parti ith particle; -1 if no particle
 * @param partj jth particle; -1 if no particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_getParts(struct MxDihedralHandleHandle *handle, int *parti, int *partj);

/**
 * @brief Get the potential of the bond
 * 
 * @param handle populated handle
 * @param potential bond potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_getPotential(struct MxDihedralHandleHandle *handle, struct MxPotentialHandle *potential);

/**
 * @brief Get the dissociation energy
 * 
 * @param handle populated handle
 * @param value dissociation energy; -1 if not defined
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_getDissociationEnergy(struct MxDihedralHandleHandle *handle, float *value);

/**
 * @brief Set the dissociation energy
 * 
 * @param handle populated handle
 * @param value dissociation energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_setDissociationEnergy(struct MxDihedralHandleHandle *handle, float value);

/**
 * @brief Get the half life
 * 
 * @param handle populated handle
 * @param value half life; -1 if not defined
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_getHalfLife(struct MxDihedralHandleHandle *handle, float *value);

/**
 * @brief Set the half life
 * 
 * @param handle populated handle
 * @param value half life
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_setHalfLife(struct MxDihedralHandleHandle *handle, float value);

/**
 * @brief Test whether a bond is active
 * 
 * @param handle populated handle
 * @param flag true when active
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_getActive(struct MxDihedralHandleHandle *handle, bool *flag);

/**
 * @brief Set whether a bond is active
 * 
 * @param handle populated handle
 * @param flag true when active
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_setActive(struct MxDihedralHandleHandle *handle, bool flag);

/**
 * @brief Get the bond style
 * 
 * @param handle populated handle
 * @param style bond style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_getStyle(struct MxDihedralHandleHandle *handle, struct MxStyleHandle *style);

/**
 * @brief Set the bond style
 * 
 * @param handle populated handle
 * @param style bond style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_setStyle(struct MxDihedralHandleHandle *handle, struct MxStyleHandle *style);

/**
 * @brief Get the age of the bond
 * 
 * @param handle populated handle
 * @param value bond age
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_getAge(struct MxDihedralHandleHandle *handle, double *value);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_toString(struct MxDihedralHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_fromString(struct MxDihedralHandleHandle *handle, const char *str);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Gets all bonds in the universe
 * 
 * @param handles
 * @param numBonds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondHandle_getAll(struct MxBondHandleHandle **handles, unsigned int *numBonds);

/**
 * @brief Apply bonds to a list of particles. 
 * 
 * @param pot the potential of the created bonds
 * @param parts list of particles
 * @param cutoff cutoff distance of particles that are bonded
 * @param ppairsA first elements of type pairs that are bonded
 * @param ppairsB second elements of type pairs that are bonded
 * @param numTypePairs number of type pairs
 * @param half_life bond half life, optional
 * @param bond_energy bond energy, optional
 * @param bonds created bonds
 * @param numBonds number of created bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBond_pairwise(struct MxPotentialHandle *pot, 
                                    struct MxParticleListHandle *parts, 
                                    double cutoff, 
                                    struct MxParticleTypeHandle *ppairsA, 
                                    struct MxParticleTypeHandle *ppairsB, 
                                    unsigned int numTypePairs, 
                                    double *half_life, 
                                    double *bond_energy, 
                                    struct MxBondHandleHandle **bonds, 
                                    unsigned int *numBonds);

/**
 * @brief Find all the bonds that interact with the given particle id
 * 
 * @param pid particle id
 * @param bids bond ids
 * @param numIds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBond_getIdsForParticle(unsigned int pid, unsigned int **bids, unsigned int *numIds);

/**
 * @brief Deletes all bonds in the universe. 
 * 
 * Automatically updates when running on a CUDA device. 
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBond_destroyAll();

/**
 * @brief Gets all bonds in the universe
 * 
 * @param handles
 * @param numBonds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleHandle_getAll(struct MxAngleHandleHandle **handles, unsigned int *numBonds);

/**
 * @brief Find all the bonds that interact with the given particle id
 * 
 * @param pid particle id
 * @param bids bond ids
 * @param numIds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngle_getIdsForParticle(unsigned int pid, unsigned int **bids, unsigned int *numIds);

/**
 * @brief Deletes all bonds in the universe. 
 * 
 * Automatically updates when running on a CUDA device. 
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngle_destroyAll();

/**
 * @brief Gets all bonds in the universe
 * 
 * @param handles
 * @param numBonds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedralHandle_getAll(struct MxDihedralHandleHandle **handles, unsigned int *numBonds);

/**
 * @brief Find all the bonds that interact with the given particle id
 * 
 * @param pid particle id
 * @param bids bond ids
 * @param numIds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedral_getIdsForParticle(unsigned int pid, unsigned int **bids, unsigned int *numIds);

/**
 * @brief Deletes all bonds in the universe. 
 * 
 * Automatically updates when running on a CUDA device. 
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCDihedral_destroyAll();

#endif // _WRAPS_C_MXCBOND_H_