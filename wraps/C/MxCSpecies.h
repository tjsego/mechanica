/**
 * @file MxCSpecies.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxSpecies and associated features
 * @date 2022-04-01
 */

#ifndef _WRAPS_C_MXCSPECIES_H_
#define _WRAPS_C_MXCSPECIES_H_

#include <mx_port.h>

#include "MxCStateVector.h"
#include "MxCParticle.h"

// Handles

/**
 * @brief Handle to a @ref MxSpecies instance
 * 
 */
struct CAPI_EXPORT MxSpeciesHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxSpeciesList instance
 * 
 */
struct CAPI_EXPORT MxSpeciesListHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxSpeciesValue instance
 * 
 */
struct CAPI_EXPORT MxSpeciesValueHandle {
    void *MxObj;
};


///////////////
// MxSpecies //
///////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_init(struct MxSpeciesHandle *handle);

/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param s string constructor
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_initS(struct MxSpeciesHandle *handle, const char *s);

/**
 * @brief Copy an instance
 * 
 * @param source populated handle
 * @param destination handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_copy(struct MxSpeciesHandle *source, struct MxSpeciesHandle *destination);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_destroy(struct MxSpeciesHandle *handle);

/**
 * @brief Get the species id
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getId(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the species id
 * 
 * @param handle populated handle
 * @param sid id string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setId(struct MxSpeciesHandle *handle, const char *sid);

/**
 * @brief Get the species name
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getName(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the species name
 * 
 * @param handle populated handle
 * @param name name string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setName(struct MxSpeciesHandle *handle, const char *name);

/**
 * @brief Get the species type
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array character
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getSpeciesType(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the species type
 * 
 * @param handle populated handle
 * @param sid type string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setSpeciesType(struct MxSpeciesHandle *handle, const char *sid);

/**
 * @brief Get the species compartment
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getCompartment(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the species compartment
 * 
 * @param handle populated handle
 * @param sid compartment string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setCompartment(struct MxSpeciesHandle *handle, const char *sid);

/**
 * @brief Get the initial amount
 * 
 * @param handle populated handle
 * @param value initial amount
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getInitialAmount(struct MxSpeciesHandle *handle, double *value);

/**
 * @brief Set the initial amount
 * 
 * @param handle populated handle
 * @param value initial amount
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setInitialAmount(struct MxSpeciesHandle *handle, double value);

/**
 * @brief Get the initial concentration
 * 
 * @param handle populated handle
 * @param value initial concentration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getInitialConcentration(struct MxSpeciesHandle *handle, double *value);

/**
 * @brief Set the initial concentration
 * 
 * @param handle populated handle
 * @param value initial concentration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setInitialConcentration(struct MxSpeciesHandle *handle, double value);

/**
 * @brief Get the substance units
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getSubstanceUnits(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the substance units
 * 
 * @param handle populated handle
 * @param sid substance units string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setSubstanceUnits(struct MxSpeciesHandle *handle, const char *sid);

/**
 * @brief Get the spatial size units
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getSpatialSizeUnits(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the spatial size units
 * 
 * @param handle populated handle
 * @param sid spatial size units string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setSpatialSizeUnits(struct MxSpeciesHandle *handle, const char *sid);

/**
 * @brief Get the units
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getUnits(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the units
 * 
 * @param handle populated handle
 * @param sname units string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setUnits(struct MxSpeciesHandle *handle, const char *sname);

/**
 * @brief Get whether a species has only substance units
 * 
 * @param handle populated handle
 * @param value flag signifying whether a species has only substance units
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getHasOnlySubstanceUnits(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Set whether a species has only substance units
 * 
 * @param handle populated handle
 * @param value flag signifying whether a species has only substance units
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setHasOnlySubstanceUnits(struct MxSpeciesHandle *handle, bool value);

/**
 * @brief Get whether a species has a boundary condition
 * 
 * @param handle populated handle
 * @param value flag signifying whether a species has a boundary condition
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getBoundaryCondition(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Set whether a species has a boundary condition
 * 
 * @param handle populated handle
 * @param value flag signifying whether a species has a boundary condition
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setBoundaryCondition(struct MxSpeciesHandle *handle, bool value);

/**
 * @brief Get the species charge
 * 
 * @param handle populated handle
 * @param charge species charge
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getCharge(struct MxSpeciesHandle *handle, int *charge);

/**
 * @brief Set the species charge
 * 
 * @param handle populated handle
 * @param charge species charge
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setCharge(struct MxSpeciesHandle *handle, int value);

/**
 * @brief Get whether a species is constnat
 * 
 * @param handle populated handle
 * @param value flag signifying whether a species is constnat
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getConstant(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Set whether a species is constnat
 * 
 * @param handle populated handle
 * @param value flag signifying whether a species is constnat
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setConstant(struct MxSpeciesHandle *handle, int value);

/**
 * @brief Get the species conversion factor
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_getConversionFactor(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the species conversion factor
 * 
 * @param handle populated handle
 * @param sid species conversion factor string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_setConversionFactor(struct MxSpeciesHandle *handle, const char *sid);

/**
 * @brief Test whether the species id is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetId(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species name is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetName(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species type is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetSpeciesType(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species compartment is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetCompartment(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species initial amount is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetInitialAmount(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species initial concentration is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetInitialConcentration(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species substance units are set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetSubstanceUnits(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species spatial size units are set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetSpatialSizeUnits(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species units are set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetUnits(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species charge is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetCharge(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species conversion factor is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetConversionFactor(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species is constant
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetConstant(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species boundary condition is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetBoundaryCondition(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether a species has only substance units
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_isSetHasOnlySubstanceUnits(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Unset the species id
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetId(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species name
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetName(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species constant flag
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetConstant(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species type
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetSpeciesType(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species initial amount
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetInitialAmount(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species initial concentration
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetInitialConcentration(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species substance units
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetSubstanceUnits(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species spatial size units
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetSpatialSizeUnits(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species units
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetUnits(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species charge
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetCharge(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species conversion factor
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetConversionFactor(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species compartment
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetCompartment(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species boundary condition
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetBoundaryCondition(struct MxSpeciesHandle *handle);

/**
 * @brief Unset the species has only substance units flag
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_unsetHasOnlySubstanceUnits(struct MxSpeciesHandle *handle);

/**
 * @brief Test whether a species has required attributes
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_hasRequiredAttributes(struct MxSpeciesHandle *handle, bool *value);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_toString(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpecies_fromString(struct MxSpeciesHandle *handle, const char *str);


///////////////////
// MxSpeciesList //
///////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesList_init(struct MxSpeciesListHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesList_destroy(struct MxSpeciesListHandle *handle);

/**
 * @brief Get a string representation
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesList_getStr(struct MxSpeciesListHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Get the index of a species name
 * 
 * @param handle populated handle
 * @param s species name
 * @param i species index
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesList_indexOf(struct MxSpeciesListHandle *handle, const char *s, unsigned int *i);

/**
 * @brief Get the size of a list
 * 
 * @param handle populated handle
 * @param size size of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesList_getSize(struct MxSpeciesListHandle *handle, unsigned int *size);

/**
 * @brief Get a species by index
 * 
 * @param handle populated handle
 * @param index index of the species
 * @param species species at the given index
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesList_getItem(struct MxSpeciesListHandle *handle, unsigned int index, struct MxSpeciesHandle *species);

/**
 * @brief Get a species by name
 * 
 * @param handle populated handle
 * @param s name of species
 * @param species species at the given index
 * @return MxSpecies* 
 */
CAPI_FUNC(HRESULT) MxCSpeciesList_getItemS(struct MxSpeciesListHandle *handle, const char *s, struct MxSpeciesHandle *species);

/**
 * @brief Insert a species
 * 
 * @param handle populated handle
 * @param species species to insert
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesList_insert(struct MxSpeciesListHandle *handle, struct MxSpeciesHandle *species);

/**
 * @brief Insert a species by name
 * 
 * @param handle populated handle
 * @param s name of the species to insert
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesList_insertS(struct MxSpeciesListHandle *handle, const char *s);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesList_toString(struct MxSpeciesListHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesList_fromString(struct MxSpeciesListHandle *handle, const char *str);


////////////////////
// MxSpeciesValue //
////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param value species value
 * @param state_vector state vector of species
 * @param index species index in state vector
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_init(struct MxSpeciesValueHandle *handle, double value, struct MxStateVectorHandle *state_vector, unsigned int index);

/**
 * @brief Get the value
 * 
 * @param handle populated handle 
 * @param value species value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_getValue(struct MxSpeciesValueHandle *handle, double *value);

/**
 * @brief Set the value
 * 
 * @param handle populated handle 
 * @param value species value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_setValue(struct MxSpeciesValueHandle *handle, double value);

/**
 * @brief Get the state vector
 * 
 * @param handle populated handle 
 * @param state_vector state vector
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_getStateVector(struct MxSpeciesValueHandle *handle, struct MxStateVectorHandle *state_vector);

/**
 * @brief Get the species index
 * 
 * @param handle populated handle 
 * @param index species index
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_getIndex(struct MxSpeciesValueHandle *handle, unsigned int *index);

/**
 * @brief Get the species
 * 
 * @param handle populated handle 
 * @param species handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_getSpecies(struct MxSpeciesValueHandle *handle, struct MxSpeciesHandle *species);

/**
 * @brief Test whether the species has a boundary condition
 * 
 * @param handle populated handle 
 * @param value flag signifying whether the species has a boundary condition
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_getBoundaryCondition(struct MxSpeciesValueHandle *handle, bool *value);

/**
 * @brief Set whether the species has a boundary condition
 * 
 * @param handle populated handle 
 * @param value flag signifying whether the species has a boundary condition
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_setBoundaryCondition(struct MxSpeciesValueHandle *handle, bool value);

/**
 * @brief Get the species initial amount
 * 
 * @param handle populated handle 
 * @param value initial amount
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_getInitialAmount(struct MxSpeciesValueHandle *handle, double *value);

/**
 * @brief Set the species initial amount
 * 
 * @param handle populated handle 
 * @param value initial amount
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_setInitialAmount(struct MxSpeciesValueHandle *handle, double value);

/**
 * @brief Get the species initial concentration
 * 
 * @param handle populated handle 
 * @param value initial concentration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_getInitialConcentration(struct MxSpeciesValueHandle *handle, double *value);

/**
 * @brief Set the species initial concentration
 * 
 * @param handle populated handle 
 * @param value initial concentration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_setInitialConcentration(struct MxSpeciesValueHandle *handle, double value);

/**
 * @brief Test whether the species is constant
 * 
 * @param handle populated handle 
 * @param value flag signifying whether the species is constant
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_getConstant(struct MxSpeciesValueHandle *handle, bool *value);

/**
 * @brief Set whether the species is constant
 * 
 * @param handle populated handle
 * @param value flag signifying whether the species is constant
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_setConstant(struct MxSpeciesValueHandle *handle, int value);

/**
 * @brief Secrete this species into a neighborhood. 
 * 
 * @param handle populated handle
 * @param amount amount to secrete. 
 * @param to list of particles to secrete to. 
 * @param secreted amount actually secreted, accounting for availability and other subtleties. 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_secreteL(struct MxSpeciesValueHandle *handle, double amount, struct MxParticleListHandle *to, double *secreted);

/**
 * @brief Secrete this species into a neighborhood. 
 * 
 * @param handle populated handle
 * @param amount amount to secrete. 
 * @param distance neighborhood distance. 
 * @param secreted amount actually secreted, accounting for availability and other subtleties. 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSpeciesValue_secreteD(struct MxSpeciesValueHandle *handle, double amount, double distance, double *secreted);

#endif // _WRAPS_C_MXCSPECIES_H_