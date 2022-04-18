/**
 * @file MxCSpecies.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxSpecies and associated features
 * @date 2022-04-01
 */

#include "MxCSpecies.h"

#include "mechanica_c_private.h"

#include <state/MxSpecies.h>
#include <state/MxSpeciesList.h>
#include <state/MxSpeciesValue.h>
#include <state/MxStateVector.h>
#include <MxParticleList.hpp>

namespace mx { 

MxSpecies *castC(struct MxSpeciesHandle *handle) {
    return castC<MxSpecies, MxSpeciesHandle>(handle);
}

MxSpeciesList *castC(struct MxSpeciesListHandle *handle) {
    return castC<MxSpeciesList, MxSpeciesListHandle>(handle);
}

MxSpeciesValue *castC(struct MxSpeciesValueHandle *handle) {
    return castC<MxSpeciesValue, MxSpeciesValueHandle>(handle);
}

}

#define MXSPECIES_GET(handle, varname) \
    MxSpecies *varname = mx::castC<MxSpecies, MxSpeciesHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXSPECIESLIST_GET(handle, varname) \
    MxSpeciesList *varname = mx::castC<MxSpeciesList, MxSpeciesListHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXSPECIESVALUE_GET(handle, varname) \
    MxSpeciesValue *varname = mx::castC<MxSpeciesValue, MxSpeciesValueHandle>(handle); \
    MXCPTRCHECK(varname);


///////////////
// MxSpecies //
///////////////


HRESULT MxCSpecies_init(struct MxSpeciesHandle *handle) {
    MXCPTRCHECK(handle);
    MxSpecies *species = new MxSpecies();
    handle->MxObj = (void*)species;
    return S_OK;
}

HRESULT MxCSpecies_initS(struct MxSpeciesHandle *handle, const char *s) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(s);
    MxSpecies *species = new MxSpecies(s);
    MXCPTRCHECK(species);
    handle->MxObj = (void*)species;
    return S_OK;
}

HRESULT MxCSpecies_copy(struct MxSpeciesHandle *source, struct MxSpeciesHandle *destination) {
    MXSPECIES_GET(source, srcSpecies);
    MXCPTRCHECK(destination);
    MxSpecies *dstSpecies = new MxSpecies(*srcSpecies);
    MXCPTRCHECK(dstSpecies);
    destination->MxObj = (void*)dstSpecies;
    return S_OK;
}

HRESULT MxCSpecies_destroy(struct MxSpeciesHandle *handle) {
    return mx::capi::destroyHandle<MxSpecies, MxSpeciesHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCSpecies_getId(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars) {
    MXSPECIES_GET(handle, species);
    return mx::capi::str2Char(species->getId(), str, numChars);
}

HRESULT MxCSpecies_setId(struct MxSpeciesHandle *handle, const char *sid) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(sid);
    return species->setId(sid);
}

HRESULT MxCSpecies_getName(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars) {
    MXSPECIES_GET(handle, species);
    return mx::capi::str2Char(species->getName(), str, numChars);
}

HRESULT MxCSpecies_setName(struct MxSpeciesHandle *handle, const char *name) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(name);
    return species->setName(name);
}

HRESULT MxCSpecies_getSpeciesType(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars) {
    MXSPECIES_GET(handle, species);
    return mx::capi::str2Char(species->getSpeciesType(), str, numChars);
}

HRESULT MxCSpecies_setSpeciesType(struct MxSpeciesHandle *handle, const char *sid) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(sid);
    return species->setSpeciesType(sid);
}

HRESULT MxCSpecies_getCompartment(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars) {
    MXSPECIES_GET(handle, species);
    return mx::capi::str2Char(species->getCompartment(), str, numChars);
}

HRESULT MxCSpecies_setCompartment(struct MxSpeciesHandle *handle, const char *sid) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(sid);
    return species->setCompartment(sid);
}

HRESULT MxCSpecies_getInitialAmount(struct MxSpeciesHandle *handle, double *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->getInitialAmount();
    return S_OK;
}

HRESULT MxCSpecies_setInitialAmount(struct MxSpeciesHandle *handle, double value) {
    MXSPECIES_GET(handle, species);
    return species->setInitialAmount(value);
}

HRESULT MxCSpecies_getInitialConcentration(struct MxSpeciesHandle *handle, double *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->getInitialConcentration();
    return S_OK;
}

HRESULT MxCSpecies_setInitialConcentration(struct MxSpeciesHandle *handle, double value) {
    MXSPECIES_GET(handle, species);
    return species->setInitialConcentration(value);
}

HRESULT MxCSpecies_getSubstanceUnits(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars) {
    MXSPECIES_GET(handle, species);
    return mx::capi::str2Char(species->getSubstanceUnits(), str, numChars);
}

HRESULT MxCSpecies_setSubstanceUnits(struct MxSpeciesHandle *handle, const char *sid) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(sid);
    return species->setSubstanceUnits(sid);
}

HRESULT MxCSpecies_getSpatialSizeUnits(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars) {
    MXSPECIES_GET(handle, species);
    return mx::capi::str2Char(species->getSpatialSizeUnits(), str, numChars);
}

HRESULT MxCSpecies_setSpatialSizeUnits(struct MxSpeciesHandle *handle, const char *sid) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(sid);
    return species->setSpatialSizeUnits(sid);
}

HRESULT MxCSpecies_getUnits(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars) {
    MXSPECIES_GET(handle, species);
    return mx::capi::str2Char(species->getUnits(), str, numChars);
}

HRESULT MxCSpecies_setUnits(struct MxSpeciesHandle *handle, const char *sname) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(sname);
    return species->setUnits(sname);
}

HRESULT MxCSpecies_getHasOnlySubstanceUnits(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->getHasOnlySubstanceUnits();
    return S_OK;
}

HRESULT MxCSpecies_setHasOnlySubstanceUnits(struct MxSpeciesHandle *handle, bool value) {
    MXSPECIES_GET(handle, species);
    return species->setHasOnlySubstanceUnits(value);
}

HRESULT MxCSpecies_getBoundaryCondition(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->getBoundaryCondition();
    return S_OK;
}

HRESULT MxCSpecies_setBoundaryCondition(struct MxSpeciesHandle *handle, bool value) {
    MXSPECIES_GET(handle, species);
    return species->setBoundaryCondition(value);
}

HRESULT MxCSpecies_getCharge(struct MxSpeciesHandle *handle, int *charge) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(charge);
    *charge = species->getCharge();
    return S_OK;
}

HRESULT MxCSpecies_setCharge(struct MxSpeciesHandle *handle, int value) {
    MXSPECIES_GET(handle, species);
    return species->setCharge(value);
}

HRESULT MxCSpecies_getConstant(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->getConstant();
    return S_OK;
}

HRESULT MxCSpecies_setConstant(struct MxSpeciesHandle *handle, int value) {
    MXSPECIES_GET(handle, species);
    return species->setConstant(value);
}

HRESULT MxCSpecies_getConversionFactor(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars) {
    MXSPECIES_GET(handle, species);
    return mx::capi::str2Char(species->getConversionFactor(), str, numChars);
}

HRESULT MxCSpecies_setConversionFactor(struct MxSpeciesHandle *handle, const char *sid) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(sid);
    return species->setConversionFactor(sid);
}

HRESULT MxCSpecies_isSetId(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetId();
    return S_OK;
}

HRESULT MxCSpecies_isSetName(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetName();
    return S_OK;
}

HRESULT MxCSpecies_isSetSpeciesType(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetSpeciesType();
    return S_OK;
}

HRESULT MxCSpecies_isSetCompartment(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetCompartment();
    return S_OK;
}

HRESULT MxCSpecies_isSetInitialAmount(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetInitialAmount();
    return S_OK;
}

HRESULT MxCSpecies_isSetInitialConcentration(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetInitialConcentration();
    return S_OK;
}

HRESULT MxCSpecies_isSetSubstanceUnits(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetSubstanceUnits();
    return S_OK;
}

HRESULT MxCSpecies_isSetSpatialSizeUnits(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetSpatialSizeUnits();
    return S_OK;
}

HRESULT MxCSpecies_isSetUnits(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetUnits();
    return S_OK;
}

HRESULT MxCSpecies_isSetCharge(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetCharge();
    return S_OK;
}

HRESULT MxCSpecies_isSetConversionFactor(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetConversionFactor();
    return S_OK;
}

HRESULT MxCSpecies_isSetConstant(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetConstant();
    return S_OK;
}

HRESULT MxCSpecies_isSetBoundaryCondition(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetBoundaryCondition();
    return S_OK;
}

HRESULT MxCSpecies_isSetHasOnlySubstanceUnits(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->isSetHasOnlySubstanceUnits();
    return S_OK;
}

HRESULT MxCSpecies_unsetId(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    species->unsetId();
    return S_OK;
}

HRESULT MxCSpecies_unsetName(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetName();
}

HRESULT MxCSpecies_unsetConstant(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetConstant();
}

HRESULT MxCSpecies_unsetSpeciesType(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetSpeciesType();
}

HRESULT MxCSpecies_unsetInitialAmount(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetInitialAmount();
}

HRESULT MxCSpecies_unsetInitialConcentration(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetInitialConcentration();
}

HRESULT MxCSpecies_unsetSubstanceUnits(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetSubstanceUnits();
}

HRESULT MxCSpecies_unsetSpatialSizeUnits(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetSpatialSizeUnits();
}

HRESULT MxCSpecies_unsetUnits(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetUnits();
}

HRESULT MxCSpecies_unsetCharge(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetCharge();
}

HRESULT MxCSpecies_unsetConversionFactor(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetConversionFactor();
}

HRESULT MxCSpecies_unsetCompartment(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetCompartment();
}

HRESULT MxCSpecies_unsetBoundaryCondition(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetBoundaryCondition();
}

HRESULT MxCSpecies_unsetHasOnlySubstanceUnits(struct MxSpeciesHandle *handle) {
    MXSPECIES_GET(handle, species);
    return species->unsetHasOnlySubstanceUnits();
}

HRESULT MxCSpecies_hasRequiredAttributes(struct MxSpeciesHandle *handle, bool *value) {
    MXSPECIES_GET(handle, species);
    MXCPTRCHECK(value);
    *value = species->hasRequiredAttributes();
    return S_OK;
}

HRESULT MxCSpecies_toString(struct MxSpeciesHandle *handle, char **str, unsigned int *numChars) {
    MXSPECIES_GET(handle, species);
    return mx::capi::str2Char(species->toString(), str, numChars);
}

HRESULT MxCSpecies_fromString(struct MxSpeciesHandle *handle, const char *str) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(str);
    MxSpecies *species = MxSpecies::fromString(str);
    MXCPTRCHECK(species);
    handle->MxObj = (void*)species;
    return S_OK;
}


///////////////////
// MxSpeciesList //
///////////////////


HRESULT MxCSpeciesList_init(struct MxSpeciesListHandle *handle) {
    MXCPTRCHECK(handle);
    MxSpeciesList *slist = new MxSpeciesList();
    handle->MxObj = (void*)slist;
    return S_OK;
}

HRESULT MxCSpeciesList_destroy(struct MxSpeciesListHandle *handle) {
    return mx::capi::destroyHandle<MxSpeciesList, MxSpeciesListHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCSpeciesList_getStr(struct MxSpeciesListHandle *handle, char **str, unsigned int *numChars) {
    MXSPECIESLIST_GET(handle, slist);
    return mx::capi::str2Char(slist->str(), str, numChars);
}

HRESULT MxCSpeciesList_indexOf(struct MxSpeciesListHandle *handle, const char *s, unsigned int *i) {
    MXSPECIESLIST_GET(handle, slist);
    MXCPTRCHECK(s);
    MXCPTRCHECK(i);
    int _i = slist->index_of(s);
    if(_i < 0) 
        return E_FAIL;
    *i = _i;
    return S_OK;
}

HRESULT MxCSpeciesList_getSize(struct MxSpeciesListHandle *handle, unsigned int *size) {
    MXSPECIESLIST_GET(handle, slist);
    MXCPTRCHECK(size);
    *size = slist->size();
    return S_OK;
}

HRESULT MxCSpeciesList_getItem(struct MxSpeciesListHandle *handle, unsigned int index, struct MxSpeciesHandle *species) {
    MXSPECIESLIST_GET(handle, slist);
    MXCPTRCHECK(species);
    MxSpecies *_species = slist->item(index);
    MXCPTRCHECK(_species);
    species->MxObj = (void*)_species;
    return S_OK;
}

HRESULT MxCSpeciesList_getItemS(struct MxSpeciesListHandle *handle, const char *s, struct MxSpeciesHandle *species) {
    MXSPECIESLIST_GET(handle, slist);
    MXCPTRCHECK(species);
    MxSpecies *_species = slist->item(s);
    MXCPTRCHECK(_species);
    species->MxObj = (void*)_species;
    return S_OK;
}

HRESULT MxCSpeciesList_insert(struct MxSpeciesListHandle *handle, struct MxSpeciesHandle *species) {
    MXSPECIESLIST_GET(handle, slist);
    MXSPECIES_GET(species, _species);
    return slist->insert(_species);
}

HRESULT MxCSpeciesList_insertS(struct MxSpeciesListHandle *handle, const char *s) {
    MXSPECIESLIST_GET(handle, slist);
    MXCPTRCHECK(s);
    return slist->insert(s);
}

HRESULT MxCSpeciesList_toString(struct MxSpeciesListHandle *handle, char **str, unsigned int *numChars) {
    MXSPECIESLIST_GET(handle, slist);
    return mx::capi::str2Char(slist->toString(), str, numChars);
}

HRESULT MxCSpeciesList_fromString(struct MxSpeciesListHandle *handle, const char *str) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(str);
    MxSpeciesList *slist = MxSpeciesList::fromString(str);
    MXCPTRCHECK(slist);
    handle->MxObj = (void*)slist;
    return S_OK;
}


////////////////////
// MxSpeciesValue //
////////////////////


HRESULT MxCSpeciesValue_init(struct MxSpeciesValueHandle *handle, double value, struct MxStateVectorHandle *state_vector, unsigned int index) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(state_vector); MXCPTRCHECK(state_vector->MxObj);
    MxSpeciesValue *sval = new MxSpeciesValue(value, (MxStateVector*)state_vector->MxObj, index);
    handle->MxObj = (void*)sval;
    return S_OK;
}

HRESULT MxCSpeciesValue_getValue(struct MxSpeciesValueHandle *handle, double *value) {
    MXSPECIESVALUE_GET(handle, sval);
    MXCPTRCHECK(value);
    *value = sval->value;
    return S_OK;
}

HRESULT MxCSpeciesValue_setValue(struct MxSpeciesValueHandle *handle, double value) {
    MXSPECIESVALUE_GET(handle, sval);
    sval->value = value;
    return S_OK;
}

HRESULT MxCSpeciesValue_getStateVector(struct MxSpeciesValueHandle *handle, struct MxStateVectorHandle *state_vector) {
    MXSPECIESVALUE_GET(handle, sval);
    MXCPTRCHECK(state_vector);
    MXCPTRCHECK(sval->state_vector);
    state_vector->MxObj = (void*)sval->state_vector;
    return S_OK;
}

HRESULT MxCSpeciesValue_getIndex(struct MxSpeciesValueHandle *handle, unsigned int *index) {
    MXSPECIESVALUE_GET(handle, sval);
    MXCPTRCHECK(index);
    *index = sval->index;
    return S_OK;
}

HRESULT MxCSpeciesValue_getSpecies(struct MxSpeciesValueHandle *handle, struct MxSpeciesHandle *species) {
    MXSPECIESVALUE_GET(handle, sval);
    MXCPTRCHECK(species);
    MxSpecies *_species = sval->species();
    MXCPTRCHECK(_species);
    species->MxObj = (void*)_species;
    return S_OK;
}

HRESULT MxCSpeciesValue_getBoundaryCondition(struct MxSpeciesValueHandle *handle, bool *value) {
    MXSPECIESVALUE_GET(handle, sval);
    MXCPTRCHECK(value);
    *value = sval->getBoundaryCondition();
    return S_OK;
}

HRESULT MxCSpeciesValue_setBoundaryCondition(struct MxSpeciesValueHandle *handle, bool value) {
    MXSPECIESVALUE_GET(handle, sval);
    return sval->setBoundaryCondition(value);
}

HRESULT MxCSpeciesValue_getInitialAmount(struct MxSpeciesValueHandle *handle, double *value) {
    MXSPECIESVALUE_GET(handle, sval);
    MXCPTRCHECK(value);
    *value = sval->getInitialAmount();
    return S_OK;
}

HRESULT MxCSpeciesValue_setInitialAmount(struct MxSpeciesValueHandle *handle, double value) {
    MXSPECIESVALUE_GET(handle, sval);
    return sval->setInitialAmount(value);
}

HRESULT MxCSpeciesValue_getInitialConcentration(struct MxSpeciesValueHandle *handle, double *value) {
    MXSPECIESVALUE_GET(handle, sval);
    MXCPTRCHECK(value);
    *value = sval->getInitialConcentration();
    return S_OK;
}

HRESULT MxCSpeciesValue_setInitialConcentration(struct MxSpeciesValueHandle *handle, double value) {
    MXSPECIESVALUE_GET(handle, sval);
    return sval->setInitialConcentration(value);
}

HRESULT MxCSpeciesValue_getConstant(struct MxSpeciesValueHandle *handle, bool *value) {
    MXSPECIESVALUE_GET(handle, sval);
    MXCPTRCHECK(value);
    *value = sval->getConstant();
    return S_OK;
}

HRESULT MxCSpeciesValue_setConstant(struct MxSpeciesValueHandle *handle, int value) {
    MXSPECIESVALUE_GET(handle, sval);
    return sval->setConstant(value);
}

HRESULT MxCSpeciesValue_secreteL(struct MxSpeciesValueHandle *handle, double amount, struct MxParticleListHandle *to, double *secreted) {
    MXSPECIESVALUE_GET(handle, sval);
    MXCPTRCHECK(to); MXCPTRCHECK(to->MxObj);
    MXCPTRCHECK(secreted);
    *secreted = sval->secrete(amount, *(MxParticleList*)to->MxObj);
    return S_OK;
}

HRESULT MxCSpeciesValue_secreteD(struct MxSpeciesValueHandle *handle, double amount, double distance, double *secreted) {
    MXSPECIESVALUE_GET(handle, sval);
    MXCPTRCHECK(secreted);
    *secreted = sval->secrete(amount, distance);
    return S_OK;
}
