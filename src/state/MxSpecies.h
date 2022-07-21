/**
 * @file MxSpecies.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines wrap of SBML species; derived from carbon CSpecies.hpp and c_species.h written by Andy Somogyi
 * @date 2021-07-03
 * 
 */

#ifndef SRC_STATE_MXSPECIES_H_
#define SRC_STATE_MXSPECIES_H_

#include <mx_port.h>
#include <io/mx_io.h>
#include <string>

namespace libsbml{
    class Species;
    class SBMLNamespaces;
};

enum MxSpeciesFlags {

    SPECIES_BOUNDARY  = 1 << 0,
    SPECIES_SUBSTANCE = 1 << 2,
    SPECIES_CONSTANT  = 1 << 3,
    SPECIES_KONSTANT  = SPECIES_BOUNDARY | SPECIES_CONSTANT
};

/**
 * @brief The Mechanica species object. 
 * 
 * Mostly, this is a wrap of libSBML Species. 
 */
struct CAPI_EXPORT MxSpecies {
    libsbml::Species *species;
    int32_t flags() const;
    std::string str() const;
    void initDefaults();

    const std::string getId() const;
    int setId(const char *sid);
    const std::string getName() const;
    int setName(const char *name);
    const std::string getSpeciesType() const;
    int setSpeciesType(const char *sid);
    const std::string getCompartment() const;
    int setCompartment(const char *sid);
    double getInitialAmount() const;
    int setInitialAmount(double value);
    double getInitialConcentration() const;
    int setInitialConcentration(double value);
    const std::string getSubstanceUnits() const;
    int setSubstanceUnits(const char *sid);
    const std::string getSpatialSizeUnits() const;
    int setSpatialSizeUnits(const char *sid);
    const std::string getUnits() const;
    int setUnits(const char *sname);
    bool getHasOnlySubstanceUnits() const;
    int setHasOnlySubstanceUnits(int value);
    bool getBoundaryCondition() const;
    int setBoundaryCondition(int value);
    int getCharge() const;
    int setCharge(int value);
    bool getConstant() const;
    int setConstant(int value);
    const std::string getConversionFactor() const;
    int setConversionFactor(const char *sid);
    bool isSetId() const;
    bool isSetName() const;
    bool isSetSpeciesType() const;
    bool isSetCompartment() const;
    bool isSetInitialAmount() const;
    bool isSetInitialConcentration() const;
    bool isSetSubstanceUnits() const;
    bool isSetSpatialSizeUnits() const;
    bool isSetUnits() const;
    bool isSetCharge() const;
    bool isSetConversionFactor() const;
    bool isSetConstant() const;
    bool isSetBoundaryCondition() const;
    bool isSetHasOnlySubstanceUnits() const;
    int unsetId();
    int unsetName();
    int unsetConstant();
    int unsetSpeciesType();
    int unsetInitialAmount();
    int unsetInitialConcentration();
    int unsetSubstanceUnits();
    int unsetSpatialSizeUnits();
    int unsetUnits();
    int unsetCharge();
    int unsetConversionFactor();
    int unsetCompartment();
    int unsetBoundaryCondition();
    int unsetHasOnlySubstanceUnits();
    int hasRequiredAttributes();

    MxSpecies();
    MxSpecies(const std::string &s);
    MxSpecies(const MxSpecies &other);
    ~MxSpecies();

    /**
     * @brief Get a JSON string representation
     * 
     * @return std::string 
     */
    std::string toString();

    /**
     * @brief Create from a JSON string representation. 
     * 
     * @param str 
     * @return MxSpecies* 
     */
    static MxSpecies *fromString(const std::string &str);
};


CAPI_FUNC(libsbml::SBMLNamespaces*) MxGetSBMLNamespaces();

namespace mx { namespace io { 

template <>
HRESULT toFile(const MxSpecies &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxSpecies *dataElement);

}};

#endif /* SRC_STATE_MXSPECIES_H_ */
