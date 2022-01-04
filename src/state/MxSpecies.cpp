/**
 * @file MxSpecies.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines wrap of SBML species; derived from carbon CSpecies.cpp written by Andy Somogyi
 * @date 2021-07-03
 * 
 */

#include "MxSpecies.h"
#include <MxLogger.h>
#include <mx_error.h>

#include <sbml/Species.h>
#include <sbml/SBMLNamespaces.h>

#include <iostream>
#include <regex>


static libsbml::SBMLNamespaces *sbmlns = NULL;

libsbml::SBMLNamespaces* MxGetSBMLNamespaces() {
    if(!sbmlns) {
        sbmlns = new libsbml::SBMLNamespaces();
    }
    return sbmlns;
}

const std::string MxSpecies::getId() const
{
    if (species) return this->species->getId();
    return "";
}

int MxSpecies::setId(const char *sid)
{
    if (species) return species->setId(std::string(sid));
    return -1;
}

const std::string MxSpecies::getName() const
{
    if(species && species->isSetName()) return species->getName();
    return "";
}

int MxSpecies::setName(const char *name)
{
    if (species) return species->setName(std::string(name));
    return -1;
}

const std::string MxSpecies::getSpeciesType() const
{
    if (species) return species->getSpeciesType();
    return "";
}

int MxSpecies::setSpeciesType(const char *sid)
{
    if(species) return species->setSpeciesType(sid);
    return -1;
}

const std::string MxSpecies::getCompartment() const
{
    if(species) return species->getCompartment();
    return "";
}

double MxSpecies::getInitialAmount() const
{
    if(species && species->isSetInitialAmount()) return species->getInitialAmount();
    return 0.0;
}

int MxSpecies::setInitialAmount(double value)
{
    if(species) return species->setInitialAmount(value);
    return -1;
}

double MxSpecies::getInitialConcentration() const
{
    if(species && species->isSetInitialConcentration()) return species->getInitialConcentration();
    return 0.0;
}

int MxSpecies::setInitialConcentration(double value)
{
    if(species) return species->setInitialConcentration(value);
    return -1;
}

const std::string MxSpecies::getSubstanceUnits() const
{
    if(species) return species->getSubstanceUnits();
    return "";
}

const std::string MxSpecies::getSpatialSizeUnits() const
{
    if(species) return species->getSpatialSizeUnits();
    return "";
}

const std::string MxSpecies::getUnits() const
{
    if(species) return species->getUnits();
    return "";
}

bool MxSpecies::getHasOnlySubstanceUnits() const
{
    if(species && species->isSetHasOnlySubstanceUnits()) return species->getHasOnlySubstanceUnits();
    return false;
}

int MxSpecies::setHasOnlySubstanceUnits(int value)
{
    if(species) return species->setHasOnlySubstanceUnits((bool)value);
    return -1;
}

bool MxSpecies::getBoundaryCondition() const
{
    if(species && species->isSetBoundaryCondition()) return species->getBoundaryCondition();
    return false;
}

int MxSpecies::setBoundaryCondition(int value)
{
    if(species) return species->setBoundaryCondition((bool)value);
    return -1;
}

int MxSpecies::getCharge() const
{
    if(species) return species->getCharge();
    return 0;
}

bool MxSpecies::getConstant() const
{
    if(species && species->isSetConstant()) return species->getConstant();
    return false;
}

int MxSpecies::setConstant(int value)
{
    if(species) return species->setBoundaryCondition((bool)value);
    return -1;
}

const std::string MxSpecies::getConversionFactor() const
{
    if(species) return species->getConversionFactor();
    return "";
}

bool MxSpecies::isSetId() const
{
    if(species) return species->isSetId();
    return false;
}

bool MxSpecies::isSetName() const
{
    if(species) return species->isSetName();
    return false;
}

bool MxSpecies::isSetSpeciesType() const
{
    if(species) return species->isSetSpeciesType();
    return false;
}

bool MxSpecies::isSetCompartment() const
{
    if(species) return species->isSetCompartment();
    return false;
}

bool MxSpecies::isSetInitialAmount() const
{
    if(species) return species->isSetInitialAmount();
    return false;
}

bool MxSpecies::isSetInitialConcentration() const
{
    if(species) return species->isSetInitialConcentration();
    return false;
}

bool MxSpecies::isSetSubstanceUnits() const
{
    if(species) return species->isSetSubstanceUnits();
    return false;
}

bool MxSpecies::isSetSpatialSizeUnits() const
{
    if(species) return species->isSetSpatialSizeUnits();
    return false;
}

bool MxSpecies::isSetUnits() const
{
    if(species) return species->isSetUnits();
    return false;
}

bool MxSpecies::isSetCharge() const
{
    if(species) return species->isSetCharge();
    return false;
}

bool MxSpecies::isSetConversionFactor() const
{
    if(species) return species->isSetConversionFactor();
    return false;
}

bool MxSpecies::isSetConstant() const
{
    if(species) return species->isSetConstant();
    return false;
}

bool MxSpecies::isSetBoundaryCondition() const
{
    if(species) return species->isSetBoundaryCondition();
    return false;
}

bool MxSpecies::isSetHasOnlySubstanceUnits() const
{
    if(species) return species->isSetHasOnlySubstanceUnits();
    return false;
}

int MxSpecies::setCompartment(const char *sid)
{
    if(species) return species->setCompartment(sid);
    return -1;
}

int MxSpecies::setSubstanceUnits(const char *sid)
{
    if(species) return species->setSubstanceUnits(sid);
    return -1;
}

int MxSpecies::setSpatialSizeUnits(const char *sid)
{
    if(species) return species->setSpatialSizeUnits(sid);
    return -1;
}

int MxSpecies::setUnits(const char *sname)
{
    if(species) return species->setUnits(sname);
    return -1;
}

int MxSpecies::setCharge(int value)
{
    if(species) return species->setCharge(value);
    return -1;
}

int MxSpecies::setConversionFactor(const char *sid)
{
    if(species) return species->setConversionFactor(sid);
    return -1;
}

int MxSpecies::unsetId()
{
    if(species) return species->unsetId();
    return -1;
}

int MxSpecies::unsetName()
{
    if(species) return species->unsetName();
    return -1;
}

int MxSpecies::unsetConstant()
{
    if(species) return species->unsetConstant();
    return -1;
}

int MxSpecies::unsetSpeciesType()
{
    if(species) return species->unsetSpeciesType();
    return -1;
}

int MxSpecies::unsetInitialAmount()
{
    if(species) return species->unsetInitialAmount();
    return -1;
}

int MxSpecies::unsetInitialConcentration()
{
    if(species) return species->unsetInitialConcentration();
    return -1;
}

int MxSpecies::unsetSubstanceUnits()
{
    if(species) return species->unsetSubstanceUnits();
    return -1;
}

int MxSpecies::unsetSpatialSizeUnits()
{
    if(species) return species->unsetSpatialSizeUnits();
    return -1;
}

int MxSpecies::unsetUnits()
{
    if(species) return species->unsetUnits();
    return -1;
}

int MxSpecies::unsetCharge()
{
    if(species) return species->unsetCharge();
    return -1;
}

int MxSpecies::unsetConversionFactor()
{
    if(species) return species->unsetConversionFactor();
    return -1;
}

int MxSpecies::unsetCompartment()
{
    if(species) return species->unsetCompartment();
    return -1;
}

int MxSpecies::unsetBoundaryCondition()
{
    if(species) return species->unsetBoundaryCondition();
    return -1;
}

int MxSpecies::unsetHasOnlySubstanceUnits()
{
    if(species) return species->unsetHasOnlySubstanceUnits();
    return -1;
}

int MxSpecies::hasRequiredAttributes()
{
    if(species) return species->hasRequiredAttributes();
    return -1;
}

static int MxSpecies_init(MxSpecies *self, const std::string &s) {
    try {
        
        static std::regex e ("\\s*(const\\s+)?(\\$)?(\\w+)(\\s*=\\s*)?([-+]?[0-9]*\\.?[0-9]+)?\\s*");
        
        std::smatch sm;    // same as std::match_results<string::const_iterator> sm;
        
        // if we have a match, it looks like this:
        // matches for "const S1 = 234234.5"
        // match(0):(19)"const S1 = 234234.5"
        // match(1):(6)"const "
        // match(2):(0)""
        // match(3):(2)"S1"
        // match(4):(3)" = "
        // match(5):(8)"234234.5"
        static const int CNST = 1;  // Windows complains if name is CONST
        static const int BOUNDARY = 2;
        static const int ID = 3;
        static const int EQUAL = 4;
        static const int INIT = 5;
        
        if(std::regex_match (s,sm,e) && sm.size() == 6) {
            // check if name is valid sbml id
            if(!sm[ID].matched || !libsbml::SyntaxChecker_isValidSBMLSId(sm[ID].str().c_str())) {
                mx_exp(std::runtime_error("invalid Species id: \"" + sm[ID].str() + "\""));
                return -1;
            }
            
            if(sm[INIT].matched && !sm[EQUAL].matched) {
                mx_exp(std::runtime_error("Species has initial assignment value without equal symbol: \"" + s + "\""));
                return -1;
            }
            
            self->species = new libsbml::Species(MxGetSBMLNamespaces());
            self->species->setId(sm[ID].str());
            self->species->setBoundaryCondition(sm[BOUNDARY].matched);
            self->species->setConstant(sm[CNST].matched);
            
            if(sm[INIT].matched) {
                self->species->setInitialConcentration(std::stod(sm[INIT].str()));
            }
            
            return 0;
        }
        else {
            mx_exp(std::runtime_error("invalid Species string: \"" + s + "\""));
            return -1;
        }
    }
    catch(const std::exception &e) {
        mx_exp(std::runtime_error("error creating Species(" + s + "\") : " + e.what()));
        return -1;
    }
    return -1;
}

std::string MxSpecies::str() const {
    std::string s = "Species('";
    if(species->isSetBoundaryCondition() && species->getBoundaryCondition()) {
        s += "$";
    }
    s += species->getId();
    s += "')";
    return s;
}

int32_t MxSpecies::flags() const
{
    int32_t r = 0;

    if(species) {
    
        if(species->getBoundaryCondition()) {
            r |= SPECIES_BOUNDARY;
        }
        
        if(species->getHasOnlySubstanceUnits()) {
            r |= SPECIES_SUBSTANCE;
        }
        
        if(species->getConstant()) {
            r |= SPECIES_CONSTANT;
        }

    }
    
    return r;
}

void MxSpecies::initDefaults()
{
    species->initDefaults();
}

MxSpecies::MxSpecies() {}

MxSpecies::MxSpecies(const std::string &s) : MxSpecies() {
    int result = MxSpecies_init(this, s);
    if(result) Log(LOG_CRITICAL) << "Species creation failed with return code " << result;
    else Log(LOG_DEBUG) << "Species creation success: " << species->getId();
}

MxSpecies::MxSpecies(const MxSpecies &other) : MxSpecies() {
    species = libsbml::Species_clone(other.species);
}

MxSpecies::~MxSpecies() {
    if(species) {
        delete species;
        species = 0;
    }
}

namespace mx{ namespace io { 

template <>
HRESULT toFile(const MxSpecies &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MxIOElement *fe;

    fe = new MxIOElement();
    if(toFile(dataElement.getId(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["id"] = fe;

    fe = new MxIOElement();
    if(toFile(dataElement.getName(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["name"] = fe;

    fe = new MxIOElement();
    if(toFile(dataElement.getSpeciesType(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["speciesType"] = fe;

    fe = new MxIOElement();
    if(toFile(dataElement.getCompartment(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["compartment"] = fe;

    fe = new MxIOElement();
    if(toFile(dataElement.getInitialAmount(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["initialAmount"] = fe;

    fe = new MxIOElement();
    if(toFile(dataElement.getInitialConcentration(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["initialConcentration"] = fe;

    fe = new MxIOElement();
    if(toFile(dataElement.getSubstanceUnits(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["substanceUnits"] = fe;

    fe = new MxIOElement();
    if(toFile(dataElement.getSpatialSizeUnits(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["spatialSizeUnits"] = fe;

    fe = new MxIOElement();
    if(toFile(dataElement.getUnits(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["units"] = fe;
    
    fe = new MxIOElement();
    if(toFile(dataElement.getHasOnlySubstanceUnits(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["hasOnlySubstanceUnits"] = fe;
    
    fe = new MxIOElement();
    if(toFile(dataElement.getBoundaryCondition(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["boundaryCondition"] = fe;
    
    fe = new MxIOElement();
    if(toFile(dataElement.getCharge(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["charge"] = fe;
    
    fe = new MxIOElement();
    if(toFile(dataElement.getConstant(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["constant"] = fe;
    
    fe = new MxIOElement();
    if(toFile(dataElement.getConversionFactor(), metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["conversionFactor"] = fe;

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxSpecies *dataElement) {
    std::unordered_map<std::string, MxIOElement*>::const_iterator feItr;
    auto &c = fileElement.children;
    
    std::string s;
    double d;
    int i;

    dataElement->species = new libsbml::Species(MxGetSBMLNamespaces());

    feItr = c.find("id");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &s) != S_OK) 
        return E_FAIL;
    dataElement->setId(s.c_str());

    feItr = c.find("name");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &s) != S_OK) 
        return E_FAIL;
    dataElement->setName(s.c_str());

    feItr = c.find("speciesType");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &s) != S_OK) 
        return E_FAIL;
    dataElement->setSpeciesType(s.c_str());

    feItr = c.find("compartment");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &s) != S_OK) 
        return E_FAIL;
    dataElement->setCompartment(s.c_str());

    feItr = c.find("initialAmount");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &d) != S_OK) 
        return E_FAIL;
    dataElement->setInitialAmount(d);

    feItr = c.find("initialConcentration");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &d) != S_OK) 
        return E_FAIL;
    dataElement->setInitialConcentration(d);

    feItr = c.find("substanceUnits");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &s) != S_OK) 
        return E_FAIL;
    dataElement->setSubstanceUnits(s.c_str());

    feItr = c.find("spatialSizeUnits");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &s) != S_OK) 
        return E_FAIL;
    dataElement->setSpatialSizeUnits(s.c_str());

    feItr = c.find("units");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &s) != S_OK) 
        return E_FAIL;
    dataElement->setUnits(s.c_str());

    feItr = c.find("hasOnlySubstanceUnits");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &i) != S_OK) 
        return E_FAIL;
    dataElement->setHasOnlySubstanceUnits(i);

    feItr = c.find("boundaryCondition");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &i) != S_OK) 
        return E_FAIL;
    dataElement->setBoundaryCondition(i);

    feItr = c.find("charge");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &i) != S_OK) 
        return E_FAIL;
    dataElement->setCharge(i);

    feItr = c.find("constant");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &i) != S_OK) 
        return E_FAIL;
    dataElement->setConstant(i);

    feItr = c.find("conversionFactor");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &s) != S_OK) 
        return E_FAIL;
    dataElement->setConversionFactor(s.c_str());

    return S_OK;
}

}};
