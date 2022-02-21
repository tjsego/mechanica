/**
 * @file MxFIO.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica data import/export interface
 * @date 2021-12-20
 * 
 */

#include <nlohmann/json.hpp>

#include <MxSimulator.h>
#include <MxLogger.h>
#include <mx_error.h>

#include <fstream>
#include <iostream>

#include "MxFIO.h"


using json = nlohmann::json;


const std::string MxFIO::KEY_TYPE = "MxIOType";
const std::string MxFIO::KEY_VALUE = "MxIOValue";

const std::string MxFIO::KEY_METADATA = "MetaData";
const std::string MxFIO::KEY_SIMULATOR = "Simulator";
const std::string MxFIO::KEY_UNIVERSE = "Universe";
const std::string MxFIO::KEY_MODULES = "Modules";


namespace mx { namespace io {


template <>
HRESULT toFile(const json &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    for(auto &el : dataElement.items()) {
        auto key = el.key();
        
        if(key == MxFIO::KEY_TYPE) {
            fileElement->type = el.value().get<std::string>();
        }
        else if(key == MxFIO::KEY_VALUE) {
            fileElement->value = el.value().get<std::string>();
        }
        else {
        
            fe = new MxIOElement();
            if(toFile(el.value(), metaData, fe) != S_OK) 
                return E_FAIL;
            
            fe->parent = fileElement;
            fileElement->children[key] = fe;

        }
    }

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, json *dataElement) { 

    try {

        (*dataElement)[MxFIO::KEY_TYPE] = fileElement.type;
        (*dataElement)[MxFIO::KEY_VALUE] = fileElement.value;
        for(MxIOChildMap::const_iterator feItr = fileElement.children.begin(); feItr != fileElement.children.end(); feItr++) {
            json &jv = (*dataElement)[feItr->first.c_str()];
            if(fromFile(*feItr->second, metaData, &jv) != S_OK) 
                return E_FAIL;
        }

    }
    catch (...) {
        Log(LOG_CRITICAL) << "Could not generate JSON data: " << fileElement.type << ", " << fileElement.value;
        return E_FAIL;
    }

    return S_OK;
}

std::string toStr(MxIOElement *fileElement, const MxMetaData &metaData) {

    json jroot;

    json jmetadata;
    MxIOElement femetadata;
    mx::io::toFile(metaData, MxMetaData(), &femetadata);

    if(mx::io::fromFile(femetadata, metaData, &jmetadata) != S_OK) 
        mx_exp(std::runtime_error("Could not translate meta data"));

    jroot[MxFIO::KEY_METADATA] = jmetadata;

    json jvalue;
    
    if(mx::io::fromFile(*fileElement, metaData, &jvalue) != S_OK) 
        mx_exp(std::runtime_error("Could not translate data"));

    jroot[MxFIO::KEY_VALUE] = jvalue;

    return jroot.dump(4);
}

std::string toStr(MxIOElement *fileElement) {

    return toStr(fileElement, MxMetaData());

}

MxIOElement *fromStr(const std::string &str, const MxMetaData &metaData) {

    json jroot = json::parse(str);

    MxIOElement *fe = new MxIOElement();

    if(mx::io::toFile(jroot[MxFIO::KEY_VALUE], metaData, fe) != S_OK) 
        mx_exp(std::runtime_error("Could not translate data"));

    return fe;
}

MxIOElement *fromStr(const std::string &str) {

    json jroot = json::parse(str);

    MxIOElement *fevalue = new MxIOElement(), femetadata;
    MxMetaData strMetaData, metaData;

    if(mx::io::toFile(jroot[MxFIO::KEY_METADATA], metaData, &femetadata) != S_OK) 
        mx_exp(std::runtime_error("Could not parse meta data"));
    
    if(mx::io::fromFile(femetadata, metaData, &strMetaData) != S_OK) 
        mx_exp(std::runtime_error("Could not translate meta data"));

    if(mx::io::toFile(jroot[MxFIO::KEY_VALUE], strMetaData, fevalue) != S_OK) 
        mx_exp(std::runtime_error("Could not translate data"));

    return fevalue;
}

}}


// MxFIOModule

void MxFIOModule::registerIOModule() {
    MxFIO::registerModule(this->moduleName(), this);
}

void MxFIOModule::load() {
    MxFIO::loadModule(this->moduleName());
}


// MxFIO


MxIOElement *MxFIO::fromFile(const std::string &loadFilePath) { 

    // Build root node from file contents

    std::ifstream fileContents_ifs(loadFilePath, std::ifstream::binary);
    if(!fileContents_ifs || !fileContents_ifs.good() || fileContents_ifs.fail()) 
        mx_exp(std::runtime_error(std::string("Error loading file: ") + loadFilePath));

    // Create a reader and get root node

    json jroot;

    fileContents_ifs >> jroot;

    fileContents_ifs.close();
    
    Log(LOG_INFORMATION) << "Loaded source: " << loadFilePath;

    // Create root io element and populate from root node

    MxMetaData metaData, metaDataFile;
    MxIOElement feMetaData;
    if(mx::io::toFile(jroot[MxFIO::KEY_METADATA], metaData, &feMetaData) != S_OK) 
        mx_exp(std::runtime_error("Could not unpack metadata"));
    if(mx::io::fromFile(feMetaData, metaData, &metaDataFile) != S_OK) 
        mx_exp(std::runtime_error("Could not load metadata"));

    Log(LOG_INFORMATION) << "Got file metadata: " << metaDataFile.versionMajor << "." << metaDataFile.versionMinor << "." << metaDataFile.versionPatch;

    MxFIO::currentRootElement = new MxIOElement();
    if(mx::io::toFile(jroot, metaDataFile, MxFIO::currentRootElement)) 
        mx_exp(std::runtime_error("Could not load simulation data"));
    
    Log(LOG_INFORMATION) << "Generated i/o from source: " << loadFilePath;
    
    return MxFIO::currentRootElement;
}

MxIOElement *MxFIO::generateMxIORootElement() {

    if(MxFIO::currentRootElement != NULL) 
        if(MxFIO::releaseMxIORootElement() != S_OK) 
            return NULL;

    MxIOElement *mxData = new MxIOElement();

    // Add metadata

    MxMetaData metaData;
    MxIOElement *feMetaData = new MxIOElement();
    if(mx::io::toFile(metaData, metaData, feMetaData) != S_OK) 
        mx_exp(std::runtime_error("Could not store metadata"));
    mxData->children[MxFIO::KEY_METADATA] = feMetaData;
    feMetaData->parent = mxData;

    // Add simulator
    
    auto simulator = MxSimulator::get();
    MxIOElement *feSimulator = new MxIOElement();
    if(mx::io::toFile(*simulator, metaData, feSimulator) != S_OK) 
        mx_exp(std::runtime_error("Could not store simulator"));
    mxData->children[MxFIO::KEY_SIMULATOR] = feSimulator;
    feSimulator->parent = mxData;

    // Add universe
    
    auto universe = MxUniverse::get();
    MxIOElement *feUniverse = new MxIOElement();
    if(mx::io::toFile(*universe, metaData, feUniverse) != S_OK) 
        mx_exp(std::runtime_error("Could not store universe"));
    mxData->children[MxFIO::KEY_UNIVERSE] = feUniverse;
    feUniverse->parent = mxData;

    // Add modules

    if(MxFIO::modules == NULL) 
        MxFIO::modules = new std::unordered_map<std::string, MxFIOModule*>();

    if(MxFIO::modules->size() > 0) {
        MxIOElement *feModules = new MxIOElement();
        for(auto &itr : *MxFIO::modules) {
            MxIOElement *feModule = new MxIOElement();
            if(itr.second->toFile(metaData, feModule) != S_OK) 
                mx_exp(std::runtime_error("Could not store module: " + itr.first));
            feModules->children[itr.first] = feModule;
            feModule->parent = feModules;
        }

        mxData->children[MxFIO::KEY_MODULES] = feModules;
        feModules->parent = mxData;
    }

    MxFIO::currentRootElement = mxData;

    return mxData;
}

HRESULT MxFIO::releaseMxIORootElement() { 

    if(MxFIO::currentRootElement == NULL) 
        return S_OK;

    MxIOElement *feMetaData = MxFIO::currentRootElement->children[MxFIO::KEY_METADATA];
    MxIOElement *feSimulator = MxFIO::currentRootElement->children[MxFIO::KEY_SIMULATOR];
    MxIOElement *feUniverse = MxFIO::currentRootElement->children[MxFIO::KEY_UNIVERSE];

    auto itr = MxFIO::currentRootElement->children.find(MxFIO::KEY_MODULES);
    if(itr != MxFIO::currentRootElement->children.end()) {
        for(auto &mItr : itr->second->children) 
            delete mItr.second;
        delete itr->second;
    }

    delete feMetaData;
    delete feSimulator;
    delete feUniverse;
    delete MxFIO::currentRootElement;

    MxFIO::currentRootElement = NULL;

    return S_OK;
}

HRESULT MxFIO::toFile(const std::string &saveFilePath) { 

    MxMetaData metaData;
    MxIOElement *mxData = generateMxIORootElement();

    // Create root node

    json jroot;

    if(mx::io::fromFile(*mxData, metaData, &jroot) != S_OK) 
        mx_exp(std::runtime_error("Could not translate final data"));

    // Write

    std::ofstream saveFile(saveFilePath);
    
    saveFile << jroot.dump(4);

    saveFile.close();

    return releaseMxIORootElement();
}

std::string MxFIO::toString() {

    MxMetaData metaData;
    MxIOElement *mxData = generateMxIORootElement();

    // Create root node

    json jroot;

    if(mx::io::fromFile(*mxData, metaData, &jroot) != S_OK) 
        mx_exp(std::runtime_error("Could not translate final data"));

    // Write

    std::string result = jroot.dump(4);

    if(releaseMxIORootElement() != S_OK) 
        mx_exp(std::runtime_error("Could not close root element"));

    return result;
}

void MxFIO::registerModule(const std::string moduleName, MxFIOModule *module) {
    if(MxFIO::modules == NULL) 
        MxFIO::modules = new std::unordered_map<std::string, MxFIOModule*>();
    
    auto itr = MxFIO::modules->find(moduleName);
    if(itr != MxFIO::modules->end()) 
        mx_exp(std::runtime_error("I/O module already registered: " + moduleName));

    (*MxFIO::modules)[moduleName] = module;
}

void MxFIO::loadModule(const std::string moduleName) {

    if(MxFIO::modules == NULL) 
        MxFIO::modules = new std::unordered_map<std::string, MxFIOModule*>();

    // Get registered module

    auto itr = MxFIO::modules->find(moduleName);
    if(itr == MxFIO::modules->end()) 
        mx_exp(std::runtime_error("I/O module not registered: " + moduleName));

    // Validate previous main import
    
    if(MxFIO::currentRootElement == NULL) 
        mx_exp(std::runtime_error("No import state"));
    
    // Get file metadata
    
    MxIOElement *feMetaData = MxFIO::currentRootElement->children[MxFIO::KEY_METADATA];
    MxMetaData metaData, metaDataFile;
    if(mx::io::fromFile(*feMetaData, metaData, &metaDataFile) != S_OK) 
        mx_exp(std::runtime_error("Could not load metadata"));

    // Get modules element
    
    auto mItr = MxFIO::currentRootElement->children.find(MxFIO::KEY_MODULES);
    if(mItr == MxFIO::currentRootElement->children.end()) 
        mx_exp(std::runtime_error("No loaded modules"));
    auto feModules = mItr->second;

    // Get module element
    
    mItr = feModules->children.find(moduleName);
    if(mItr == feModules->children.end()) 
        mx_exp(std::runtime_error("Module data not available: " + moduleName));

    // Issue module import
    
    if((*MxFIO::modules)[moduleName]->fromFile(metaDataFile, *mItr->second) != S_OK) 
        mx_exp(std::runtime_error("Module import failed: " + moduleName));
}

bool MxFIO::hasImport() {
    return MxFIO::importSummary != NULL;
}
