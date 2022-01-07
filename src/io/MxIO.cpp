/**
 * @file MxIO.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica import/export interface
 * @date 2021-12-21
 * 
 */

#include "MxIO.h"


int MxIO::mapImportParticleId(const unsigned int &pId) {
    if(MxFIO::importSummary == NULL) 
        return -1;
    
    auto itr = MxFIO::importSummary->particleIdMap.find(pId);
    if(itr == MxFIO::importSummary->particleIdMap.end()) 
        return -1;
    
    return itr->second;
}

int MxIO::mapImportParticleTypeId(const unsigned int &pId) {
    if(MxFIO::importSummary == NULL) 
        return -1;
    
    auto itr = MxFIO::importSummary->particleTypeIdMap.find(pId);
    if(itr == MxFIO::importSummary->particleTypeIdMap.end()) 
        return -1;
    
    return itr->second;
}
