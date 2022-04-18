/**
 * @file MxIO.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica import/export interface
 * @date 2021-12-21
 * 
 */

#include <mechanica_private.h>

#include "Mx3DFIO.h"
#include "MxFIO.h"


/**
 * @brief Mechanica import/export interface
 * 
 */
struct CAPI_EXPORT MxIO {
    /**
     * @brief Load a 3D format file
     * 
     * @param filePath path of file
     * @return Mx3DFStructure* 3D format data container
     */
    static Mx3DFStructure *fromFile3DF(const std::string &filePath) {
        return Mx3DFIO::fromFile(filePath);
    }

    /**
     * @brief Export engine state to a 3D format file
     * 
     * @param format format of file
     * @param filePath path of file
     * @param pRefinements mesh refinements applied when generating meshes
     * @return HRESULT 
     */
    static HRESULT toFile3DF(const std::string &format, const std::string &filePath, const unsigned int &pRefinements=0) {
        return Mx3DFIO::toFile(format, filePath, pRefinements);
    }

    /**
     * @brief Save a simulation to file
     * 
     * @param saveFilePath absolute path to file
     * @return HRESULT 
     */
    static HRESULT toFile(const std::string &saveFilePath) {
        return MxFIO::toFile(saveFilePath);
    }

    /**
     * @brief Return a simulation state as a JSON string
     * 
     * @return std::string 
     */
    static std::string toString() {
        return MxFIO::toString();
    }

    /**
     * @brief Get the id of a particle according to import data that 
     * corresponds to a particle id of current data. 
     * 
     * Only valid between initialization and the first simulation step, 
     * after which the import summary data is purged. 
     * 
     * @param pId id of particle in exported data
     * @return int >=0 if particle is found; -1 otherwise
     */
    static int mapImportParticleId(const unsigned int &pId);

    /**
     * @brief Get the id of a particle type according to import data that 
     * corresponds to a particle type id of current data. 
     * 
     * Only valid between initialization and the first simulation step, 
     * after which the import summary data is purged. 
     * 
     * @param pId id of particle type in exported data
     * @return int >=0 if particle type is found; -1 otherwise
     */
    static int mapImportParticleTypeId(const unsigned int &pId);
};
