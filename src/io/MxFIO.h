/**
 * @file MxFIO.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica data import/export interface
 * @date 2021-12-20
 * 
 */

#ifndef SRC_IO_MXFIO_H_
#define SRC_IO_MXFIO_H_


#include "mx_io.h"


namespace mx { namespace io {

/**
 * @brief Generate a JSON string representation of an intermediate I/O object. 
 * 
 * @param fileElement object to convert
 * @param metaData meta data of target installation
 * @return std::string 
 */
std::string toStr(MxIOElement *fileElement, const MxMetaData &metaData);

/**
 * @brief Generate a JSON string representation of an intermediate I/O object. 
 * 
 * Current installation is target installation.
 * 
 * @param fileElement object to convert
 * @return std::string 
 */
std::string toStr(MxIOElement *fileElement);

/**
 * @brief Generate an intermediate I/O object from a JSON string. 
 * 
 * @param str JSON string
 * @param metaData meta data of target installation
 * @return MxIOElement* 
 */
MxIOElement *fromStr(const std::string &str, const MxMetaData &metaData);

/**
 * @brief Generate an intermediate I/O object from a JSON string. 
 * 
 * Installation during string export is target installation.
 * 
 * @param str JSON string
 * @return MxIOElement* 
 */
MxIOElement *fromStr(const std::string &str);

/**
 * @brief Generate a JSON string representation of an object. 
 * 
 * @tparam T type of source object
 * @param dataElement source object
 * @param metaData meta data of target installation
 * @return std::string 
 */
template <typename T>
std::string toString(const T &dataElement, const MxMetaData &metaData) {
    MxIOElement *fe = new MxIOElement();
    if(toFile<T>(dataElement, metaData, fe) != S_OK) 
        return "";
    return toStr(fe, metaData);
}

/**
 * @brief Generate a JSON string representation of an object. 
 * 
 * Current installation is target installation. 
 * 
 * @tparam T type of source object
 * @param dataElement source object
 * @return std::string 
 */
template <typename T>
std::string toString(const T &dataElement) {
    return toString(dataElement, MxMetaData());
}

/**
 * @brief Generate an object from a JSON string. 
 * 
 * @tparam T type of object
 * @param str JSON string
 * @param metaData meta data of target installation
 * @return T 
 */
template <typename T>
T fromString(const std::string &str, const MxMetaData &metaData) { 
    MxIOElement *fe = fromStr(str, metaData);
    T de;
    fromFile<T>(*fe, metaData, &de);
    return de;
}

/**
 * @brief Generate an object from a JSON string. 
 * 
 * Current installation is target installation. 
 * 
 * @tparam T type of object
 * @param str JSON string
 * @return T 
 */
template <typename T>
T fromString(const std::string &str) {
    return fromString<T>(str, MxMetaData());
}

/**
 * @brief Delete a file element and all child elements
 * 
 * @param fileElement file element to delete; pointer is set to 0
 * @return HRESULT 
 */
HRESULT deleteElement(MxIOElement **fileElement);

}};

/**
 * @brief Mechanica data import summary. 
 * 
 * Not every datum of an imported Mechanica simulation state is conserved 
 * (e.g., particle id). This class provides the information necessary 
 * to exactly translate the imported data of a previously exported simulation. 
 * 
 */
struct MxFIOImportSummary {

    std::unordered_map<unsigned int, unsigned int> particleIdMap;
    /** Map of imported particle ids to loaded particle ids */

    std::unordered_map<unsigned int, unsigned int> particleTypeIdMap;
    /** Map of imported particle type ids to loaded particle ids */

};


/**
 * @brief Interface for Mechanica peripheral module I/O (e.g., models)
 * 
 */
struct MxFIOModule {

    /**
     * @brief Name of module. Used as a storage key. 
     * 
     * @return std::string 
     */
    virtual std::string moduleName() = 0;

    /**
     * @brief Export module data. 
     * 
     * @param metaData metadata of current installation
     * @param fileElement container to store serialized data
     * @return HRESULT 
     */
    virtual HRESULT toFile(const MxMetaData &metaData, MxIOElement *fileElement) = 0;

    /**
     * @brief Import module data. 
     * 
     * @param metaData metadata of import file. 
     * @param fileElement container of stored serialized data
     * @return HRESULT 
     */
    virtual HRESULT fromFile(const MxMetaData &metaData, const MxIOElement &fileElement) = 0;

    /**
     * @brief Register this module for I/O events
     * 
     */
    void registerIOModule();

    /**
     * @brief User-facing function to load module data from main import. 
     * 
     * Must only be called after main import. 
     * 
     */
    void load();

};


/**
 * @brief Mechanica data import/export interface. 
 * 
 * This interface provides methods for serializing/deserializing 
 * the state of a Mechanica simulation. 
 * 
 */
struct MxFIO {

    static const std::string KEY_TYPE;
    /** Key for basic element type storage */

    static const std::string KEY_VALUE;
    /** Key for basic element value storage */

    static const std::string KEY_METADATA;
    /** Key for simulation metadata storage */

    static const std::string KEY_SIMULATOR;
    /** Key for simulation simulator storage */

    static const std::string KEY_UNIVERSE;
    /** Key for simulation universe storage */

    static const std::string KEY_MODULES;
    /** Key for module i/o storage */

    inline static MxIOElement *currentRootElement = NULL;
    /** Current root element, if any */

    inline static MxFIOImportSummary *importSummary = NULL;
    /** Import summary of most recent import */

    /**
     * @brief Generate root element from current simulation state
     * 
     * @return MxIOElement* 
     */
    static MxIOElement *generateMxIORootElement();

    /**
     * @brief Release current root element
     * 
     * @return HRESULT 
     */
    static HRESULT releaseMxIORootElement();

    /**
     * @brief Load a simulation from file
     * 
     * @param loadFilePath absolute path to file
     * @return MxIOElement* 
     */
    static MxIOElement *fromFile(const std::string &loadFilePath);

    /**
     * @brief Save a simulation to file
     * 
     * @param saveFilePath absolute path to file
     * @return HRESULT 
     */
    static HRESULT toFile(const std::string &saveFilePath);

    /**
     * @brief Return a simulation state as a JSON string
     * 
     * @return std::string 
     */
    static std::string toString();

    /**
     * @brief Register a module for I/O events
     * 
     * @param moduleName name of module
     * @param module borrowed pointer to module
     */
    static void registerModule(const std::string moduleName, MxFIOModule *module);

    /**
     * @brief Load a module from imported data. 
     * 
     * Can only be called after main initial import. 
     * 
     * @param moduleName 
     */
    static void loadModule(const std::string moduleName);

    /**
     * @brief Test whether imported data is available. 
     * 
     * @return true 
     * @return false 
     */
    static bool hasImport();

private:
    
    inline static std::unordered_map<std::string, MxFIOModule*> *modules = NULL;

};


#endif // SRC_IO_MXFIO_H_
