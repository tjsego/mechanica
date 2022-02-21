/**
 * @file mx_cuda.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines CUDA support and utilities in Mechanica runtime
 * @date 2021-11-09
 * 
 */

// TODO: implement support for JIT-compiled programs and kernel usage in wrapped languages

#ifndef _INCLUDE_MX_CUDA_H_
#define _INCLUDE_MX_CUDA_H_

#include "mx_error.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <vector>
#include <stdexcept>
#include <string>

inline CUresult mx_cuda_errorchk(CUresult retCode, const char *file, int line) {
    if(retCode != CUDA_SUCCESS) {
        std::string msg = "CUDA failed with error: ";
        const char *cmsg;
        cuGetErrorName(retCode, &cmsg);
        msg += std::string(cmsg);
        msg += ", " + std::string(file) + ", " + std::to_string(line);
        mx_exp(std::runtime_error(msg.c_str()));
    }
    return retCode;
}
#ifndef MX_CUDA_CALL
#define MX_CUDA_CALL(res) mx_cuda_errorchk(res, __FILE__, __LINE__)
#endif

inline nvrtcResult mx_nvrtc_errorchk(nvrtcResult retCode, const char *file, int line) {
    if(retCode != NVRTC_SUCCESS) {
        std::string msg = "NVRTC failed with error: ";
        msg += std::string(nvrtcGetErrorString(retCode));
        msg += ", " + std::string(file) + ", " + std::to_string(line);
        mx_exp(std::runtime_error(msg.c_str()));
    }
    return retCode;
}
#ifndef MX_NVRTC_CALL
#define MX_NVRTC_CALL(res) mx_nvrtc_errorchk(res, __FILE__, __LINE__)
#endif

inline cudaError_t mx_cudart_errorchk(cudaError_t retCode, const char *file, int line) {
    if(retCode != cudaSuccess) {
        std::string msg = "NVRTC failed with error: ";
        msg += std::string(cudaGetErrorString(retCode));
        msg += ", " + std::string(file) + ", " + std::to_string(line);
        mx_exp(std::runtime_error(msg.c_str()));
    }
    return retCode;
}
#ifndef MX_CUDART_CALL
#define MX_CUDART_CALL(res) mx_cudart_errorchk(res, __FILE__, __LINE__)
#endif


/**
 * @brief Convenience class for loading source from file and storing, here intended for CUDA
 * 
 */
struct MxCUDARTSource {
    std::string source;
    const char *name;

    MxCUDARTSource(const char *filePath, const char *_name);
    const char *c_str() const;
};


/**
 * @brief A JIT-compiled CUDA Mechanica program. 
 * 
 * This object wraps the procedures for turning CUDA source into executable kernels at runtime 
 * using NVRTC. 
 * 
 */
struct MxCUDARTProgram {

    nvrtcProgram *prog;
    char *ptx;
    std::vector<std::string> opts;
    std::vector<std::string> namedExprs;
    std::vector<std::string> includePaths;
    int arch;
    bool is_compute;

    MxCUDARTProgram();
    ~MxCUDARTProgram();

    /**
     * @brief Add a compilation option
     * 
     * @param opt 
     */
    void addOpt(const std::string &opt);

    /**
     * @brief Add a directory to include in the search path
     * 
     * @param ipath 
     */
    void addIncludePath(const std::string &ipath);

    /**
     * @brief Add a named expression
     * 
     * @param namedExpr 
     */
    void addNamedExpr(const std::string &namedExpr);

    /**
     * @brief Compile the program
     * 
     * @param src 
     * @param name 
     * @param numHeaders 
     * @param headers 
     * @param includeNames 
     */
    void compile(const char *src, const char *name, int numHeaders=0, const char *const *headers=0, const char *const *includeNames=0);

    /**
     * @brief Get the lowered name of a named expression. Cannot be called until after compilaton. 
     * 
     * @param namedExpr 
     * @return std::string 
     */
    std::string loweredName(const std::string namedExpr);

};

struct MxCUDAContext;


/**
 * @brief A CUDA kernel from a JIT-compiled Mechanica program
 * 
 */
struct MxCUDAFunction {
    const std::string name;
    unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes;
    CUstream hStream;
    void **extra;

    MxCUDAFunction(const std::string &name, MxCUDAContext *context);
    ~MxCUDAFunction();

    HRESULT autoConfig(const unsigned int &_nr_arrayElems, 
                       size_t dynamicSMemSize=0, 
                       size_t (*blockSizeToDynamicSMemSize)(int)=0, 
                       int blockSizeLimit=0);

    void operator()(void **args);
    void operator()(int nargs, ...);

private:
    CUfunction *function;
    MxCUDAContext *context;
};


/**
 * @brief A convenience wrap of the CUDA context for JIT-compiled Mechanica programs. 
 * 
 */
struct MxCUDAContext {

    CUcontext *context;
    CUdevice device;
    CUmodule *module;
    std::vector<CUjit_option> compileOpts;
    std::vector<void*> compileOptVals;

    /* Flag signifying whether this context is attached to a CPU thread. */
    bool attached;

    MxCUDAContext(CUdevice device=0);
    ~MxCUDAContext();

    void addOpt(CUjit_option opt, void *val);

    /**
     * @brief Load a compiled program. 
     * 
     * @param prog 
     * @param numOptions 
     * @param opts 
     * @param optionalValues 
     */
    void loadProgram(const MxCUDARTProgram &prog);

    /**
     * @brief Load pre-compiled ptx
     * 
     * @param ptx 
     */
    void loadPTX(const char *ptx);

    /**
     * @brief Get a cuda function from a loaded module. 
     * 
     * @param name 
     * @return MxCUDAfunction* 
     */
    MxCUDAFunction *getFunction(const char *name);

    /**
     * @brief Get a global pointer from a loaded module. 
     * 
     * @param name 
     * @return CUdeviceptr* 
     */
    CUdeviceptr *getGlobal(const char *name);

    /**
     * @brief Get the size of a global pointer from a loaded module. 
     * 
     * @param name 
     * @return size_t 
     */
    size_t getGlobalSize(const char *name);

    /**
     * @brief Push the context onto the stack of current contexts of the CPU thread. 
     * 
     * The context becomes the current context of the CPU thread. 
     * 
     */
    void pushCurrent();

    /**
     * @brief Pop the context from the stack and returns the new current context of contexts of the CPU thread. 
     * 
     * After being popped, the context can be pushed to a different CPU thread. 
     * 
     * @return CUcontext* 
     */
    CUcontext *popCurrent();

    /**
     * @brief Destroy the context. 
     * 
     */
    void destroy();

    /**
     * @brief Get the API version of this context. 
     * 
     * @return int 
     */
    int getAPIVersion();

    /**
     * @brief Synchronize GPU with calling CPU thread. Blocks until all preceding tasks of the current context are complete. 
     * 
     */
    static void sync();
};


/**
 * @brief A simple interface with a CUDA device
 * 
 */
struct MxCUDADevice {

    CUdevice *device;

    MxCUDADevice();
    ~MxCUDADevice();
    
    /**
     * @brief Attach a CUDA-supporting device by id. 
     * 
     * @param deviceId id of device
     */
    void attachDevice(const int &deviceId=0);

    /**
     * @brief Detach currently attached device. 
     * 
     */
    void detachDevice();

    /**
     * @brief Get the name of attached device
     * 
     * @return std::string 
     */
    std::string name();

    /**
     * @brief Get architecture of attached device
     * 
     * @return int 
     */
    int arch();

    /**
     * @brief Get the total memory of attached device
     * 
     * @return size_t 
     */
    size_t totalMem();

    /**
     * @brief Get the attribute value of attached device
     * 
     * @param attrib 
     * @return int 
     */
    int getAttribute(const int &attrib);

    /**
     * @brief Get the PCI bus id of this device. 
     * 
     * @return std::string 
     */
    std::string PCIBusId();

    /**
     * @brief Create a context on this device. 
     * 
     * Calling thread is responsible for destroying context. 
     * 
     * @return MxCUDAContext* 
     */
    MxCUDAContext *createContext();

    /**
     * @brief Get the current context. If none exists, one is created. 
     * 
     * @return MxCUDAContext* 
     */
    MxCUDAContext *currentContext();

    /**
     * @brief Get the name of a device
     * 
     * @param deviceId 
     * @return std::string 
     */
    static std::string getDeviceName(const int &deviceId);

    /**
     * @brief Get the total memory of device
     * 
     * @param deviceId 
     * @return size_t 
     */
    static size_t getDeviceTotalMem(const int &deviceId);

    /**
     * @brief Get the attribute value of a device
     * 
     * @param deviceId 
     * @param attrib 
     * @return int 
     */
    static int getDeviceAttribute(const int &deviceId, const int &attrib);

    /**
     * @brief Get number of available compute-capable devices
     * 
     * @return int 
     */
    static int getNumDevices();

    /**
     * @brief Get the PCI bus id of a device
     * 
     * @param deviceId 
     * @return std::string 
     */
    static std::string getDevicePCIBusId(const int &deviceId);

    /**
     * @brief Get the device id of the current context of the calling CPU thread. 
     * 
     * @return int 
     */
    static int getCurrentDevice();

    /**
     * @brief Maximum number of threads per block
     * 
     * @return int 
     */
    int maxThreadsPerBlock();

    /**
     * @brief Maximum x-dimension of a block
     * 
     * @return int 
     */
    int maxBlockDimX();

    /**
     * @brief Maximum y-dimension of a block
     * 
     * @return int 
     */
    int maxBlockDimY();

    /**
     * @brief Maximum z-dimension of a block
     * 
     * @return int 
     */
    int maxBlockDimZ();

    /**
     * @brief Maximum x-dimension of a grid
     * 
     * @return int 
     */
    int maxGridDimX();

    /**
     * @brief Maximum y-dimension of a grid
     * 
     * @return int 
     */
    int maxGridDimY();

    /**
     * @brief Maximum z-dimension of a grid
     * 
     * @return int 
     */
    int maxGridDimZ();

    /**
     * @brief Maximum amount of shared memory available to a thread block in bytes
     * 
     * @return int 
     */
    int maxSharedMemPerBlock();

    /**
     * @brief Memory available on device for __constant__ variables in a CUDA C kernel in bytes
     * 
     * @return int 
     */
    int maxTotalMemConst();

    /**
     * @brief Warp size in threads
     * 
     * @return int 
     */
    int warpSize();

    /**
     * @brief Maximum number of 32-bit registers available to a thread block
     * 
     * @return int 
     */
    int maxRegsPerBlock();

    /**
     * @brief The typical clock frequency in kilohertz
     * 
     * @return int 
     */
    int clockRate();

    /**
     * @brief Test if the device can concurrently copy memory between host and device while executing a kernel
     * 
     * @return true 
     * @return false 
     */
    bool gpuOverlap();

    /**
     * @brief Number of multiprocessors on the device
     * 
     * @return int 
     */
    int numMultiprocessors();

    /**
     * @brief Test if there is a run time limit for kernels executed on the device
     * 
     * @return true 
     * @return false 
     */
    bool kernelExecTimeout();

    /**
     * @brief Test if device is not restricted and can have multiple CUDA contexts present at a single time
     * 
     * @return true 
     * @return false 
     */
    bool computeModeDefault();

    /**
     * @brief Test if device is prohibited from creating new CUDA contexts
     * 
     * @return true 
     * @return false 
     */
    bool computeModeProhibited();

    /**
     * @brief Test if device can have only one context used by a single process at a time
     * 
     * @return true 
     * @return false 
     */
    bool computeModeExclusive();

    /**
     * @brief PCI device (also known as slot) identifier of the device
     * 
     * @return int 
     */
    int PCIDeviceId();

    /**
     * @brief PCI domain identifier of the device
     * 
     * @return int 
     */
    int PCIDomainId();

    /**
     * @brief Peak memory clock frequency in kilohertz
     * 
     * @return int 
     */
    int clockRateMem();

    /**
     * @brief Global memory bus width in bits
     * 
     * @return int 
     */
    int globalMemBusWidth();

    /**
     * @brief Size of L2 cache in bytes. 0 if the device doesn't have L2 cache
     * 
     * @return int 
     */
    int L2CacheSize();

    /**
     * @brief Maximum resident threads per multiprocessor
     * 
     * @return int 
     */
    int maxThreadsPerMultiprocessor();

    /**
     * @brief Major compute capability version number
     * 
     * @return int 
     */
    int computeCapabilityMajor();

    /**
     * @brief Minor compute capability version number
     * 
     * @return int 
     */
    int computeCapabilityMinor();

    /**
     * @brief Test if device supports caching globals in L1 cache
     * 
     * @return true 
     * @return false 
     */
    bool L1CacheSupportGlobal();

    /**
     * @brief Test if device supports caching locals in L1 cache
     * 
     * @return true 
     * @return false 
     */
    bool L1CacheSupportLocal();

    /**
     * @brief Maximum amount of shared memory available to a multiprocessor in bytes
     * 
     * @return int 
     */
    int maxSharedMemPerMultiprocessor();

    /**
     * @brief Maximum number of 32-bit registers available to a multiprocessor
     * 
     * @return int 
     */
    int maxRegsPerMultiprocessor();

    /**
     * @brief Test if device supports allocating managed memory on this system
     * 
     * @return true 
     * @return false 
     */
    bool managedMem();

    /**
     * @brief Test if device is on a multi-GPU board
     * 
     * @return true 
     * @return false 
     */
    bool multiGPUBoard();

    /**
     * @brief Unique identifier for a group of devices associated with the same board
     * 
     * @return int 
     */
    int multiGPUBoardGroupId();

private:
    
    void validateAttached();
    static void validateDeviceId(const int &deviceId);
};


/**
 * @brief Mechanica CUDA interface
 * 
 */
struct MxCUDA {
    /**
     * @brief Initialize CUDA
     * 
     */
    static void init();

    static void setGLDevice(const int &deviceId);

    /**
     * @brief Get the name of a device
     * 
     * @param deviceId 
     * @return std::string 
     */
    static std::string getDeviceName(const int &deviceId);

    /**
     * @brief Get the total memory of device
     * 
     * @param deviceId 
     * @return size_t 
     */
    static size_t getDeviceTotalMem(const int &deviceId);

    /**
     * @brief Get the attribute value of a device
     * 
     * @param deviceId 
     * @param attrib 
     * @return int 
     */
    static int getDeviceAttribute(const int &deviceId, const int &attrib);

    /**
     * @brief Get number of available compute-capable devices
     * 
     * @return int 
     */
    static int getNumDevices();

    /**
     * @brief Get the PCI bus id of a device
     * 
     * @param deviceId 
     * @return std::string 
     */
    static std::string getDevicePCIBusId(const int &deviceId);

    /**
     * @brief Get the device id of the current context of the calling CPU thread. 
     * 
     * @return int 
     */
    static int getCurrentDevice();

    /**
     * @brief Maximum number of threads per block
     * 
     * @return int 
     */
    static int maxThreadsPerBlock(const int &deviceId);

    /**
     * @brief Maximum x-dimension of a block
     * 
     * @return int 
     */
    static int maxBlockDimX(const int &deviceId);

    /**
     * @brief Maximum y-dimension of a block
     * 
     * @return int 
     */
    static int maxBlockDimY(const int &deviceId);

    /**
     * @brief Maximum z-dimension of a block
     * 
     * @return int 
     */
    static int maxBlockDimZ(const int &deviceId);

    /**
     * @brief Maximum x-dimension of a grid
     * 
     * @return int 
     */
    static int maxGridDimX(const int &deviceId);

    /**
     * @brief Maximum y-dimension of a grid
     * 
     * @return int 
     */
    static int maxGridDimY(const int &deviceId);

    /**
     * @brief Maximum z-dimension of a grid
     * 
     * @return int 
     */
    static int maxGridDimZ(const int &deviceId);

    /**
     * @brief Maximum amount of shared memory available to a thread block in bytes
     * 
     * @return int 
     */
    static int maxSharedMemPerBlock(const int &deviceId);

    /**
     * @brief Memory available on device for __constant__ variables in a CUDA C kernel in bytes
     * 
     * @return int 
     */
    static int maxTotalMemConst(const int &deviceId);

    /**
     * @brief Warp size in threads
     * 
     * @return int 
     */
    static int warpSize(const int &deviceId);

    /**
     * @brief Maximum number of 32-bit registers available to a thread block
     * 
     * @return int 
     */
    static int maxRegsPerBlock(const int &deviceId);

    /**
     * @brief The typical clock frequency in kilohertz
     * 
     * @return int 
     */
    static int clockRate(const int &deviceId);

    /**
     * @brief Test if the device can concurrently copy memory between host and device while executing a kernel
     * 
     * @return true 
     * @return false 
     */
    static bool gpuOverlap(const int &deviceId);

    /**
     * @brief Number of multiprocessors on the device
     * 
     * @return int 
     */
    static int numMultiprocessors(const int &deviceId);

    /**
     * @brief Test if there is a run time limit for kernels executed on the device
     * 
     * @return true 
     * @return false 
     */
    static bool kernelExecTimeout(const int &deviceId);

    /**
     * @brief Test if device is not restricted and can have multiple CUDA contexts present at a single time
     * 
     * @return true 
     * @return false 
     */
    static bool computeModeDefault(const int &deviceId);

    /**
     * @brief Test if device is prohibited from creating new CUDA contexts
     * 
     * @return true 
     * @return false 
     */
    static bool computeModeProhibited(const int &deviceId);

    /**
     * @brief Test if device can have only one context used by a single process at a time
     * 
     * @return true 
     * @return false 
     */
    static bool computeModeExclusive(const int &deviceId);

    /**
     * @brief PCI device (also known as slot) identifier of the device
     * 
     * @return int 
     */
    static int PCIDeviceId(const int &deviceId);

    /**
     * @brief PCI domain identifier of the device
     * 
     * @return int 
     */
    static int PCIDomainId(const int &deviceId);

    /**
     * @brief Peak memory clock frequency in kilohertz
     * 
     * @return int 
     */
    static int clockRateMem(const int &deviceId);

    /**
     * @brief Global memory bus width in bits
     * 
     * @return int 
     */
    static int globalMemBusWidth(const int &deviceId);

    /**
     * @brief Size of L2 cache in bytes. 0 if the device doesn't have L2 cache
     * 
     * @return int 
     */
    static int L2CacheSize(const int &deviceId);

    /**
     * @brief Maximum resident threads per multiprocessor
     * 
     * @return int 
     */
    static int maxThreadsPerMultiprocessor(const int &deviceId);

    /**
     * @brief Major compute capability version number
     * 
     * @return int 
     */
    static int computeCapabilityMajor(const int &deviceId);

    /**
     * @brief Minor compute capability version number
     * 
     * @return int 
     */
    static int computeCapabilityMinor(const int &deviceId);

    /**
     * @brief Test if device supports caching globals in L1 cache
     * 
     * @return true 
     * @return false 
     */
    static bool L1CacheSupportGlobal(const int &deviceId);

    /**
     * @brief Test if device supports caching locals in L1 cache
     * 
     * @return true 
     * @return false 
     */
    static bool L1CacheSupportLocal(const int &deviceId);

    /**
     * @brief Maximum amount of shared memory available to a multiprocessor in bytes
     * 
     * @return int 
     */
    static int maxSharedMemPerMultiprocessor(const int &deviceId);

    /**
     * @brief Maximum number of 32-bit registers available to a multiprocessor
     * 
     * @return int 
     */
    static int maxRegsPerMultiprocessor(const int &deviceId);

    /**
     * @brief Test if device supports allocating managed memory on this system
     * 
     * @return true 
     * @return false 
     */
    static bool managedMem(const int &deviceId);

    /**
     * @brief Test if device is on a multi-GPU board
     * 
     * @return true 
     * @return false 
     */
    static bool multiGPUBoard(const int &deviceId);

    /**
     * @brief Unique identifier for a group of devices associated with the same board
     * 
     * @return int 
     */
    static int multiGPUBoardGroupId(const int &deviceId);

    /**
     * @brief Tests JIT-compiled program execution and deployment. 
     * 
     * Enable logger at debug level to see step-by-step report. 
     * 
     * @param numBlocks number of blocks
     * @param numThreads number of threads
     * @param numEls number of elements in calculations
     * @param deviceId ID of CUDA device
     */
    static void test(const int &numBlocks, const int &numThreads, const int &numEls, const int &deviceId=0);
};


/**
 * @brief Returns the path to the installed Mechanica include directory
 * 
 * @return std::string 
 */
std::string MxIncludePath();

/**
 * @brief Returns the path to the installed Mechanica private include directory
 * 
 * @return std::string 
 */
std::string MxPrivateIncludePath();

/**
 * @brief Returns the path to the installed Mechanica CUDA resources directory
 * 
 * @return std::string 
 */
std::string MxCUDAPath();

/**
 * @brief Returns the path to the installed CUDA include directory
 * 
 * @return std::string 
 */
std::string MxCUDAIncludePath();

/**
 * @brief Returns an absolute path to a subdirectory of the install Mechanica CUDA resources directory
 * 
 * @param relativePath 
 * @return std::string 
 */
std::string MxCUDAResourcePath(const std::string &relativePath);

/**
 * @brief Returns the relative path to the installed Mechanica CUDA PTX object directory. 
 * 
 * The path is relative to the Mechanica CUDA resources directory, and depends on the build type of the installation. 
 * 
 * @return std::string 
 */
std::string MxCUDAPTXObjectRelPath();

/**
 * @brief Returns the supported CUDA architectures of the installation. 
 * 
 * @return std::vector<std::string> 
 */
std::vector<std::string> MxCUDAArchs();


/**
 * C API
 */


// TODO: finish proper C API for mx_cuda

/**
 * @brief Returns the path to the installed Mechanica include directory
 * 
 */
CAPI_FUNC(HRESULT) MxIncludePath(char *includePath);

/**
 * @brief Returns the path to the installed Mechanica private include directory
 * 
 */
CAPI_FUNC(HRESULT) MxPrivateIncludePath(char *pIncludePath);

/**
 * @brief Returns the path to the installed Mechanica CUDA resources directory
 * 
 */
CAPI_FUNC(HRESULT) MxCUDAPath(char *cudaPath);

/**
 * @brief Returns the path to the installed CUDA include directory
 * 
 */
CAPI_FUNC(HRESULT) MxCUDAIncludePath(char *includePath);

/**
 * @brief Returns an absolute path to a subdirectory of the install Mechanica CUDA resources directory
 * 
 */
CAPI_FUNC(HRESULT) MxCUDAResourcePath(const char *relativePath, char *resourcePath);

/**
 * @brief Returns the relative path to the installed Mechanica CUDA PTX object directory. 
 * 
 */
CAPI_FUNC(HRESULT) MxCUDAPTXObjectRelPath(char *relPath);

/**
 * @brief Returns the supported CUDA architectures of the installation. 
 * 
 */
CAPI_FUNC(HRESULT) MxCUDAArchs(char *cudaArchs);

#endif // _INCLUDE_MX_CUDA_H_
