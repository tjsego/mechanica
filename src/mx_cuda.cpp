/**
 * @file mx_cuda.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines CUDA support and utilities in Mechanica runtime
 * @date 2021-11-09
 * 
 */
#include <mx_cuda.h>
#include <mx_config.h>

#include <MxLogger.h>

#include <cstdarg>
#include <filesystem>
#include <iostream>
#include <fstream>


#define MX_CUDA_TMP_CURRENTCTX(instr)   \
    bool ic = this->attached;           \
    if(!ic) this->pushCurrent();        \
    instr                               \
    if(!ic) this->popCurrent();



// MxCUDARTSource



MxCUDARTSource::MxCUDARTSource(const char *filePath, const char *_name) :
    name{_name}
{
    std::ifstream mx_cuda_ifs(filePath);
    if(!mx_cuda_ifs || !mx_cuda_ifs.good() || mx_cuda_ifs.fail()) mx_exp(std::runtime_error(std::string("Error loading MxCUDART: ") + _name));

    std::string mx_cuda_s((std::istreambuf_iterator<char>(mx_cuda_ifs)), (std::istreambuf_iterator<char>()));
    mx_cuda_ifs.close();

    Log(LOG_INFORMATION) << "Loaded source: " << filePath;

    this->source = mx_cuda_s;
}

const char *MxCUDARTSource::c_str() const {
    return this->source.c_str();
}


// MxCUDARTProgram


MxCUDARTProgram::MxCUDARTProgram() :
    prog{NULL}, 
    ptx{NULL}, 
    is_compute{true}
{
    cuInit(0);
}

MxCUDARTProgram::~MxCUDARTProgram() {
    if(this->prog) {
        MX_NVRTC_CALL(nvrtcDestroyProgram(this->prog));
        delete this->prog;
        this->prog = NULL;
    }
    if(this->ptx) {
        delete this->ptx;
        this->ptx = NULL;
    }
}

void MxCUDARTProgram::addOpt(const std::string &opt) {
    if(this->prog) mx_exp(std::logic_error("Program already compiled."));

    this->opts.push_back(opt);
}

void MxCUDARTProgram::addIncludePath(const std::string &ipath) {
    if(this->prog) mx_exp(std::logic_error("Program already compiled."));

    this->includePaths.push_back(ipath);
}

void MxCUDARTProgram::addNamedExpr(const std::string &namedExpr) {
    if(this->prog) mx_exp(std::logic_error("Program already compiled."));

    this->namedExprs.push_back(namedExpr);
}

void MxCUDARTProgram::compile(const char *src, const char *name, int numHeaders, const char *const *headers, const char *const *includeNames) {
    if(this->prog) mx_exp(std::logic_error("Program already compiled."));

    std::vector<std::string> _opts(this->opts);
    if(this->is_compute) _opts.push_back("--gpu-architecture=compute_" + std::to_string(this->arch));
    else _opts.push_back("--gpu-architecture=sm_" + std::to_string(this->arch));

    std::string _includeOpt = "--include-path=";
    _opts.push_back(_includeOpt + MxCUDAIncludePath());
    _opts.push_back(_includeOpt + MxIncludePath());
    _opts.push_back(_includeOpt + MxPrivateIncludePath());
    for(auto &s : this->includePaths) _opts.push_back(_includeOpt + s);
    
    #ifdef MX_CUDA_DEBUG

    _opts.push_back("--device-debug");
    _opts.push_back("--generate-line-info");
    _opts.push_back("--display-error-number");

    #endif

    char **charOpts = new char*[_opts.size()];
    for(int i = 0; i < _opts.size(); i++) {
        charOpts[i] = const_cast<char*>(_opts[i].c_str());

        Log(LOG_INFORMATION) << "Got compile option: " << std::string(charOpts[i]);
    }

    this->prog = new nvrtcProgram();
    MX_NVRTC_CALL(nvrtcCreateProgram(this->prog, src, name, numHeaders, headers, includeNames));

    for(auto ne : this->namedExprs) MX_NVRTC_CALL(nvrtcAddNameExpression(*this->prog, ne.c_str()));

    auto compileResult = nvrtcCompileProgram(*this->prog, _opts.size(), charOpts);
    // Dump log on compile failure
    if(compileResult != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(*this->prog, &logSize);
        char *log = new char[logSize];
        nvrtcGetProgramLog(*this->prog, log);
        std::cout << log << std::endl;
        delete[] log;
    }
    MX_NVRTC_CALL(compileResult);

    size_t ptxSize;
    MX_NVRTC_CALL(nvrtcGetPTXSize(*this->prog, &ptxSize));
    this->ptx = new char[ptxSize];
    MX_NVRTC_CALL(nvrtcGetPTX(*this->prog, this->ptx));
}

std::string MxCUDARTProgram::loweredName(const std::string namedExpr) {
    if(!this->prog) mx_exp(std::logic_error("Program not compiled."));

    const char *name;
    MX_NVRTC_CALL(nvrtcGetLoweredName(*this->prog, namedExpr.c_str(), &name));

    if(!name) return "";

    Log(LOG_DEBUG) << namedExpr << " -> " << name;

    return std::string(name);
}


// MxCUDAFunction


MxCUDAFunction::MxCUDAFunction(const std::string &name, MxCUDAContext *context) : 
    name{name}, 
    gridDimX{1}, gridDimY{1}, gridDimZ{1}, 
    blockDimX{1}, blockDimY{1}, blockDimZ{1}, 
    sharedMemBytes{0}, hStream{NULL}, extra{NULL}, 
    context{context}
{
    if(!context || !context->module) mx_exp(std::logic_error("No loaded programs."));

    this->function = new CUfunction();
    MX_CUDA_CALL(cuModuleGetFunction(this->function, *context->module, name.c_str()));
}

MxCUDAFunction::~MxCUDAFunction() {
    if(this->function) {
        delete this->function;
        this->function = 0;
    }
}

HRESULT MxCUDAFunction::autoConfig(const unsigned int &_nr_arrayElems, 
                                   size_t dynamicSMemSize, 
                                   size_t (*blockSizeToDynamicSMemSize)(int), 
                                   int blockSizeLimit)
{
    int minGridSize;
    int blockSize;
    MX_CUDA_CALL(cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, *this->function, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit));

    if(blockSize == 0) {
        Log(LOG_ERROR) << "Auto-config failed!";

        return E_FAIL;
    }
    this->blockDimX = blockSize;
    this->gridDimX = (_nr_arrayElems + blockSize - 1) / blockSize;
    this->sharedMemBytes = dynamicSMemSize;
    
    Log(LOG_INFORMATION) << "CUDA function " << this->name << " configured...";
    Log(LOG_INFORMATION) << "... array elements : " << _nr_arrayElems;
    if(blockSizeLimit > 0) { Log(LOG_INFORMATION) << "... maximum threads: " << blockSizeLimit; }
    Log(LOG_INFORMATION) << "... block size     : " << this->blockDimX << " threads";
    Log(LOG_INFORMATION) << "... grid size      : " << this->gridDimX << " blocks";
    Log(LOG_INFORMATION) << "... shared memory  : " << this->sharedMemBytes << " B/block";

    return S_OK;
}

void MxCUDAFunction::operator()(void **args) {
    MX_CUDA_CALL(cuLaunchKernel(
        *this->function, 
        this->gridDimX, this->gridDimY, this->gridDimZ, 
        this->blockDimX, this->blockDimY, this->blockDimZ, 
        this->sharedMemBytes, this->hStream, args, this->extra
    ));
}

void MxCUDAFunction::operator()(int nargs, ...) 
{
    void **args;
    args = (void**)malloc(nargs * sizeof(void*));
    std::va_list argsList;
    va_start(argsList, nargs);
    for(int i = 0; i < nargs; i++) args[i] = va_arg(argsList, void*);
    va_end(argsList);

    (*this)(args);
}


// MxCUDAContext


MxCUDAContext::MxCUDAContext(CUdevice device) : 
    context{NULL}, 
    device{device}, 
    module{NULL}
{
    this->context = new CUcontext();
    MX_CUDA_CALL(cuCtxCreate(this->context, 0, this->device));
    this->attached = true;
}

MxCUDAContext::~MxCUDAContext() {
    if(this->context) this->destroy();
}

void MxCUDAContext::addOpt(CUjit_option opt, void *val) {
    this->compileOpts.push_back(opt);
    this->compileOptVals.push_back(val);
}

void MxCUDAContext::loadProgram(const MxCUDARTProgram &prog) {
    if(!prog.ptx) mx_exp(std::logic_error("Program not compiled."));

    this->loadPTX(prog.ptx);
}

void MxCUDAContext::loadPTX(const char *ptx) {
    if(!this->module) this->module = new CUmodule();

    #ifdef MX_CUDA_DEBUG
    this->addOpt(CU_JIT_GENERATE_DEBUG_INFO, (void*)(size_t)1);
    #endif

    size_t numJitOpts = this->compileOpts.size();
    CUjit_option *jitopts;
    void **jitoptvals;
    if(numJitOpts == 0) {
        jitopts = 0;
        jitoptvals = 0;
    }
    else {
        jitopts = new CUjit_option[numJitOpts];
        jitoptvals = new void*[numJitOpts];
        for(int i = 0; i < numJitOpts; i++) {
            jitopts[i] = this->compileOpts[i];
            jitoptvals[i] = this->compileOptVals[i];

            Log(LOG_INFORMATION) << "Got JIT compile option: " << jitopts[i] << ", " << jitoptvals[i];
        }
    }

    MX_CUDA_TMP_CURRENTCTX(
        MX_CUDA_CALL(cuModuleLoadDataEx(this->module, ptx, numJitOpts, jitopts, jitoptvals));
    )

    if(numJitOpts > 0) {
        delete[] jitopts;
        delete[] jitoptvals;
    }
}

MxCUDAFunction *MxCUDAContext::getFunction(const char *name) {
    if(!this->module) mx_exp(std::logic_error("No loaded programs."));

    return new MxCUDAFunction(name, this);
}

CUdeviceptr *MxCUDAContext::getGlobal(const char *name) {
    if(!this->module) mx_exp(std::logic_error("No loaded programs."));

    CUdeviceptr *dptr; 
    size_t bytes;
    MX_CUDA_CALL(cuModuleGetGlobal(dptr, &bytes, *this->module, name));
    return dptr;
}

size_t MxCUDAContext::getGlobalSize(const char *name) {
    if(!this->module) mx_exp(std::logic_error("No loaded programs."));

    CUdeviceptr *dptr;
    size_t bytes;
    MX_CUDA_CALL(cuModuleGetGlobal(dptr, &bytes, *this->module, name));
    return bytes;
}

void MxCUDAContext::pushCurrent() {
    if(this->attached) mx_exp(std::logic_error("Context already attached."));

    MX_CUDA_CALL(cuCtxPushCurrent(*this->context));
    this->attached = true;
}

CUcontext *MxCUDAContext::popCurrent() {
    if(!this->attached) mx_exp(std::logic_error("Context not attached."));

    CUcontext *cu;
    MX_CUDA_CALL(cuCtxPopCurrent(cu));
    this->attached = false;
    return cu;
}

void MxCUDAContext::destroy() {
    if(!this->context) mx_exp(std::logic_error("No context to destroy."));

    if(this->module) {
        MX_CUDA_TMP_CURRENTCTX(
            MX_CUDA_CALL(cuModuleUnload(*this->module));
        )
        delete this->module;
        this->module = NULL;
    }

    MX_CUDA_CALL(cuCtxDestroy(*this->context));
    delete this->context;
    this->context = NULL;
    this->attached = false;
}

int MxCUDAContext::getAPIVersion() {
    if(!this->context) mx_exp(std::logic_error("No context."));

    unsigned int v;
    MX_CUDA_CALL(cuCtxGetApiVersion(*this->context, &v));
    return v;
}

void MxCUDAContext::sync() {
    MX_CUDA_CALL(cuCtxSynchronize());
}


// MxCUDADevice


MxCUDADevice::MxCUDADevice() :
    device{NULL}
{
    cuInit(0);
}

MxCUDADevice::~MxCUDADevice() {
    if(this->device != NULL) this->detachDevice();
}

void MxCUDADevice::attachDevice(const int &deviceId) {
    MxCUDADevice::validateDeviceId(deviceId);

    if(this->device != NULL) mx_exp(std::logic_error("Device already attached."));

    this->device = new int();
    MX_CUDA_CALL(cuDeviceGet(this->device, deviceId));
}

void MxCUDADevice::detachDevice() {
    this->validateAttached();
    this->device = NULL;
}

std::string MxCUDADevice::name() {
    this->validateAttached();
    return MxCUDADevice::getDeviceName(*this->device);
}

int MxCUDADevice::arch() {
    this->validateAttached();
    int vmaj = this->computeCapabilityMajor();
    int vmin = this->computeCapabilityMinor();
    return vmaj * 10 + vmin;
}

size_t MxCUDADevice::totalMem() {
    this->validateAttached();
    return MxCUDADevice::getDeviceTotalMem(*this->device);
}

int MxCUDADevice::getAttribute(const int &attrib) {
    this->validateAttached();
    return MxCUDADevice::getDeviceAttribute(*this->device, attrib);
}

std::string MxCUDADevice::PCIBusId() {
    this->validateAttached();
    return MxCUDADevice::getDevicePCIBusId(*this->device);
}

MxCUDAContext *MxCUDADevice::createContext() {
    this->validateAttached();
    return new MxCUDAContext(*this->device);
}

MxCUDAContext *MxCUDADevice::currentContext() {
    this->validateAttached();
    
    MxCUDAContext *mxContext;
    
    CUcontext *context = new CUcontext();
    if(MX_CUDA_CALL(cuCtxGetCurrent(context)) != CUDA_SUCCESS) {
        return NULL;
    }

    if(context == NULL) { 
        Log(LOG_TRACE);

        delete context;

        mxContext = new MxCUDAContext(*this->device); 
    }
    else {
        Log(LOG_TRACE);

        mxContext = new MxCUDAContext();
        mxContext->context = context;
        mxContext->device = *this->device;
        mxContext->attached = true;
    }
    return mxContext;
}

std::string MxCUDADevice::getDeviceName(const int &deviceId) {
    MxCUDADevice::validateDeviceId(deviceId);

    size_t nameLen = 256;
    char name[nameLen];
    MX_CUDA_CALL(cuDeviceGetName(name, nameLen, deviceId));
    return std::string(name);
}

size_t MxCUDADevice::getDeviceTotalMem(const int &deviceId) {
    MxCUDADevice::validateDeviceId(deviceId);

    size_t size;
    MX_CUDA_CALL(cuDeviceTotalMem(&size, deviceId));
    return size;
}

int MxCUDADevice::getDeviceAttribute(const int &deviceId, const int &attrib) {
    MxCUDADevice::validateDeviceId(deviceId);

    int pi;
    CUdevice_attribute attr = (CUdevice_attribute)attrib;
    MX_CUDA_CALL(cuDeviceGetAttribute(&pi, attr, deviceId));
    return pi;
}

int MxCUDADevice::getNumDevices() {
    int count;
    MX_CUDA_CALL(cuDeviceGetCount(&count));
    return count;
}

std::string MxCUDADevice::getDevicePCIBusId(const int &deviceId) {
    MxCUDADevice::validateDeviceId(deviceId);

    char *pciBusId;
    MX_CUDA_CALL(cuDeviceGetPCIBusId(pciBusId, 256, deviceId));
    return pciBusId;
}

int MxCUDADevice::getCurrentDevice() {
    int deviceId;
    MX_CUDA_CALL(cuCtxGetDevice(&deviceId));
    return deviceId;
}

void MxCUDADevice::validateAttached() {
    if(!this->device) mx_exp(std::logic_error("No device attached."));
}

void MxCUDADevice::validateDeviceId(const int &deviceId) {
    if(deviceId < 0 || deviceId > MxCUDADevice::getNumDevices())
        mx_exp(std::range_error("Invalid ID selection."));

    return;
}

int MxCUDADevice::maxThreadsPerBlock() { return MxCUDA::maxThreadsPerBlock(*this->device); }
int MxCUDADevice::maxBlockDimX() { return MxCUDA::maxBlockDimX(*this->device); }
int MxCUDADevice::maxBlockDimY() { return MxCUDA::maxBlockDimY(*this->device); }
int MxCUDADevice::maxBlockDimZ() { return MxCUDA::maxBlockDimZ(*this->device); }
int MxCUDADevice::maxGridDimX() { return MxCUDA::maxGridDimX(*this->device); }
int MxCUDADevice::maxGridDimY() { return MxCUDA::maxGridDimY(*this->device); }
int MxCUDADevice::maxGridDimZ() { return MxCUDA::maxGridDimZ(*this->device); }
int MxCUDADevice::maxSharedMemPerBlock() { return MxCUDA::maxSharedMemPerBlock(*this->device); }
int MxCUDADevice::maxTotalMemConst() { return MxCUDA::maxTotalMemConst(*this->device); }
int MxCUDADevice::warpSize() { return MxCUDA::warpSize(*this->device); }
int MxCUDADevice::maxRegsPerBlock() { return MxCUDA::maxRegsPerBlock(*this->device); }
int MxCUDADevice::clockRate() { return MxCUDA::clockRate(*this->device); }
bool MxCUDADevice::gpuOverlap() { return MxCUDA::gpuOverlap(*this->device); }
int MxCUDADevice::numMultiprocessors() { return MxCUDA::numMultiprocessors(*this->device); }
bool MxCUDADevice::kernelExecTimeout() { return MxCUDA::kernelExecTimeout(*this->device); }
bool MxCUDADevice::computeModeDefault() { return MxCUDA::computeModeDefault(*this->device); }
bool MxCUDADevice::computeModeProhibited() { return MxCUDA::computeModeProhibited(*this->device); }
bool MxCUDADevice::computeModeExclusive() { return MxCUDA::computeModeExclusive(*this->device); }
int MxCUDADevice::PCIDeviceId() { return MxCUDA::PCIDeviceId(*this->device); }
int MxCUDADevice::PCIDomainId() { return MxCUDA::PCIDomainId(*this->device); }
int MxCUDADevice::clockRateMem() { return MxCUDA::clockRateMem(*this->device); }
int MxCUDADevice::globalMemBusWidth() { return MxCUDA::globalMemBusWidth(*this->device); }
int MxCUDADevice::L2CacheSize() { return MxCUDA::L2CacheSize(*this->device); }
int MxCUDADevice::maxThreadsPerMultiprocessor() { return MxCUDA::maxThreadsPerMultiprocessor(*this->device); }
int MxCUDADevice::computeCapabilityMajor() { return MxCUDA::computeCapabilityMajor(*this->device); }
int MxCUDADevice::computeCapabilityMinor() { return MxCUDA::computeCapabilityMinor(*this->device); }
bool MxCUDADevice::L1CacheSupportGlobal() { return MxCUDA::L1CacheSupportGlobal(*this->device); }
bool MxCUDADevice::L1CacheSupportLocal() { return MxCUDA::L1CacheSupportLocal(*this->device); }
int MxCUDADevice::maxSharedMemPerMultiprocessor() { return MxCUDA::maxSharedMemPerMultiprocessor(*this->device); }
int MxCUDADevice::maxRegsPerMultiprocessor() { return MxCUDA::maxRegsPerMultiprocessor(*this->device); }
bool MxCUDADevice::managedMem() { return MxCUDA::managedMem(*this->device); }
bool MxCUDADevice::multiGPUBoard() { return MxCUDA::multiGPUBoard(*this->device); }
int MxCUDADevice::multiGPUBoardGroupId() { return MxCUDA::multiGPUBoardGroupId(*this->device); }


// MxCUDA


void MxCUDA::init() {
    cuInit(0);
}

std::string MxCUDA::getDeviceName(const int &deviceId) {
    return MxCUDADevice::getDeviceName(deviceId);
}

size_t MxCUDA::getDeviceTotalMem(const int &deviceId) {
    return MxCUDADevice::getDeviceTotalMem(deviceId);
}

int MxCUDA::getDeviceAttribute(const int &deviceId, const int &attrib) {
    return MxCUDADevice::getDeviceAttribute(deviceId, attrib);
}

int MxCUDA::getNumDevices() {
    return MxCUDADevice::getNumDevices();
}

std::string MxCUDA::getDevicePCIBusId(const int &deviceId) {
    return MxCUDADevice::getDevicePCIBusId(deviceId);
}

int MxCUDA::getCurrentDevice() {
    return MxCUDADevice::getCurrentDevice();
}

int MxCUDA::maxThreadsPerBlock(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK); }
int MxCUDA::maxBlockDimX(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X); }
int MxCUDA::maxBlockDimY(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y); }
int MxCUDA::maxBlockDimZ(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z); }
int MxCUDA::maxGridDimX(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X); }
int MxCUDA::maxGridDimY(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y); }
int MxCUDA::maxGridDimZ(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z); }
int MxCUDA::maxSharedMemPerBlock(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK); }
int MxCUDA::maxTotalMemConst(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY); }
int MxCUDA::warpSize(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_WARP_SIZE); }
int MxCUDA::maxRegsPerBlock(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK); }
int MxCUDA::clockRate(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_CLOCK_RATE); }
bool MxCUDA::gpuOverlap(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP); }
int MxCUDA::numMultiprocessors(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT); }
bool MxCUDA::kernelExecTimeout(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT); }
bool MxCUDA::computeModeDefault(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) == CU_COMPUTEMODE_DEFAULT; }
bool MxCUDA::computeModeProhibited(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) == CU_COMPUTEMODE_PROHIBITED; }
bool MxCUDA::computeModeExclusive(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) == CU_COMPUTEMODE_EXCLUSIVE_PROCESS; }
int MxCUDA::PCIDeviceId(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID); }
int MxCUDA::PCIDomainId(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID); }
int MxCUDA::clockRateMem(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE); }
int MxCUDA::globalMemBusWidth(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH); }
int MxCUDA::L2CacheSize(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE); }
int MxCUDA::maxThreadsPerMultiprocessor(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR); }
int MxCUDA::computeCapabilityMajor(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR); }
int MxCUDA::computeCapabilityMinor(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR); }
bool MxCUDA::L1CacheSupportGlobal(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED); }
bool MxCUDA::L1CacheSupportLocal(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED); }
int MxCUDA::maxSharedMemPerMultiprocessor(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR); }
int MxCUDA::maxRegsPerMultiprocessor(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR); }
bool MxCUDA::managedMem(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY); }
bool MxCUDA::multiGPUBoard(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD); }
int MxCUDA::multiGPUBoardGroupId(const int &deviceId) { return MxCUDA::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID); }

const char *test_program = "                                                \n\
extern \"C\" __global__ void gpu_test(int len, int *vals, int *result) {    \n\
    int i = blockIdx.x * blockDim.x + threadIdx.x;                          \n\
    if (i >= len) return;                                                   \n\
                                                                            \n\
    int n = 0;                                                              \n\
    int ri = vals[i];                                                       \n\
    for (int j = 0; j < len; ++j)                                           \n\
        if (vals[j] == ri) n++;                                             \n\
    result[i] = n;                                                          \n\
}                                                                           \n";

void MxCUDA::test(const int &numBlocks, const int &numThreads, const int &numEls, const int &deviceId) {
    MxCUDA::init();

    Log(LOG_DEBUG) << "*****************************";
    Log(LOG_DEBUG) << "Starting Mechanica CUDA test";
    Log(LOG_DEBUG) << "*****************************";

    Log(LOG_DEBUG) << "Initializing device...";
    
    MxCUDADevice device;
    device.attachDevice(deviceId);
    auto ctx = device.createContext();

    int cc_major = device.computeCapabilityMajor();
    int cc_minor = device.computeCapabilityMinor();
    int arch = cc_major * 10 + cc_minor;

    Log(LOG_DEBUG) << "     Number of devices               : " << MxCUDA::getNumDevices();
    Log(LOG_DEBUG) << "     Name of device                  : " << device.name();
    Log(LOG_DEBUG) << "     Compute capability of device    : " << cc_major << "." << cc_minor;
    Log(LOG_DEBUG) << "     Number of threads per block     : " << device.maxThreadsPerBlock();

    Log(LOG_DEBUG) << "JIT compiling program...";

    MxCUDARTProgram prog;
    prog.arch = device.arch();
    prog.compile(test_program, "mx_test_program.cu");

    Log(LOG_DEBUG) << "Loading program...";

    ctx->loadProgram(prog);

    Log(LOG_DEBUG) << "Preparing work...";

    size_t bufferSize = numEls * sizeof(float);
    int *vals = (int*)malloc(bufferSize);
    for (int i = 0; i < numEls; ++i) vals[i] = numEls % (i + 1);

    CUstream stream;
    Log(LOG_DEBUG) << "    ... " << MX_CUDA_CALL(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CUdeviceptr vals_d;
    Log(LOG_DEBUG) << "    ... " << MX_CUDA_CALL(cuMemAlloc(&vals_d, bufferSize));
    Log(LOG_DEBUG) << "    ... " << MX_CUDA_CALL(cuMemcpyHtoDAsync(vals_d, vals, bufferSize, stream));
    
    int *results = (int*)malloc(bufferSize);
    CUdeviceptr results_d;
    Log(LOG_DEBUG) << "    ... " << MX_CUDA_CALL(cuMemAlloc(&results_d, bufferSize));
    int n = numEls;

    Log(LOG_DEBUG) << "Optimizing kernel...";

    auto f = ctx->getFunction("gpu_test");
    if(f->autoConfig(numEls) != S_OK) {
        Log(LOG_DEBUG) << "    ... failed!" << std::endl;
        f->gridDimX = numBlocks;
        f->blockDimX = numThreads;
    }
    else { Log(LOG_DEBUG) << "    ... done!" << std::endl; }

    Log(LOG_DEBUG) << "Doing work...";

    Log(LOG_DEBUG) << "    ... " << MX_CUDA_CALL(cuStreamSynchronize(stream));
    void *args[] = {&n, &vals_d, &results_d};
    (*f)(args);
    ctx->sync();

    Log(LOG_DEBUG) << "Retrieving results... " << MX_CUDA_CALL(cuMemcpyDtoHAsync(results, results_d, bufferSize, stream));

    Log(LOG_DEBUG) << "Cleaning up...";
    
    Log(LOG_DEBUG) << "    ... " << MX_CUDA_CALL(cuMemFree(vals_d));
    Log(LOG_DEBUG) << "    ... " << MX_CUDA_CALL(cuMemFree(results_d));
    free(vals);
    free(results);
    Log(LOG_DEBUG) << "    ... " << MX_CUDA_CALL(cuStreamSynchronize(stream));
    Log(LOG_DEBUG) << "    ... " << MX_CUDA_CALL(cuStreamDestroy(stream));
    delete f;
    ctx->destroy();
    device.detachDevice();

    Log(LOG_DEBUG) << "*****************************";
    Log(LOG_DEBUG) << "Completed Mechanica CUDA test";
    Log(LOG_DEBUG) << "*****************************";
}


// Misc.


std::string MxIncludePath() {
    auto p = std::filesystem::absolute(MX_INCLUDE_DIR);
    return p.string();
}

std::string MxPrivateIncludePath() {
    auto p = std::filesystem::absolute(MX_INCLUDE_DIR);
    p.append("private");
    return p.string();
}

std::string MxCUDAPath() {
    auto p = std::filesystem::absolute(MX_CUDA_DIR);
    return p.string();
}

std::string MxCUDAIncludePath() {
    auto p = std::filesystem::absolute(MX_CUDA_INCLUDE_DIR);
    return p.string();
}

std::string MxCUDAResourcePath(const std::string &relativePath) {
    auto p = std::filesystem::absolute(MX_CUDA_DIR);
    p.append(relativePath);
    return p.string();;
}

std::string MxCUDAPTXObjectRelPath() {
    return std::string("objects-") + std::string(MX_BUILD_TYPE);
}

std::vector<std::string> MxCUDAArchs() {
    char s[] = MX_CUDA_ARCHS;
    char *token = strtok(s, ";");
    std::vector<std::string> result;
    while(token != NULL) {
        result.push_back(std::string(token));
        token = strtok(s, ";");
    }
    return result;
}

HRESULT MxIncludePath(char *includePath) {
    includePath = const_cast<char*>(MxIncludePath().c_str());
    return S_OK;
}

HRESULT MxPrivateIncludePath(char *pIncludePath) {
    pIncludePath = const_cast<char*>(MxPrivateIncludePath().c_str());
    return S_OK;
}

HRESULT MxCUDAPath(char *cudaPath) {
    cudaPath = const_cast<char*>(MxCUDAPath().c_str());
    return S_OK;
}

HRESULT MxCUDAIncludePath(char *includePath) {
    includePath =const_cast<char*>(MxCUDAIncludePath().c_str());
    return S_OK;
}

HRESULT MxCUDAResourcePath(const char *relativePath, char *resourcePath) {
    resourcePath = const_cast<char*>(MxCUDAResourcePath(relativePath).c_str());
    return S_OK;
}

HRESULT MxCUDAPTXObjectRelPath(char *relPath) {
    relPath = const_cast<char*>(MxCUDAPTXObjectRelPath().c_str());
    return S_OK;
}

HRESULT MxCUDAArchs(char *cudaArchs) {
    char c[] = MX_CUDA_ARCHS;
    *cudaArchs = *c;
    return S_OK;
}