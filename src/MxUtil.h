/*
 * MxParticles.h
 *
 *  Created on: Feb 25, 2017
 *      Author: andy
 */

#ifndef SRC_MXUTIL_H_
#define SRC_MXUTIL_H_

#include "mechanica_private.h"

#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>
#include <bitset>
#include <cycle.h>
#include <string>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <random>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <limits>
#include <type_traits>

typedef std::mt19937 MxRandomType;
MxRandomType &MxRandomEngine();

/**
 * @brief Get the current seed for the pseudo-random number generator
 * 
 */
CAPI_FUNC(unsigned int) getSeed();

/**
 * @brief Set the current seed for the pseudo-random number generator
 * 
 * @param _seed 
 * @return HRESULT 
 */
CAPI_FUNC(HRESULT) setSeed(const unsigned int *_seed=0);

enum class MxPointsType : unsigned int {
    Sphere,
    SolidSphere,
    Disk,
    SolidCube,
    Cube,
    Ring
};

std::vector<std::string> MxColor3_Names();
Magnum::Color3 Color3_Parse(const std::string &str);

template <typename Type, typename Klass>
inline constexpr size_t offset_of(Type Klass::*member) {
    constexpr Klass object {};
    return size_t(&(object.*member)) - size_t(&object);
}

#ifdef _WIN32
// windows
#define MxAligned_Malloc(size,  alignment) _aligned_malloc(size,  alignment)
#define MxAligned_Free(mem) _aligned_free(mem)
#elif __APPLE__
// mac
inline void* MxAligned_Malloc(size_t size, size_t alignment)
{
    enum {
        void_size = sizeof(void*)
    };
    if (!size) {
        return 0;
    }
    if (alignment < void_size) {
        alignment = void_size;
    }
    void* p;
    if (::posix_memalign(&p, alignment, size) != 0) {
        p = 0;
    }
    return p;
}
#define MxAligned_Free(mem) free(mem)
#else
// linux
#define MxAligned_Malloc(size,  alignment) aligned_alloc(alignment,  size)
#define MxAligned_Free(mem) free(mem)
#endif

MxVector3f MxRandomPoint(const MxPointsType &kind, 
                         const float &dr=0, 
                         const float &phi0=0, 
                         const float &phi1=M_PI);

std::vector<MxVector3f> MxRandomPoints(const MxPointsType &kind, 
                                       const int &n=1, 
                                       const float &dr=0, 
                                       const float &phi0=0, 
                                       const float &phi1=M_PI);

std::vector<MxVector3f> MxPoints(const MxPointsType &kind, const int &n=1);

/**
 * @brief Get the coordinates of a uniformly filled cube. 
 * 
 * @param corner1 first corner of cube
 * @param corner2 second corner of cube
 * @param nParticlesX number of particles along x-direction of filling axes (>=2)
 * @param nParticlesY number of particles along y-direction of filling axes (>=2)
 * @param nParticlesZ number of particles along z-direction of filling axes (>=2)
 * @return std::vector<MxVector3f> 
 */
std::vector<MxVector3f> MxFilledCubeUniform(const MxVector3f &corner1, 
                                            const MxVector3f &corner2, 
                                            const int &nParticlesX=2, 
                                            const int &nParticlesY=2, 
                                            const int &nParticlesZ=2);

/**
 * @brief Get the coordinates of a randomly filled cube. 
 * 
 * @param corner1 first corner of cube
 * @param corner2 second corner of cube
 * @param nParticles number of points in the cube
 * @return std::vector<MxVector3f> 
 */
std::vector<MxVector3f> MxFilledCubeRandom(const MxVector3f &corner1, const MxVector3f &corner2, const int &nParticles);

extern const char* MxColor3Names[];

HRESULT Mx_Icosphere(const int subdivisions, float phi0, float phi1,
                     std::vector<MxVector3f> &verts,
                     std::vector<int32_t> &inds);

namespace mx {
    template<class T>
    typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp = 2)
    {
        // the machine epsilon has to be scaled to the magnitude of the values used
        // and multiplied by the desired precision in ULPs (units in the last place)
        return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
        // unless the result is subnormal
        || std::fabs(x-y) < std::numeric_limits<T>::min();
    }
}

enum Mx_InstructionSet : std::int64_t {
    IS_3DNOW              = 1ll << 0,
    IS_3DNOWEXT           = 1ll << 1,
    IS_ABM                = 1ll << 2,
    IS_ADX                = 1ll << 3,
    IS_AES                = 1ll << 4,
    IS_AVX                = 1ll << 5,
    IS_AVX2               = 1ll << 6,
    IS_AVX512CD           = 1ll << 7,
    IS_AVX512ER           = 1ll << 8,
    IS_AVX512F            = 1ll << 9,
    IS_AVX512PF           = 1ll << 10,
    IS_BMI1               = 1ll << 11,
    IS_BMI2               = 1ll << 12,
    IS_CLFSH              = 1ll << 13,
    IS_CMPXCHG16B         = 1ll << 14,
    IS_CX8                = 1ll << 15,
    IS_ERMS               = 1ll << 16,
    IS_F16C               = 1ll << 17,
    IS_FMA                = 1ll << 18,
    IS_FSGSBASE           = 1ll << 19,
    IS_FXSR               = 1ll << 20,
    IS_HLE                = 1ll << 21,
    IS_INVPCID            = 1ll << 23,
    IS_LAHF               = 1ll << 24,
    IS_LZCNT              = 1ll << 25,
    IS_MMX                = 1ll << 26,
    IS_MMXEXT             = 1ll << 27,
    IS_MONITOR            = 1ll << 28,
    IS_MOVBE              = 1ll << 28,
    IS_MSR                = 1ll << 29,
    IS_OSXSAVE            = 1ll << 30,
    IS_PCLMULQDQ          = 1ll << 31,
    IS_POPCNT             = 1ll << 32,
    IS_PREFETCHWT1        = 1ll << 33,
    IS_RDRAND             = 1ll << 34,
    IS_RDSEED             = 1ll << 35,
    IS_RDTSCP             = 1ll << 36,
    IS_RTM                = 1ll << 37,
    IS_SEP                = 1ll << 38,
    IS_SHA                = 1ll << 39,
    IS_SSE                = 1ll << 40,
    IS_SSE2               = 1ll << 41,
    IS_SSE3               = 1ll << 42,
    IS_SSE41              = 1ll << 43,
    IS_SSE42              = 1ll << 44,
    IS_SSE4a              = 1ll << 45,
    IS_SSSE3              = 1ll << 46,
    IS_SYSCALL            = 1ll << 47,
    IS_TBM                = 1ll << 48,
    IS_XOP                = 1ll << 49,
    IS_XSAVE              = 1ll << 50,
};

#if defined(__x86_64__) || defined(_M_X64)

// Yes, Windows has the __cpuid and __cpuidx macros in the #include <intrin.h>
// header file, but it seg-faults when we try to call them from clang.
// this version of the cpuid seems to work with clang on both Windows and mac.

// adapted from https://github.com/01org/linux-sgx/blob/master/common/inc/internal/linux/cpuid_gnu.h
/* This is a PIC-compliant version of CPUID */
static inline void __mx_cpuid(int *eax, int *ebx, int *ecx, int *edx)
{
#if defined(__x86_64__)
    asm("cpuid"
            : "=a" (*eax),
            "=b" (*ebx),
            "=c" (*ecx),
            "=d" (*edx)
            : "0" (*eax), "2" (*ecx));

#else
    /*on 32bit, ebx can NOT be used as PIC code*/
    asm volatile ("xchgl %%ebx, %1; cpuid; xchgl %%ebx, %1"
            : "=a" (*eax), "=r" (*ebx), "=c" (*ecx), "=d" (*edx)
            : "0" (*eax), "2" (*ecx));
#endif
}

#ifdef _WIN32

// TODO: PATHETIC HACK for windows. 
// don't know why, but calling cpuid in release mode, and ONLY in release 
// mode causes a segfault. Hack is to flush stdout, push some junk on the stack. 
// and force a task switch. 
// dont know why this works, but if any of these are not here, then it segfaults
// in release mode. 
// this also seems to work, but force it non-inline and return random
// number. 
// Maybe the optimizer is inlining it, and inlining causes issues
// calling cpuid??? 

static __declspec(noinline) int mx_cpuid(int a[4], int b)
{
    a[0] = b;
    a[2] = 0;
    __mx_cpuid(&a[0], &a[1], &a[2], &a[3]);
    return std::rand();
}

static __declspec(noinline) int mx_cpuidex(int a[4], int b, int c)
{
    a[0] = b;
    a[2] = c;
    __mx_cpuid(&a[0], &a[1], &a[2], &a[3]);
    return std::rand();
}

#else 

static  void mx_cpuid(int a[4], int b)
{
    a[0] = b;
    a[2] = 0;
    __mx_cpuid(&a[0], &a[1], &a[2], &a[3]);
}

static void mx_cpuidex(int a[4], int b, int c)
{
    a[0] = b;
    a[2] = c;
    __mx_cpuid(&a[0], &a[1], &a[2], &a[3]);
}

#endif

             
// InstructionSet.cpp
// Compile by using: cl /EHsc /W4 InstructionSet.cpp
// processor: x86, x64
// Uses the __cpuid intrinsic to get information about
// CPU extended instruction set support.



class CAPI_EXPORT InstructionSet
{

private:
    
    typedef Magnum::Vector4i VectorType;

    class InstructionSet_Internal
    {
    public:
        InstructionSet_Internal() :
                nIds_ { 0 }, nExIds_ { 0 }, isIntel_ { false }, isAMD_ { false }, f_1_ECX_ {
                        0 }, f_1_EDX_ { 0 }, f_7_EBX_ { 0 }, f_7_ECX_ { 0 }, f_81_ECX_ {
                        0 }, f_81_EDX_ { 0 }, data_ { }, extdata_ { }
        {
            VectorType cpui;

            // Calling mx_cpuid with 0x0 as the function_id argument
            // gets the number of the highest valid function ID.
            mx_cpuid(cpui.data(), 0);
            nIds_ = cpui[0];

            for (int i = 0; i <= nIds_; ++i) {
                mx_cpuidex(cpui.data(), i, 0);
                data_.push_back(cpui);
            }

            // Capture vendor string
            char vendor[0x20];
            memset(vendor, 0, sizeof(vendor));
            *reinterpret_cast<int*>(vendor) = data_[0][1];
            *reinterpret_cast<int*>(vendor + 4) = data_[0][3];
            *reinterpret_cast<int*>(vendor + 8) = data_[0][2];
            vendor_ = vendor;
            if (vendor_ == "GenuineIntel") {
                isIntel_ = true;
            } else if (vendor_ == "AuthenticAMD") {
                isAMD_ = true;
            }

            // load bitset with flags for function 0x00000001
            if (nIds_ >= 1) {
                f_1_ECX_ = data_[1][2];
                f_1_EDX_ = data_[1][3];
            }

            // load bitset with flags for function 0x00000007
            if (nIds_ >= 7) {
                f_7_EBX_ = data_[7][1];
                f_7_ECX_ = data_[7][2];
            }

            // Calling mx_cpuid with 0x80000000 as the function_id argument
            // gets the number of the highest valid extended ID.
            mx_cpuid(cpui.data(), 0x80000000);
            nExIds_ = cpui[0];

            char brand[0x40];
            memset(brand, 0, sizeof(brand));

            for (int i = 0x80000000; i <= nExIds_; ++i) {
                mx_cpuidex(cpui.data(), i, 0);
                extdata_.push_back(cpui);
            }

            // load bitset with flags for function 0x80000001
            if (nExIds_ >= 0x80000001) {
                f_81_ECX_ = extdata_[1][2];
                f_81_EDX_ = extdata_[1][3];
            }

            // Interpret CPU brand string if reported
            if (nExIds_ >= 0x80000004) {
                memcpy(brand, extdata_[2].data(), sizeof(cpui));
                memcpy(brand + 16, extdata_[3].data(), sizeof(cpui));
                memcpy(brand + 32, extdata_[4].data(), sizeof(cpui));
                brand_ = brand;
            }
        };


        int nIds_;
        int nExIds_;
        std::string vendor_;
        std::string brand_;
        bool isIntel_;
        bool isAMD_;
        std::bitset<32> f_1_ECX_;
        std::bitset<32> f_1_EDX_;
        std::bitset<32> f_7_EBX_;
        std::bitset<32> f_7_ECX_;
        std::bitset<32> f_81_ECX_;
        std::bitset<32> f_81_EDX_;
        std::vector<VectorType> data_;
        std::vector<VectorType> extdata_;
    };
    
    InstructionSet_Internal CPU_Rep;


public:
    // getters
    std::string Vendor(void);
    std::string Brand(void);

    inline bool SSE3(void);
    inline bool PCLMULQDQ(void);
    inline bool MONITOR(void);
    inline bool SSSE3(void);
    inline bool FMA(void);
    inline bool CMPXCHG16B(void);
    inline bool SSE41(void);
    inline bool SSE42(void);
    inline bool MOVBE(void);
    inline bool POPCNT(void);
    inline bool AES(void);
    inline bool XSAVE(void);
    inline bool OSXSAVE(void);
    inline bool AVX(void);
    inline bool F16C(void);
    inline bool RDRAND(void);
    inline bool MSR(void);
    inline bool CX8(void);
    inline bool SEP(void);
    inline bool CMOV(void);
    inline bool CLFSH(void);
    inline bool MMX(void);
    inline bool FXSR(void);
    inline bool SSE(void);
    inline bool SSE2(void);
    inline bool FSGSBASE(void);
    inline bool BMI1(void);
    inline bool HLE(void);
    inline bool AVX2(void);
    inline bool BMI2(void);
    inline bool ERMS(void);
    inline bool INVPCID(void);
    inline bool RTM(void);
    inline bool AVX512F(void);
    inline bool RDSEED(void);
    inline bool ADX(void);
    inline bool AVX512PF(void);
    inline bool AVX512ER(void);
    inline bool AVX512CD(void);
    inline bool SHA(void);
    inline bool PREFETCHWT1(void);
    inline bool LAHF(void);
    inline bool LZCNT(void);
    inline bool ABM(void);
    inline bool SSE4a(void);
    inline bool XOP(void);
    inline bool TBM(void);
    inline bool SYSCALL(void);
    inline bool MMXEXT(void);
    inline bool RDTSCP(void);
    inline bool _3DNOWEXT(void);
    inline bool _3DNOW(void);

    std::unordered_map<std::string, bool> featuresMap;

    InstructionSet();
};

std::unordered_map<std::string, bool> getFeaturesMap();

#else // #if defined(__x86_64__) || defined(_M_X64)

std::unordered_map<std::string, bool> getFeaturesMap();

#endif // #if defined(__x86_64__) || defined(_M_X64)

class CAPI_EXPORT MxCompileFlags {

    std::unordered_map<std::string, unsigned int> flags;
    std::list<std::string> flagNames;

public:

    MxCompileFlags();
    ~MxCompileFlags() {};

    const std::list<std::string> getFlags();
    const int getFlag(const std::string &_flag);

};


CAPI_FUNC(double) MxWallTime();

CAPI_FUNC(double) MxCPUTime();


class WallTime {
public:
    WallTime();
    ~WallTime();
    double start;
};
    
class PerformanceTimer {
public:
    PerformanceTimer(unsigned id);
    ~PerformanceTimer();
    ticks _start;
    unsigned _id;
};

CAPI_FUNC(uint64_t) MxMath_NextPrime(const uint64_t &start_prime);

std::vector<uint64_t> MxMath_FindPrimes(const uint64_t &start_prime, int n);

CAPI_FUNC(HRESULT) MxMath_FindPrimes(uint64_t start_prime, int n, uint64_t *result);

/**
 * random test function.
 */
std::tuple<Magnum::Vector4, float> _MxTest(const MxVector3f &a, const MxVector3f &b, const MxVector3f &c);


struct Differentiator {
    double (*func)(double);
    double xmin, xmax, inc_cf = 1e-3;

    Differentiator(double (*f)(double), const double &xmin, const double &xmax, const double &inc_cf=1e-3);

    double fnp(const double &x, const unsigned int &order=0);
    double operator() (const double &x);
};


/**
 * @brief Get the unique elements of a vector
 * 
 * @tparam T element type
 * @param vec vector of elements
 * @return std::vector<T> unique elements
 */
template <typename T>
std::vector<T> unique(const std::vector<T> &vec) {
    std::vector<T> result_vec;
    std::unordered_set<T> result_us;

    result_vec.reserve(vec.size());
    
    for(auto f : vec) {
        if(result_us.find(f) == result_us.end()) {
            result_vec.push_back(f);
            result_us.insert(f);
        }
    }

    result_vec.shrink_to_fit();
    return result_vec;
}


#endif /* SRC_MXUTIL_H_ */
