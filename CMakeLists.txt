# need 3.20.3 for CUDA fixes
cmake_minimum_required (VERSION 3.20.3)


if(APPLE)

  message("CMAKE_OSX_DEPLOYMENT_TARGET: ${CMAKE_OSX_DEPLOYMENT_TARGET}")

  if("${CMAKE_OSX_DEPLOYMENT_TARGET}" VERSION_GREATER_EQUAL "10.15")

    message("MacOS newer than 10.15, building universal binary")

    set (CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
    set (CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH NO)

  else()

    message("MacOS older than 10.15, not building universal binary")

  endif()

endif()

set(USING_VCPKG FALSE)
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "")
    message(" [INFO] VCPKG CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
    set(USING_VCPKG TRUE)
endif()

message("VCPKG CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")

project(Mechanica)

# sets the target types
if(UNIX)
  # Both APPLE and UNIX are defined on OSX
  if(APPLE)
    set(MX_APPLE 1)
  else()
    set(MX_LINUX 1)
  endif()
  set(MX_UNIX 1)
elseif(WIN32)
  set(MX_WINDOWS 1)
endif()

# directory where we build the mechanica python package
set(MX_PYPKG_DIR ${CMAKE_BINARY_DIR}/mechanica)

# using magnum, so whole project needs C++11
set(CMAKE_CXX_STANDARD 17)

# we're compiling a bunch of shared libs, all should be pos independent.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# handle rpath
set(CMAKE_MACOSX_RPATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
if(APPLE)
  set(RPATH_PREFIX @loader_path)
else()
  set(RPATH_PREFIX $ORIGIN)
endif()
file(RELATIVE_PATH RDIR
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR} 
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
)
set(CMAKE_INSTALL_RPATH ${RPATH_PREFIX} ${RPATH_PREFIX}/${RDIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(MX_OPENMP "Enable OpenMP parallelism" OFF)

if(MX_OPENMP)
  find_package(OpenMP)

  message("OpenMP_FOUND: ${OpenMP_FOUND}")

  message("OpenMP_CXX_INCLUDE_DIRS: ${OpenMP_CXX_INCLUDE_DIRS}")

  message("OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_INCLUDE_DIRS}")

  option(MX_OPENMP_BONDS "Enable OpenMP bonds" OFF)
  option(MX_OPENMP_INTEGRATOR "Enable OpenMP integrator" OFF)

endif()

option(MX_VECTORIZE_FLUX       "Enable vector flux calculations (AVX instructions)" OFF)
option(MX_VECTORIZE_FORCE      "Enable vector force calculations (AVX instructions)" OFF)
option(MX_VECTORIZE_BONDS      "Enable vector bonded interactions (AVX instructions)" OFF)
option(MX_VECTORIZE_INTEGRATOR "Enable vector integrator  (AVX instructions)" OFF)
option(MX_ENABLE_AVX2          "Enable AVX2 instructions" OFF)
option(MX_ENABLE_AVX           "Enable AVX  instructions" ON)
option(MX_ENABLE_SSE4          "Enable SSE4 instructions" ON)
option(MX_THREADING            "Enable multi-threaded flux and force calculations" ON)


if(MX_ENABLE_AVX2)
  set(MX_AVX2 ON)
  set(MX_AVX ON)
  set(MX_SSE42 ON)
  set(MX_SIMD_SIZE 8)
  add_compile_options(-mavx)
  add_compile_options(-mavx2)
  add_compile_options(-msse4.2)
elseif(MX_ENABLE_AVX)
  set(MX_AVX ON)
  set(MX_SSE42 ON)
  set(MX_SIMD_SIZE 8)
  add_compile_options(-mavx)
  add_compile_options(-msse4.2)
elseif(MX_ENABLE_SSE4)
  set(MX_SSE42 ON)
  set(MX_SIMD_SIZE 4)
  add_compile_options(-msse4.2)
else()
  set(MX_SIMD_SIZE 0)
endif()

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  message(STATUS "Detected CUDA")
  message(STATUS "  Version           : ${CUDAToolkit_VERSION}")
  message(STATUS "  Architectures     : ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "  Compiler          : ${CUDAToolkit_NVCC_EXECUTABLE}")
  set(MX_HAVECUDA ON)
else()
  message(STATUS "No detected CUDA support")
  set(MX_HAVECUDA OFF)
endif()

set(MX_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/mechanica)
set(MX_RESOURCE_DIR ${CMAKE_INSTALL_PREFIX}/resources/mechanica)

if(MX_HAVECUDA)
  enable_language(CUDA)
  set(MX_CUDA_DIR ${MX_RESOURCE_DIR}/cuda)
  set(MX_CUDA_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS})
  option(MX_WITHCUDA             "Enable GPU-accelerated calculations using CUDA" OFF)
  option(MX_CUDA_DEBUG           "Enable CUDA debug mode" OFF)
  if(DEFINED ENV{MX_WITHCUDA})
    set(MX_WITHCUDA ON)
  endif()
endif()

# quiet linker warnings on Apple
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Use Magnum's modules to find things, these set up cmake names / aliases using a
# magnum friendly naming scheme, add our module finders to cmake, these find
# stuff mechanica, needs, but magnum does not. Make sure to put Mechanica's
# dir before Magnums, so our versions get picked up *BEFORE* Magnum's.
list(APPEND CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake
  ${PROJECT_SOURCE_DIR}/extern/magnum/modules
  ${PROJECT_SOURCE_DIR}/extern/magnum-plugins/modules)


file(READ "VERSION.txt" ver)

string(REGEX MATCH "VERSION_MAJOR ([0-9]*)" _ ${ver})
set(MX_VERSION_MAJOR ${CMAKE_MATCH_1})

string(REGEX MATCH "VERSION_MINOR ([0-9]*)" _ ${ver})
set(MX_VERSION_MINOR ${CMAKE_MATCH_1})

string(REGEX MATCH "VERSION_PATCH ([0-9]*)" _ ${ver})
set(MX_VERSION_PATCH ${CMAKE_MATCH_1})

string(REGEX MATCH "VERSION_DEV ([0-9]*)" _ ${ver})
set(MX_VERSION_DEV ${CMAKE_MATCH_1})


message("version: ${MX_VERSION_MAJOR}.${MX_VERSION_MINOR}.${MX_VERSION_PATCH}")

message("CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message("CMAKE_OSX_DEPLOYMENT_TARGET: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message("MACOSX_DEPLOYMENT_TARGET: ${MACOSX_DEPLOYMENT_TARGET}")
message("CMAKE_OSX_SYSROOT: ${CMAKE_OSX_SYSROOT}")
message("XCODE_VERSION: ${XCODE_VERSION}")
message("CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message("CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

set(MX_HAVEPTHREADS FALSE)
if(USING_VCPKG)

  find_package(pthreads CONFIG REQUIRED)

  set_target_properties(PThreads4W::PThreads4W PROPERTIES IMPORTED_GLOBAL TRUE)
  add_library(Threads::Threads ALIAS PThreads4W::PThreads4W)
  set(MX_HAVEPTHREADS TRUE)

elseif(DEFINED PThreads_ROOT)

  find_library(PThreads 
    NAMES pthreads pthreadsVSE2 
    HINTS ${PThreads_ROOT} 
    REQUIRED 
  )
  message("PThreads path: ${PThreads}")
  add_library(pthreads UNKNOWN IMPORTED GLOBAL)
  target_include_directories(pthreads INTERFACE ${PThreads_ROOT}/include)
  set_target_properties(pthreads PROPERTIES IMPORTED_LOCATION ${PThreads})
  add_library(Threads::Threads ALIAS pthreads)
  set(MX_HAVEPTHREADS TRUE)

else()
  # use standard unix pthreads, creates Threads::Threads target
  find_package(Threads REQUIRED)
endif()

find_package(Python 3.7.0...3.9.99 REQUIRED COMPONENTS Interpreter Development)

message("Python_FOUND: ${Python_FOUND}")
message("Python_Interpeter_FOUND: ${Python_Interpeter_FOUND}")
message("Python_VERSION: ${Python_VERSION}")
message("Python_Development_FOUND: ${Python_Development_FOUND}")

message("Python_Development_FOUND: ${Python_Development_FOUND}")
message("Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message("Python_LIBRARIES: ${Python_LIBRARIES}")
message("Python_LIBRARY_DIRS: ${Python_LIBRARY_DIRS}")
message("Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")
message("Python_EXECUTABLE: ${Python_EXECUTABLE}")

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var(name='prefix'))" 
  OUTPUT_VARIABLE PY_ROOT_DIR
)
string(STRIP ${PY_ROOT_DIR} PY_ROOT_DIR)
file(TO_CMAKE_PATH "${PY_ROOT_DIR}" PY_ROOT_DIR)
message("PY_ROOT_DIR: ${PY_ROOT_DIR}")

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path(name='platlib'))" 
  OUTPUT_VARIABLE PY_SITEPACKAGES
)
string(STRIP ${PY_SITEPACKAGES} PY_SITEPACKAGES)
file(TO_CMAKE_PATH "${PY_SITEPACKAGES}" PY_SITEPACKAGES)
message("PY_SITEPACKAGES: ${PY_SITEPACKAGES}")

if(Python_Development_FOUND)
  get_target_property(TMP Python::Python IMPORTED_LOCATION_DEBUG)
  message("Python IMPORTED_LOCATION_DEBUG: ${TMP}")

  get_target_property(TMP Python::Python IMPORTED_LOCATION_RELEASE)
  message("Python IMPORTED_LOCATION_RELEASE: ${TMP}")
endif()

if(MX_APPLE AND DEFINED CMAKE_ECLIPSE_VERSION)
  # hack to get eclipse C++ indexer to pick up python headers
  message("setting eclipse include dirs: ${Python_INCLUDE_DIRS}")
  include_directories(${Python_INCLUDE_DIRS})
endif()

# Determine install site-packages location using info from environment if not provided
if(DEFINED MX_INSTALL_PREFIX_PYTHON)
  SET(MX_SITEPACKAGES ${MX_INSTALL_PREFIX_PYTHON})
else()
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig, os; print(os.path.relpath(sysconfig.get_path(name='platlib'), sysconfig.get_config_var(name='prefix')))" 
    OUTPUT_VARIABLE MX_SITEPACKAGES_REL
  )
  string(STRIP ${MX_SITEPACKAGES_REL} MX_SITEPACKAGES_REL)
  file(TO_CMAKE_PATH "${MX_SITEPACKAGES_REL}" MX_SITEPACKAGES_REL)
  set(MX_SITEPACKAGES ${CMAKE_INSTALL_PREFIX}/${MX_SITEPACKAGES_REL})
endif()
message("MX_SITEPACKAGES: ${MX_SITEPACKAGES}")

# needed for jpeg image writer
if(UNIX)
  # link statically with JPEG, so we dont have to ship a dylib,
  # brew installs newer JPEG, so that conflicts with stuff
  find_library(libJPEG NAMES libjpeg.a REQUIRED)
  message("libjpeg path: ${libJPEG}")
  set(JPEG_LIBRARY_RELEASE ${libJPEG})
endif()
find_package(JPEG REQUIRED)

message("JPEG_INCLUDE_DIRS: ${JPEG_INCLUDE_DIRS}")
message("JPEG_LIBRARIES: ${JPEG_LIBRARIES}")
message("JPEG_VERSION: ${JPEG_VERSION}")

set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/local/mechanica" CACHE PATH "Install location for Mechancia")

set(BUILD_SHARED_LIBS ON CACHE BOOL "build shared/dynamic library")

set(MX_MODEL_DIR ${PROJECT_SOURCE_DIR}/testing/models CACHE PATH "directory for mechanica models" FORCE)

find_package(OpenGL REQUIRED)

#find_package(LLVM)

#find_package(OpenMM)

add_definitions(-DCX3D_SUPPORT_OPENGL)


include_directories(${OPENGL_INCLUDE_DIR})

#if(LLVM_FOUND)
#  message("LLVM_FOUND")
#  
#  message("LLVM_CXXFLAGS: ${LLVM_CXXFLAGS}")
#  message("LLVM_HOST_TARGET: ${LLVM_HOST_TARGET}")
#  
#  message("LLVM_LDFLAGS: ${LLVM_LDFLAGS}")
#  
#  message("LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
#  
#  
#  
#  message("LLVM_LIBRARIES: ${LLVM_LIBRARIES}")
#  message("LLVM_LIBRARY_DIRS: ${LLVM_LIBRARY_DIRS}")
#  message("LLVM_ROOT_DIR: ${LLVM_ROOT_DIR}")
#  
#  message("LLVM_VERSION_MAJOR: ${LLVM_VERSION_MAJOR}")
#  message("LLVM_VERSION_MINOR: ${LLVM_VERSION_MINOR}")
#  message("LLVM_VERSION_STRING: ${LLVM_VERSION_STRING}")
#
#endif()

find_package(assimp REQUIRED)

if(assimp_FOUND)
  message("Assimp found: ${ASSIMP_INCLUDE_DIRS}")
else()
  message("ASSIMP not found")
endif()

if(DEFINED ENV{JSON_INCLUDE_DIRS})
  set(JSON_INCLUDE_DIRS "$ENV{JSON_INCLUDE_DIRS}" CACHE STRING "")
else()
  set(JSON_INCLUDE_DIRS ${CMAKE_FIND_ROOT_PATH}/include)
endif()
message("JSON inlucde: ${JSON_INCLUDE_DIRS}")


# finds: GLFW_FOUND GLFW_INCLUDE_DIR GLFW_LIBRARIES
# find_package(GLFW 3.2 REQUIRED)


#include_directories(${GLFW_INCLUDE_DIRS})

#message("GLFW_INCLUDE_DIR ${GLFW_INCLUDE_DIR}")

#include_directories(${YOUR_DIRECTORY})

#include_directories(${GLFW_INCLUDE_DIR})

#message("GLFW_LIBRARIES: ${GLFW_LIBRARIES}")

message("before extern include path: ${CMAKE_INCLUDE_PATH}")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message( "dir='${dir}'")
endforeach()

#if(MSVC)
#  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
#  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
#  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
#endif(MSVC)



# process dependencies first
add_subdirectory(extern)


message("getting mdcore interface incude:  ${MDCORE_INCLUDE_DIR}")

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(models)


# CMake policies: enable MACOSX_RPATH by default
if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()
# Don't treat imported targets with :: as files
if(POLICY CMP0028)
    cmake_policy(SET CMP0028 NEW)
endif()

find_package(Magnum REQUIRED
  Shaders
  GlfwApplication)


add_subdirectory(apps)
add_subdirectory(testing)
#add_subdirectory(magnum_examples)
add_subdirectory(examples)
add_subdirectory(package)

# try to make eclipse see the system python include dir.
if(MX_APPLE AND DEFINED CMAKE_ECLIPSE_VERSION)
  # hack to get eclipse C++ indexer to pick up python headers
  message("setting eclipse include dirs: ${Python_INCLUDE_DIRS}")
  include_directories(${Python_INCLUDE_DIRS})
endif()


set(CMAKE_INFO_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Install location for Mechancia basic info")
set(MX_INFO_FILES
  README.md 
  LICENSE.txt 
  VERSION.txt 
)
install(FILES ${MX_INFO_FILES} DESTINATION ${CMAKE_INFO_PREFIX})
