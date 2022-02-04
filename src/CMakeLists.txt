# LLVM compile options
#add_definitions(${LLVM_LDFLAGS})

add_compile_options(-DMX_STRICT)

add_subdirectory(mdcore)

# quiet linker warnings on Apple
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Make sure the compiler can find include files for our Hello library
# when other libraries or executables link to Hello
include_directories(PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})


# file(GLOB AST_SRC
#   RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
#   "ast/*.cpp"
#   )

#file(GLOB X3D_SRC
#  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
#  "x3d/*.cpp"
#  )

# Resources for the MxMeshShaderProgram
corrade_add_resource(MxMeshShaderProgramRes MxMeshShaderProgram.res)

set(SRC

  MxMeshCore.cpp
  MxMesh.cpp
  MxPolygon.cpp

  MxSimulator.cpp
  MxPropagator.cpp

  MxModel.cpp
  MxSystem.cpp
  MxPropagator.cpp
  MxCompositeParticle.cpp
  MxController.cpp
  MxView.cpp
  MxLattice.cpp
  MxLatticeCluster.cpp
  MxCell.cpp
  MeshPropagator.cpp
  MxMeshPottsPropagator.cpp
  MxExecutionContext.cpp
  MxForceFunction.cpp
  LangevinPropagator.cpp
  ScalarField.cpp
  MeshRelationships.cpp
  MeshOperations.cpp
  stochastic_rk.cpp
  MxEdge.cpp
  MeshIO.cpp
  MxTesselator.cpp

  FlipEdge.cpp
  CollapsePolygon.cpp
  SplitPolygon.cpp
  SplitCell.cpp
  SplitEdge.cpp
  FlipEdgeToPolygon.cpp
  FlipPolygonToEdge.cpp
  MxConstraints.cpp
  MxPolygonSurfaceTensionForce.cpp
  MxPolygonAreaConstraint.cpp
  MxCellVolumeConstraint.cpp
  MxSurfaceSimulator.cpp
  MxCylinderModel.cpp
  MxUniverse.cpp

  MxLogger.cpp
  MxUtil.cpp
  mx_error.cpp
  mx_parse.cpp

  io/Mx3DFVertexData.cpp
  io/Mx3DFEdgeData.cpp
  io/Mx3DFFaceData.cpp
  io/Mx3DFMeshData.cpp
  io/Mx3DFRenderData.cpp
  io/Mx3DFStructure.cpp
  io/generators/Mx3DFMeshGenerator.cpp
  io/generators/Mx3DFAngleMeshGenerator.cpp
  io/generators/Mx3DFBondMeshGenerator.cpp
  io/generators/Mx3DFDihedralMeshGenerator.cpp
  io/generators/Mx3DFPCloudMeshGenerator.cpp
  io/Mx3DFIO.cpp
  io/MxFIO.cpp
  io/MxIO.cpp
  io/mx_io.cpp

  event/MxEvent.cpp
  event/MxEventList.cpp
  event/MxEventPyExecutor.cpp
  event/MxParticleEvent.cpp
  event/MxParticleEventSingle.cpp
  event/MxParticleTimeEvent.cpp
  event/MxTimeEvent.cpp

  rendering/ArcBallInteractor.cpp
  rendering/MxGlfwApplication.cpp
  rendering/MxWindowlessApplication.cpp
  rendering/MxApplication.cpp
  rendering/MxWindowView.cpp
  rendering/MxWindow.cpp
  rendering/MxWindowProxy.cpp
  rendering/MxWindowHost.cpp
  rendering/MxGlfwWindow.cpp
  rendering/MxRenderer.cpp
  rendering/MxUniverseRenderer.cpp
  rendering/MxSubRenderer.cpp

  rendering/MxAngleRenderer.cpp
  rendering/MxArrowRenderer.cpp
  rendering/MxBondRenderer.cpp
  rendering/MxMeshRenderer.cpp
  rendering/MxDihedralRenderer.cpp
  rendering/MxOrientationRenderer.cpp
  rendering/MxMeshShaderProgram.cpp
  rendering/MxRenderTarget.cpp
  rendering/MxImageView.cpp
  rendering/MxUI.cpp
  rendering/MxTestView.cpp
  rendering/MxImageConverters.cpp

  rendering/ArcBall.cpp
  rendering/MxStyle.cpp
  rendering/MxKeyEvent.cpp

  rendering/MxColorMapper.cpp

  rendering/MxGlInfo.cpp

  rendering/MxEglInfo.cpp

  rendering/MxClipPlane.cpp

  shaders/ParticleSphereShader.cpp

  shaders/MxFlat3D.cpp
  shaders/MxPhong.cpp

  state/MxSpecies.cpp
  state/MxSpeciesList.cpp
  state/MxSpeciesValue.cpp
  state/MxStateVector.cpp

  types/mx_cast.cpp

  # ${AST_SRC}
#  ${X3D_SRC}
  )


set(HEADERS
  MxDebug.h
  mechanica_llvm.h
  mechanica_private.h
  MxMeshCore.h
  MxMesh.h
  MxPolygon.h

  MxSimulator.h

  MxModel.h
  MxSystem.h
  MxPropagator.h

  MxCompositeParticle.h
  MxController.h
  MxView.h
  MxLattice.h
  MxLatticeCluster.h
  MxCell.h

  MeshPropagator.h
  MxMeshPottsPropagator.h
  MxExecutionContext.h
  MxForceFunction.h
  LangevinPropagator.h
  ScalarField.h
  MeshRelationships.h
  MeshOperations.h
  stochastic_rk.h
  MxEdge.h
  MeshIO.h
  MxTesselator.h

  SymmetricEigen.h
  GteSymmetricEigensolver3x3.h
  MxConstraints.h
  MxPolygonSurfaceTensionForce.h
  MxPolygonAreaConstraint.h
  MxCellVolumeConstraint.h
  MxSurfaceSimulator.h
  MxCylinderModel.h
  MxUniverse.h

  MxLogger.h
  MxUtil.h
  mx_error.h
  mx_parse.h

  io/Mx3DFVertexData.h
  io/Mx3DFEdgeData.h
  io/Mx3DFFaceData.h
  io/Mx3DFMeshData.h
  io/Mx3DFRenderData.h
  io/Mx3DFStructure.h
  io/generators/Mx3DFMeshGenerator.h
  io/generators/Mx3DFAngleMeshGenerator.h
  io/generators/Mx3DFBondMeshGenerator.h
  io/generators/Mx3DFDihedralMeshGenerator.h
  io/generators/Mx3DFPCloudMeshGenerator.h
  io/Mx3DFIO.h
  io/MxFIO.h
  io/MxIO.h
  io/mx_io.h

  event/MxEvent.h
  event/MxEventList.h
  event/MxEventPyExecutor.h
  event/MxParticleEvent.h
  event/MxParticleEventSingle.h
  event/MxParticleTimeEvent.h
  event/MxTimeEvent.h

  rendering/ArcBallInteractor.h
  rendering/MxGlfwApplication.h
  rendering/MxWindowlessApplication.h
  rendering/MxWindowless.h
  rendering/MxApplication.h
  rendering/MxWindowView.h
  rendering/MxWindow.h
  rendering/MxWindowProxy.h
  rendering/MxWindowHost.h
  rendering/MxGlfwWindow.h
  rendering/MxRenderer.h
  rendering/MxUniverseRenderer.h
  rendering/MxSubRenderer.h
  rendering/WireframeObjects.h
  rendering/FlatShadeObject.h
  rendering/MxAngleRenderer.h
  rendering/MxArrowRenderer.h
  rendering/MxBondRenderer.h
  rendering/MxMeshRenderer.h
  rendering/MxDihedralRenderer.h
  rendering/MxOrientationRenderer.h
  rendering/MxMeshShaderProgram.h
  rendering/MxRenderTarget.h
  rendering/MxImageView.h
  rendering/MxUI.h
  rendering/MxTestView.h
  rendering/MxImageConverters.h

  rendering/ArcBallCamera.h
  rendering/ArcBall.h
  rendering/MxStyle.hpp
  rendering/MxKeyEvent.hpp

  rendering/MxColorMapper.hpp

  rendering/MxGlInfo.h

  rendering/MxEglInfo.h

  rendering/MxClipPlane.hpp

  rendering/colormaps/colormaps.h

  shaders/ParticleSphereShader.h

  shaders/MxFlat3D.h
  shaders/MxPhong.h

  state/MxSpecies.h
  state/MxSpeciesList.h
  state/MxSpeciesValue.h
  state/MxStateVector.h

  types/mx_cast.h
  types/mx_types.h
  types/MxVector.h
  types/MxVector2.h
  types/MxVector3.h
  types/MxVector4.h
  types/MxMatrix.h
  types/MxMatrix3.h
  types/MxMatrix4.h
  types/MxQuaternion.h
)

if(MX_WITHCUDA)
  list(APPEND SRC 
    cuda/MxAngleCUDAConfig.cpp 
    cuda/MxBondCUDAConfig.cpp 
    cuda/MxEngineCUDAConfig.cpp 
    cuda/MxSimulatorCUDAConfig.cpp
  )
  list(APPEND HEADERS 
    cuda/MxAngleCUDAConfig.h 
    cuda/MxBondCUDAConfig.h 
    cuda/MxEngineCUDAConfig.h 
    cuda/MxSimulatorCUDAConfig.h
  )
endif()

set(HEADERS_PRIVATE ${HEADERS})
foreach(_PSHDR ${MECHANICA_PUBLIC_HEADERS_STATIC})
  list(APPEND ${HEADERS} ${PROJECT_SOURCE_DIR}/include/${_PSHDR})
endforeach()

include(models/modeltree.cmake)
add_subdirectory(models)
MX_MODEL_TREE_INIT()
MX_MODEL_TREE_PROC(models)
set(SRC ${SRC} ${MX_MODEL_SRCS_LOCAL})

# For development, for the time being, we won't build the object library
# because it takes longer to build if we're only building one library,
# and won't build the py version for a while.

# build all the source as a CMake 'OBJECT' lib, this gets
# linked into the real libraries below.
#add_library(mechanica_OBJECTS OBJECT ${SRC} ${HEADERS})


# get the include directories of the dependencies that the mechanica objects use
# really goofy design decision in design of CMake OBJECT libraries -- no transitive includes
#target_include_directories(mechanica_OBJECTS PRIVATE $<TARGET_PROPERTY:mdcore_single,INTERFACE_INCLUDE_DIRECTORIES>)
#target_include_directories(mechanica_OBJECTS PRIVATE $<TARGET_PROPERTY:voro++,INTERFACE_INCLUDE_DIRECTORIES> )
#target_include_directories(mechanica_OBJECTS PRIVATE $<TARGET_PROPERTY:Magnum::Magnum,INTERFACE_INCLUDE_DIRECTORIES>)
#target_include_directories(mechanica_OBJECTS PRIVATE "${PROJECT_SOURCE_DIR}/include")
#target_include_directories(mechanica_OBJECTS PRIVATE ${NUMPY_INCLUDE_DIR})

if(MX_WITHCUDA)
  list(APPEND HEADERS mx_cuda.h)
  list(APPEND SRC mx_cuda.cpp)
endif()

# this is the "object library" target: compiles the sources only once
# need at least one src file in a project for xcode to build it, xcode
# won't work if any product *only* has an object src
add_library(mechanica_obj
  OBJECT
  ${SRC}
  ${PROJECT_BINARY_DIR}/include/mx_config.h
  ${HEADERS}
  ${MxMeshShaderProgramRes}
  )

target_compile_definitions(mechanica_obj PRIVATE C_BUILDING_DLL)

set_target_properties(mechanica_obj PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
  )

message("Shader Program Res: ${MxMeshShaderProgramRes}")


target_include_directories(mechanica_obj PRIVATE
  ${PYBIND11_INCLUDE_DIR}
  ${JPEG_INCLUDE_DIRS}
  ${ASSIMP_INCLUDE_DIRS}
  ${JSON_INCLUDE_DIRS}
  )

# set_target_properties(mechanica PROPERTIES POSITION_INDEPENDENT_CODE ON)

# temporarily add add these includes until we get a stable public API
target_include_directories(mechanica_obj PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  ${Python_INCLUDE_DIRS}
  $<INSTALL_INTERFACE:include/mechanica>
)

target_link_libraries(mechanica_obj
  DirectXMath
  glfw 
  ${GLFW_LIBRARIES} 
  LibSBML::LibSBML
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  #voro++
  #sundials_cvode_static
  assimp::assimp
  ${OPENGL_LIBRARIES}
  mdcore_single
)

if(MX_WITHCUDA)
  target_link_libraries(mechanica_obj
    CUDA::cudart 
    CUDA::cuda_driver 
    CUDA::nvrtc
  )
endif()

source_group("Public Header Files" FILES ${MECHANICA_PUBLIC_HEADERS})
source_group("Event" REGULAR_EXPRESSION "event/.*")
source_group("Rendering" REGULAR_EXPRESSION "rendering/.*")
source_group("Shaders" REGULAR_EXPRESSION "shaders/.*")
source_group("State" REGULAR_EXPRESSION "state/.*")


#message("MECHANICA_PUBLIC_HEADERS: ${MECHANICA_PUBLIC_HEADERS}")

#message("MECHANICA_SRC: ${SRC}")


# Link against LLVM libraries
#link_directories(${LLVM_LIBRARY_DIRS})
#target_link_libraries(mechanica ${LLVM_LIBRARIES})


# shared libraries need PIC
set_property(TARGET mechanica_obj PROPERTY POSITION_INDEPENDENT_CODE 1)


# shared and static libraries built from the same object files
add_library(mechanica_shared SHARED
  $<TARGET_OBJECTS:mechanica_obj>
  $<TARGET_OBJECTS:mdcore_single>
  mechanica.cpp
  )

target_compile_definitions(mechanica_shared PRIVATE C_BUILDING_DLL)

set_target_properties(mechanica_shared PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
  )

set_target_properties(mechanica_shared PROPERTIES OUTPUT_NAME mechanica)

target_include_directories(mechanica_shared PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  ${Python_INCLUDE_DIRS}
  ${ASSIMP_INCLUDE_DIRS}
  ${JSON_INCLUDE_DIRS}
  $<INSTALL_INTERFACE:include/mechanica>
)

target_include_directories(mechanica_shared PRIVATE ${PYBIND11_INCLUDE_DIR})

if(MX_WITHCUDA)
  target_include_directories(mechanica_shared PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  set_target_properties(mechanica_shared PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

find_package(SWIG 4.0 REQUIRED COMPONENTS python)
include(UseSWIG)

# shared and static libraries built from the same object files

set_source_files_properties(mechanica.i PROPERTIES CPLUSPLUS ON SWIG_FLAGS "-keyword;-py3;-doxygen")

set(SWIG_COMPILEDEFS_PY )

# Power through int size mess until properly resolved 
# See https://github.com/swig/swig/issues/1850
include(CheckTypeSize)
check_type_size("long" SIZE_LONG)
message(STATUS "Got size of long: ${SIZE_LONG} (${HAVE_SIZE_LONG})")
if(NOT MX_APPLE)
  if(${SIZE_LONG} STREQUAL 8)
    list(APPEND SWIG_COMPILEDEFS_PY SWIGWORDSIZE64)
  else()
    list(APPEND SWIG_COMPILEDEFS_PY SWIGWORDSIZE32)
  endif()
endif()

if(MX_APPLE)
  list(APPEND SWIG_COMPILEDEFS_PY CORRADE_TARGET_APPLE)
endif()

message(STATUS "SWIG_COMPILEDEFS_PY=${SWIG_COMPILEDEFS_PY}")
set_property(SOURCE mechanica.i PROPERTY COMPILE_DEFINITIONS ${SWIG_COMPILEDEFS_PY})

swig_add_library(mechanica_py TYPE SHARED LANGUAGE python SOURCES mechanica.i $<TARGET_OBJECTS:mechanica_obj>)

target_compile_definitions(mechanica_py PRIVATE C_BUILDING_DLL)

set_target_properties(mechanica_py PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
  SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE
  )


message("seting output dir to ${MX_PYPKG_DIR}")

# copy the mechanica py lib to the package dir
set_target_properties(mechanica_py PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY          ${MX_PYPKG_DIR}
  ARCHIVE_OUTPUT_DIRECTORY_DEBUG    ${MX_PYPKG_DIR}
  ARCHIVE_OUTPUT_DIRECTORY_RELEASE  ${MX_PYPKG_DIR}

  LIBRARY_OUTPUT_DIRECTORY          ${MX_PYPKG_DIR}
  LIBRARY_OUTPUT_DIRECTORY_DEBUG    ${MX_PYPKG_DIR}
  LIBRARY_OUTPUT_DIRECTORY_RELEASE  ${MX_PYPKG_DIR}

  RUNTIME_OUTPUT_DIRECTORY          ${MX_PYPKG_DIR}
  RUNTIME_OUTPUT_DIRECTORY_DEBUG    ${MX_PYPKG_DIR}
  RUNTIME_OUTPUT_DIRECTORY_RELEASE  ${MX_PYPKG_DIR}
  )


target_include_directories(mechanica_py PRIVATE
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  ${Python_INCLUDE_DIRS}
  $<INSTALL_INTERFACE:include/mechanica>
)

if(MX_WITHCUDA)
  target_include_directories(mechanica_py PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  set_target_properties(mechanica_py PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

target_include_directories(mechanica_py PRIVATE ${PYBIND11_INCLUDE_DIR})

set(mechanica_py_LIBS
  mdcore_single
  glfw 
  ${GLFW_LIBRARIES} 
  LibSBML::LibSBML 
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  #voro++
  #sundials_cvode_static
  MagnumGlfwApplication
  Magnum::SceneGraph
  Magnum::Primitives
  Magnum::MeshTools
  LibSBML::LibSBML
  JPEG::JPEG 
  TgaImageConverter
  assimp::assimp
)

if(MX_WITHCUDA)
  list(APPEND mechanica_py_LIBS 
    CUDA::cudart 
    CUDA::cuda_driver 
    CUDA::nvrtc
  )
endif()

if(MX_APPLE)
  list(APPEND mechanica_py_LIBS MagnumWindowlessCglApplication)
elseif(MX_LINUX)
  list(APPEND mechanica_py_LIBS MagnumWindowlessEglApplication)
elseif(MX_WINDOWS)
  list(APPEND mechanica_py_LIBS MagnumWindowlessWglApplication)
endif()

swig_link_libraries(mechanica_py PRIVATE ${mechanica_py_LIBS})

if(WIN32 OR CYGWIN OR (UNIX AND NOT APPLE))

  # weird bug with CMake / clang on windows, need to add link dir here
  # target_link_directories(mechanica_py PUBLIC ${Python_LIBRARY_DIRS})

  # Link against the Python shared library on Windows
  # target_link_libraries(mechanica_py PRIVATE Python::Python)
  swig_link_libraries(mechanica_py PRIVATE ${Python_LIBRARIES})

elseif(APPLE)
  # It's quite common to have multiple copies of the same Python version
  # installed on one's system. E.g.: one copy from the OS and another copy
  # that's statically linked into an application like Blender or Maya.
  # If we link our plugin library against the OS Python here and import it
  # into Blender or Maya later on, this will cause segfaults when multiple
  # conflicting Python instances are active at the same time (even when they
  # are of the same version).

  # Windows is not affected by this issue since it handles DLL imports
  # differently. The solution for Linux and Mac OS is simple: we just don't
  # link against the Python library. The resulting shared library will have
  # missing symbols, but that's perfectly fine -- they will be resolved at
  # import time.

  target_link_libraries(mechanica_py PRIVATE "-undefined dynamic_lookup")

  if(ARG_SHARED)
    # Suppress CMake >= 3.0 warning for shared libraries
    set_target_properties(mechanica_py PROPERTIES MACOSX_RPATH ON)
  endif()
endif()

# Handle rpath
if(UNIX)
  file(RELATIVE_PATH _RPATH2PYLIBS ${MX_SITEPACKAGES}/mechanica ${Python_LIBRARY_DIRS})
  file(RELATIVE_PATH _RPATH2MXLIBS ${MX_SITEPACKAGES}/mechanica ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
  file(RELATIVE_PATH _RPATH2MXENV  ${MX_SITEPACKAGES}/mechanica ${CMAKE_INSTALL_PREFIX})
  set_target_properties(mechanica_py PROPERTIES INSTALL_RPATH "${RPATH_PREFIX};${RPATH_PREFIX}/${_RPATH2PYLIBS};${RPATH_PREFIX}/${_RPATH2MXLIBS};${RPATH_PREFIX}/${_RPATH2MXENV}")
endif()

set_target_properties(mechanica_py PROPERTIES PREFIX "")
set_target_properties(mechanica_py PROPERTIES OUTPUT_NAME "_mechanica")

if(MX_WINDOWS)
  set_target_properties(mechanica_py PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties(mechanica_py PROPERTIES SUFFIX ".so")
endif()


#add_custom_command(
#  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/mechanica.py"
#  POST_BUILD
#  COMMAND ${CMAKE_COMMAND} -E copy
#  ${CMAKE_CURRENT_SRC_DIR}/mechanica.py
#  ${CMAKE_CURRENT_BINARY_DIR}/mechanica.py
#  DEPENDS mechanica.py)


#add_custom_command(
# TARGET ${destination} POST_BUILD
# COMMAND ${CMAKE_COMMAND} -E copy ${source} ${destination}
# DEPENDS ${source}
# COMMENT "symbolic link resources folder from ${source} => ${destination}"
# )

#add_custom_target(mechanica.py
#  DEPENDS mechanica.py)

#add_custom_command(
#  TARGET mechanica_py
#  DEPENDS __init__.py
#  POST_BUILD
#  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${MX_PYPKG_DIR}/__init__.py
#  )

file(COPY setup.py DESTINATION ${CMAKE_BINARY_DIR})
file(COPY __init__.py DESTINATION ${MX_PYPKG_DIR})
file(COPY lattice.py DESTINATION ${MX_PYPKG_DIR})
file(COPY jwidget.py DESTINATION ${MX_PYPKG_DIR})
file(COPY particle_type.py DESTINATION ${MX_PYPKG_DIR})


target_link_libraries(mechanica_shared
  PUBLIC
  mdcore_single
  glfw 
  ${GLFW_LIBRARIES} 
  LibSBML::LibSBML
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  #voro++
  #sundials_cvode_static
  MagnumGlfwApplication
  Magnum::SceneGraph
  Magnum::Primitives
  Magnum::MeshTools
  LibSBML::LibSBML
  JPEG::JPEG 
  TgaImageConverter
  assimp::assimp
  )


# weird bug with CMake / clang on windows, need to add link dir here
target_link_directories(mechanica_shared   PUBLIC ${Python_LIBRARY_DIRS})

# Link against the Python shared library on Windows
# target_link_libraries(mechanica_py PRIVATE Python::Python)
target_link_libraries(mechanica_shared  PUBLIC ${Python_LIBRARY_RELEASE})

if(APPLE)
  if(ARG_SHARED)
    # Suppress CMake >= 3.0 warning for shared libraries
    set_target_properties(mechanica_shared PROPERTIES MACOSX_RPATH ON)
  endif()
endif()

if(MX_WITHCUDA)
  target_link_libraries(mechanica_shared PUBLIC 
    CUDA::cudart 
    CUDA::cuda_driver 
    CUDA::nvrtc
  )
endif()

if(MX_APPLE)
  target_link_libraries(mechanica_shared PRIVATE MagnumWindowlessCglApplication)
elseif(MX_LINUX)
  target_link_libraries(mechanica_shared PRIVATE MagnumWindowlessEglApplication)
elseif(MX_WINDOWS)
  target_link_libraries(mechanica_shared PRIVATE MagnumWindowlessWglApplication)
endif()


#add_library(mechanica_static STATIC
#  $<TARGET_OBJECTS:mechanica_obj> mechanica.cpp)

#set_target_properties(mechanica_static PROPERTIES OUTPUT_NAME mechanica)

#target_link_libraries(mechanica_static
#  glfw
#  assimp::assimp
#  Magnum::Application
#  Magnum::Magnum
#  Magnum::Shaders
#  voro++
#  sundials_cvode_static
#  ${OPENGL_LIBRARIES}
#  )

#if(MX_APPLE)
#  target_link_libraries(mechanica_static MagnumWindowlessCglApplication)
#elseif(MX_LINUX)
#  target_link_libraries(mechanica_static MagnumWindowlessEglApplication)
#elseif(MX_WINDOWS)
#  target_link_libraries(mechanica_static MagnumWindowlessWglApplication)
#endif()


add_library(mechanica ALIAS mechanica_shared)
add_library(Mechanica::Mechanica ALIAS mechanica_shared)
#add_library(Mechanica::Static ALIAS mechanica_static)
add_library(Mechanica::Shared ALIAS mechanica_shared)
add_library(Mechanica::PythonLib ALIAS mechanica_py)

install(TARGETS mechanica_shared 
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" 
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" 
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

install(TARGETS mechanica_py DESTINATION ${MX_SITEPACKAGES_REL}/mechanica)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mechanica.py DESTINATION ${MX_SITEPACKAGES_REL}/mechanica)
install(FILES 
        __init__.py 
        lattice.py 
        jwidget.py
        particle_type.py 
        DESTINATION 
        ${MX_SITEPACKAGES_REL}/mechanica
)
foreach(_PRHDR ${HEADERS_PRIVATE})
  cmake_path(GET _PRHDR PARENT_PATH _PRHDR_PARENT)
  install(FILES ${_PRHDR} DESTINATION include/mechanica/private/${_PRHDR_PARENT})
endforeach()
