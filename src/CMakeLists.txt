# LLVM compile options
#add_definitions(${LLVM_LDFLAGS})

add_compile_options(-DMX_STRICT)

add_subdirectory(mdcore)

# quiet linker warnings on Apple
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Make sure the compiler can find include files for our Hello library
# when other libraries or executables link to Hello
include_directories(PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})

# Resources for the MxMeshShaderProgram
corrade_add_resource(MxMeshShaderProgramRes MxMeshShaderProgram.res)

set(SRC

  MxSimulator.cpp
  
  MxSystem.cpp
  stochastic_rk.cpp
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
  event/MxParticleEvent.cpp
  event/MxParticleEventSingle.cpp
  event/MxParticleTimeEvent.cpp
  event/MxTimeEvent.cpp

  rendering/ArcBallInteractor.cpp
  rendering/MxGlfwApplication.cpp
  rendering/MxWindowlessApplication.cpp
  rendering/MxApplication.cpp
  rendering/MxWindow.cpp
  rendering/MxGlfwWindow.cpp
  rendering/MxRenderer.cpp
  rendering/MxUniverseRenderer.cpp
  rendering/MxSubRenderer.cpp

  rendering/MxAngleRenderer.cpp
  rendering/MxArrowRenderer.cpp
  rendering/MxBondRenderer.cpp
  rendering/MxDihedralRenderer.cpp
  rendering/MxOrientationRenderer.cpp
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
  )


set(HEADERS
  MxDebug.h
  mechanica_private.h
  
  MxSimulator.h

  MxSystem.h

  stochastic_rk.h
  
  SymmetricEigen.h
  GteSymmetricEigensolver3x3.h
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
  event/MxParticleEvent.h
  event/MxParticleEventSingle.h
  event/MxParticleTimeEvent.h
  event/MxTimeEvent.h

  rendering/ArcBallInteractor.h
  rendering/MxGlfwApplication.h
  rendering/MxWindowlessApplication.h
  rendering/MxWindowless.h
  rendering/MxApplication.h
  rendering/MxWindow.h
  rendering/MxGlfwWindow.h
  rendering/MxRenderer.h
  rendering/MxUniverseRenderer.h
  rendering/MxSubRenderer.h
  rendering/WireframeObjects.h
  rendering/FlatShadeObject.h
  rendering/MxAngleRenderer.h
  rendering/MxArrowRenderer.h
  rendering/MxBondRenderer.h
  rendering/MxDihedralRenderer.h
  rendering/MxOrientationRenderer.h
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

# Python source

if(MX_WITH_PYTHON)

  list(APPEND SRC 
    langs/py/MxPy.cpp

    langs/py/MxApplicationPy.cpp 
    langs/py/MxBoundaryConditionsPy.cpp 
    langs/py/MxEventPy.cpp 
    langs/py/MxEventPyExecutor.cpp 
    langs/py/MxForcePy.cpp 
    langs/py/MxKeyEventPy.cpp 
    langs/py/MxParticleEventPy.cpp 
    langs/py/MxParticleEventSinglePy.cpp 
    langs/py/MxParticleTimeEventPy.cpp 
    langs/py/MxPotentialPy.cpp 
    langs/py/MxSimulatorPy.cpp 
    langs/py/MxSystemPy.cpp 
    langs/py/MxTimeEventPy.cpp
  )

  list(APPEND HEADERS 
    langs/py/MxPy.h
    
    langs/py/MxApplicationPy.h 
    langs/py/MxBoundaryConditionsPy.h 
    langs/py/MxEventPy.h 
    langs/py/MxEventPyExecutor.h 
    langs/py/MxForcePy.h 
    langs/py/MxKeyEventPy.h 
    langs/py/MxParticleEventPy.h 
    langs/py/MxParticleEventSinglePy.h 
    langs/py/MxParticleTimeEventPy.h 
    langs/py/MxPotentialPy.h 
    langs/py/MxSimulatorPy.h 
    langs/py/MxSystemPy.h 
    langs/py/MxTimeEventPy.h
  )

endif()

# CUDA source

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


target_include_directories(mechanica_obj PRIVATE
  ${ASSIMP_INCLUDE_DIRS}
  ${JSON_INCLUDE_DIRS}
  )

# temporarily add add these includes until we get a stable public API
target_include_directories(mechanica_obj PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/mechanica>
)

if(MX_WITH_PYTHON)
  target_include_directories(mechanica_obj PUBLIC ${Python_INCLUDE_DIRS})
endif()

target_link_libraries(mechanica_obj
  glfw 
  ${GLFW_LIBRARIES} 
  LibSBML::LibSBML
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  MagnumPlugins::StbImageConverter
  assimp::assimp
  ${OPENGL_LIBRARIES}
  mdcore_single
)

add_dependencies(mechanica_obj MagnumPlugins::StbImageConverter)

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
  ${ASSIMP_INCLUDE_DIRS}
  ${JSON_INCLUDE_DIRS}
  $<INSTALL_INTERFACE:include/mechanica>
)

if(MX_WITH_PYTHON)
  target_include_directories(mechanica_shared PUBLIC ${Python_INCLUDE_DIRS})
endif()

if(MX_WITHCUDA)
  target_include_directories(mechanica_shared PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  set_target_properties(mechanica_shared PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

target_link_libraries(mechanica_shared
  PUBLIC
  mdcore_single
  glfw 
  ${GLFW_LIBRARIES} 
  LibSBML::LibSBML
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  MagnumGlfwApplication
  Magnum::SceneGraph
  Magnum::Primitives
  Magnum::MeshTools
  LibSBML::LibSBML
  MagnumPlugins::StbImageConverter
  TgaImageConverter
  assimp::assimp
  )


if(MX_WITH_PYTHON)
  # weird bug with CMake / clang on windows, need to add link dir here
  target_link_directories(mechanica_shared PUBLIC ${Python_LIBRARY_DIRS})

  # Link against the Python shared library on Windows
  # target_link_libraries(mechanica_py PRIVATE Python::Python)
  target_link_libraries(mechanica_shared PUBLIC ${Python_LIBRARY_RELEASE})
endif()

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

install(TARGETS mechanica_shared 
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" 
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" 
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

foreach(_PRHDR ${HEADERS_PRIVATE})
  cmake_path(GET _PRHDR PARENT_PATH _PRHDR_PARENT)
  install(FILES ${_PRHDR} DESTINATION include/mechanica/private/${_PRHDR_PARENT})
endforeach()
