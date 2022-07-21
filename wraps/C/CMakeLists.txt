set(SOURCES 
    mechanica_c.cpp 
    mechanica_c_private.cpp 

    MxCBind.cpp 
    MxCBond.cpp 
    MxCBoundaryConditions.cpp 
    MxCClipPlane.cpp 
    MxCCluster.cpp 
    MxCEvent.cpp 
    MxCFlux.cpp 
    MxCForce.cpp 
    MxCIO.cpp 
    MxCLogger.cpp 
    MxCParticle.cpp 
    MxCPotential.cpp 
    MxCSimulator.cpp 
    MxCSpecies.cpp 
    MxCStateVector.cpp 
    MxCStyle.cpp 
    MxCSystem.cpp 
    MxCUniverse.cpp 
    MxCUtil.cpp 
)

set(HEADERS
    mechanica_c.h 
    mechanica_c_private.h 

    MxCBind.h 
    MxCBond.h 
    MxCBoundaryConditions.h 
    MxCClipPlane.h 
    MxCCluster.h 
    MxCEvent.h 
    MxCFlux.h 
    MxCForce.h 
    MxCIO.h 
    MxCLogger.h 
    MxCParticle.h 
    MxCPotential.h 
    MxCSimulator.h 
    MxCSpecies.h 
    MxCStateVector.h 
    MxCStyle.h 
    MxCSystem.h 
    MxCUniverse.h 
    MxCUtil.h 
)

if(MX_WITHCUDA) 
  list(APPEND SOURCES 
    MxCCUDA.cpp
  )
  list(APPEND HEADERS
    MxCCUDA.h
  )
endif()

add_subdirectory(models)
MX_MODEL_TREE_INIT()
MX_MODEL_TREE_PROC(models)
set(SOURCES ${SOURCES} ${MX_MODEL_SRCS_LOCAL})

add_library(mechanica_c SHARED ${SOURCES} ${HEADERS})

target_compile_definitions(mechanica_c PRIVATE C_BUILDING_DLL)

set_target_properties(mechanica_c PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1 
  POSITION_INDEPENDENT_CODE 1
)

if(MX_WITHCUDA)
  target_include_directories(mechanica_c PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  set_target_properties(mechanica_c PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

target_link_libraries(mechanica_c PUBLIC 
  Mechanica::Mechanica 
)
target_link_libraries(mechanica_c PRIVATE 
  mechanica_obj 
  mdcore_single 
  Mechanica::MagnumLibs
)

if(MX_WITHCUDA)
  target_link_libraries(mechanica_c PUBLIC 
    CUDA::cudart 
    CUDA::cuda_driver 
    CUDA::nvrtc
  )
endif()

add_library(Mechanica::CLib ALIAS mechanica_c)

install(TARGETS mechanica_c 
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" 
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" 
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(FILES ${HEADERS} DESTINATION ${MX_WRAPS_C_INSTALL_PREFIX})
