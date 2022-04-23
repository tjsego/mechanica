add_subdirectory(event)
add_subdirectory(io)
add_subdirectory(mdcore)
add_subdirectory(models)
add_subdirectory(rendering)
add_subdirectory(state)
add_subdirectory(types)

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

set(mechanica_py_LIBS
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

  # Link against the Python shared library on Windows
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


file(COPY setup.py DESTINATION ${CMAKE_BINARY_DIR})
file(COPY __init__.py DESTINATION ${MX_PYPKG_DIR})
file(COPY lattice.py DESTINATION ${MX_PYPKG_DIR})
file(COPY jwidget.py DESTINATION ${MX_PYPKG_DIR})
file(COPY particle_type.py DESTINATION ${MX_PYPKG_DIR})

add_library(Mechanica::PythonLib ALIAS mechanica_py)

install(TARGETS mechanica_py DESTINATION ${MX_SITEPACKAGES}/mechanica)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mechanica.py DESTINATION ${MX_SITEPACKAGES}/mechanica)
install(FILES 
        __init__.py 
        lattice.py 
        jwidget.py
        particle_type.py 
        DESTINATION 
        ${MX_SITEPACKAGES}/mechanica
)
