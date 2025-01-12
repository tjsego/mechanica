# Fixing seeming libxml2 bug not finding unicode headers on osx
if(MX_APPLE)
  include_directories(${CMAKE_FIND_ROOT_PATH}/include)
endif()

# build libsbml

set(LIBSBML_SHARED_VERSION OFF CACHE BOOL "dont build shared libsbml")

set(WITH_CPP_NAMESPACE ON CACHE BOOL "Use a C++ namespace for libSBML.")

set(WITH_PYTHON OFF CACHE BOOL "Generate the Python language interface for libSBML." )

set(WITH_SWIG OFF CACHE BOOL "Regenerate the programming language interface code (for Java, Python, etc.) using SWIG.")

set(LIBSBML_ROOT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libsbml" CACHE PATH "Path to the libSBML root source directory" FORCE)

set(LIBSBML_ROOT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/libsbml" CACHE PATH "Path to the libSBML root build directory" FORCE)

# set(LIBSBML_DEPENDENCY_DIR "${CMAKE_FIND_ROOT_PATH}" CACHE PATH "Directory containing libraries that libSBML depends upon. (Particularly important on Windows.)" FORCE)

set(LIBSBML_SKIP_SHARED_LIBRARY "ON")

# If necessary, confine doc files to share, rather than in install root
if(NOT UNIX AND NOT CYGWIN)
  function(libsbml_doc_change variable access value current stack)
    set(MISC_PREFIX "./share/sbml" PARENT_SCOPE)
  endfunction()

  variable_watch(MISC_PREFIX libsbml_doc_change)
endif()

add_subdirectory(libsbml)

# create an interface target for libsbml as a modern cmake target

add_library(libsbml-static-interface INTERFACE)

# libsbml names libs differently, why???
if(UNIX OR CYGWIN)
  set(LIBSBML_LIBRARY sbml)
else()
  if(MINGW)
    set(LIBSBML_LIBRARY sbml)
  else()
    set(LIBSBML_LIBRARY libsbml)
  endif()
endif()

set_property(TARGET ${LIBSBML_LIBRARY}-static PROPERTY C_STANDARD 11)
set_property(TARGET ${LIBSBML_LIBRARY}-static PROPERTY CXX_STANDARD 14)

target_link_libraries(libsbml-static-interface INTERFACE ${LIBSBML_LIBRARY}-static)

target_compile_definitions(libsbml-static-interface INTERFACE LIBSBML_STATIC)

target_include_directories(libsbml-static-interface INTERFACE "$<TARGET_PROPERTY:${LIBSBML_LIBRARY}-static,INCLUDE_DIRECTORIES>")

add_library(LibSBML::LibSBML ALIAS libsbml-static-interface)

install(TARGETS libsbml-static-interface 
  EXPORT libsbml-static-interface_targets 
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" 
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" 
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(EXPORT libsbml-static-interface_targets 
  FILE Mechanica-libSBMLTargets.cmake 
  NAMESPACE Mechanica:: 
  DESTINATION ${MX_SHARE_DIR} 
  EXPORT_LINK_INTERFACE_LIBRARIES 
  COMPONENT libSBML
)

# build glfw here, set up variables so magnum does not try to find it

set(WITH_PNGIMPORTER OFF CACHE BOOL "build PNG importer"    FORCE)


set(GLFW_BUILD_DOCS OFF CACHE BOOL "stuff" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "stuff" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "stuff" FORCE)
set(GLFW_INSTALL ON CACHE BOOL "stuff" FORCE)

add_subdirectory(glfw)


# glfw does not set it's target include directories, but it turns out we
# can set them after the glfw subproject is processed.
# This approach enables us to simply target_link_libraries(MyProgram glfw), and
# all the glfw include directories automatically get added to the MyProgram.
target_include_directories(glfw PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glfw/include/GLFW>
  )

target_include_directories(glfw PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glfw/deps>
  )

if(APPLE)
  set_property(TARGET glfw PROPERTY INSTALL_NAME_DIR )
endif()

# magnum likes these names:
#  GLFW::GLFW               - GLFW imported target
#  GLFW_LIBRARY             - GLFW library
#  GLFW_INCLUDE_DIR         - Root include dir
#
add_library(GLFW::GLFW ALIAS glfw)

set(GLFW_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/glfw/include/GLFW)
set(GLFW_FOUND TRUE)


message("GLFW_VERSION_MAJOR: ${GLFW_VERSION_MAJOR}")
message("GLFW_VERSION_MINOR: ${GLFW_VERSION_MINOR}")
message("GLFW_VERSION_PATCH: ${GLFW_VERSION_PATCH}")
message("GLFW_VERSION: ${GLFW_VERSION}")


message("GLFW_INCLUDE_DIR: ${GLFW_INCLUDE_DIR}")
message("GLFW_LIBRARY: ${GLFW_LIBRARY}")

if(TARGET glfw)
  message("glfw is a target")
  get_target_property(GLFW_VERSION glfw VERSION)
  message("GLFW_VERSION: ${GLFW_VERSION}")

else()
  message("glfw is NOT a target")
endif()

#add_subdirectory(soil)

#set(MDCORE_USE_FFTW3 ON CACHE BOOL "Override option" FORCE)
#option(MDCORE_USE_FFTW3 "use FFTW" YES)



# Disable building extras we won't need (pure C++ project)
# Add the configurations from nanogui
# add_subdirectory(nanogui)
#set(NANOGUI_BUILD_PYTHON  OFF CACHE BOOL " " FORCE)
#set(NANOGUI_INSTALL       OFF CACHE BOOL " " FORCE)
#set(NANOGUI_BUILD_SHARED  OFF CACHE BOOL " " FORCE)


# For reliability of parallel build, make the NanoGUI targets dependencies
#set_property(TARGET glfw glfw_objects nanogui PROPERTY FOLDER "dependencies")

# add_subdirectory(Voroxx)



# magnum / corrade want this
# stop it squawkign about global weak symbol thing. 
set(BUILD_STATIC_UNIQUE_GLOBALS OFF CACHE BOOL
    "Do somethign with symbols to make them work on Windows" FORCE)

add_subdirectory(corrade)

# set magnum options
set(WITH_GLFWAPPLICATION ON CACHE BOOL "build magnum with glfw" FORCE)
set(BUILD_STATIC ON CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_PLUGINS_STATIC ON CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_TESTS OFF CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_GL_TESTS OFF CACHE BOOL "build as static libraries for now..." FORCE)

# magnum plugins for mesh import testing
set(WITH_ASSIMPIMPORTER    OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_OBJIMPORTER       OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_OPENGEXIMPORTER   OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_ANYIMAGECONVERTER OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_AUDIO             OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_ANYSCENEIMPORTER  OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_PNGIMPORTER       OFF CACHE BOOL "build PNG importer"    FORCE)

if(APPLE)
  message("building on apple")

  set(WITH_CGLCONTEXT ON CACHE BOOL "Build CglContext library" FORCE)

  set(WITH_WINDOWLESSCGLAPPLICATION ON CACHE BOOL
    "Build WindowlessCglApplication library" FORCE)

elseif(UNIX AND NOT APPLE)
  # for Linux, BSD, Solaris, Minix
  message("building on linux")

  set(WITH_WINDOWLESSEGLAPPLICATION ON CACHE BOOL
    "Build WindowlessEglApplication library" FORCE)

  set(WITH_WINDOWLESSGLXAPPLICATION ON CACHE BOOL
    "Build WindowlessGlxApplication library" FORCE)

elseif(WIN32)

  set(WITH_WINDOWLESSWGLAPPLICATION ON CACHE BOOL
    "Build WindowlessWglApplication library" FORCE)

endif()

set(WITH_TGAIMAGECONVERTER ON CACHE BOOL "Build TGA Image Converter" FORCE)

add_subdirectory(magnum)

set(WITH_STBIMAGECONVERTER ON CACHE BOOL "Build Stb Image Converter" FORCE)

add_subdirectory(magnum-plugins EXCLUDE_FROM_ALL)

# SUNDIALS options
set(BUILD_ARKODE         OFF CACHE BOOL "Build this SUNDIALS feature" FORCE)
set(BUILD_CVODE          ON  CACHE BOOL "Build this SUNDIALS feature" FORCE)
set(BUILD_CVODES         OFF CACHE BOOL "Build this SUNDIALS feature" FORCE)
set(BUILD_IDA            OFF CACHE BOOL "Build this SUNDIALS feature" FORCE)
set(BUILD_IDAS           OFF CACHE BOOL "Build this SUNDIALS feature" FORCE)
set(BUILD_KINSOL         OFF CACHE BOOL "Build this SUNDIALS feature" FORCE)

# add_subdirectory(sundials)

#add_library(DirectXMath INTERFACE)

#target_compile_options(DirectXMath INTERFACE -fms-extensions)

#target_include_directories(DirectXMath INTERFACE
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/DirectXMath/Inc/>
#  #$<INSTALL_INTERFACE:include/mylib>  # <prefix>/include/mylib
#)
