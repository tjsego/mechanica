#add_subdirectory(mxinfo)

################################################################################
####  mx-pyrun  ################################################################
################################################################################

if(MX_WITH_PYTHON)

  configure_file(mx-pyrun.h.in mx-pyrun.h)

  add_executable(
    mx-pyrun
    mx-pyrun.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/mx-pyrun.h
  )

  target_include_directories(mx-pyrun PRIVATE ${Python_INCLUDE_DIRS})

  target_include_directories(mx-pyrun PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

  if(WIN32 OR CYGWIN)

    # weird bug with CMake / clang on windows, need to add link dir here
    target_link_directories(mx-pyrun PUBLIC ${Python_LIBRARY_DIRS})

    # Link against the Python shared library on Windows
    # target_link_libraries(mechanica_py PRIVATE Python::Python)
    target_link_libraries(mx-pyrun  ${Python_LIBRARY_RELEASE})
  else()
    target_link_libraries(mx-pyrun Python::Python)
  endif()

  target_link_libraries(mx-pyrun
    Mechanica::PythonLib
  )

endif(MX_WITH_PYTHON)

################################################################################
####  mx-glinfo  ###############################################################
################################################################################


add_executable(mx-glinfo ${PROJECT_SOURCE_DIR}/extern/magnum/src/Magnum/Platform/gl-info.cpp)

set_target_properties(mx-glinfo PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
  )


target_link_libraries(mx-glinfo PRIVATE MagnumGL)

if(MAGNUM_TARGET_HEADLESS OR CORRADE_TARGET_EMSCRIPTEN OR CORRADE_TARGET_ANDROID)
    target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessEglApplication)
elseif(CORRADE_TARGET_IOS)
    target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessIosApplication)
elseif(CORRADE_TARGET_APPLE)
    target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessCglApplication)
elseif(CORRADE_TARGET_UNIX)
    if(MAGNUM_TARGET_GLES AND NOT MAGNUM_TARGET_DESKTOP_GLES)
        target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessEglApplication)
    else()
        target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessGlxApplication)
    endif()
elseif(CORRADE_TARGET_WINDOWS)
    if(MAGNUM_TARGET_GLES AND NOT MAGNUM_TARGET_DESKTOP_GLES)
        target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessWindowsEglApplication)
    else()
        target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessWglApplication)
    endif()
else()
    message(FATAL_ERROR "mx-glinfo is not available on this platform. Set WITH_GL_INFO to OFF to skip building it.")
endif()

#set_target_properties(mx-glinfo PROPERTIES FOLDER "Magnum/Platform")




