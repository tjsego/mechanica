if(MX_WITH_PYTHON)

    # Determine install locations w.r.t. python module root
    file(REAL_PATH "${MX_SITEPACKAGES}" MX_SITEPACKAGES_REAL)
    file(RELATIVE_PATH MXPY_RPATH_BIN 
        ${MX_SITEPACKAGES_REAL}/mechanica 
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
    )
    set(MXPY_RPATH_BIN '${MXPY_RPATH_BIN}')
    file(RELATIVE_PATH MXPY_RPATH_LIB 
        ${MX_SITEPACKAGES_REAL}/mechanica 
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    )
    set(MXPY_RPATH_LIB '${MXPY_RPATH_LIB}')

    # These populate the python configuration script; overwite them accordingly per platform and package
    set(MXPY_RPATH_BIN_SITE None)
    set(MXPY_RPATH_LIB_SITE None)

    # Select package based on settings

    if(MX_PACKAGE_LOCAL)
        add_subdirectory(local)
    endif()
    if(MX_PACKAGE_CONDA)
        add_subdirectory(conda)
    endif()

endif(MX_WITH_PYTHON)