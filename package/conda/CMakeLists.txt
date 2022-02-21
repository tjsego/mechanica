if(WIN32)
    set(MXPY_RPATH_BIN_SITE ${MXPY_RPATH_BIN})
endif()

configure_file(python/mx_config.in.py ${MX_PYPKG_DIR}/mx_config.py @ONLY)
install(FILES ${MX_PYPKG_DIR}/mx_config.py DESTINATION ${MX_SITEPACKAGES}/mechanica)
