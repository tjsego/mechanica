file(RELATIVE_PATH MX_SITEPACKAGES_REL 
    ${CMAKE_INSTALL_PREFIX}/etc 
    ${MX_SITEPACKAGES}
)

file(RELATIVE_PATH PY_ROOT_DIR_REL
    ${CMAKE_INSTALL_PREFIX}/etc 
    ${PY_ROOT_DIR}
)

if(WIN32)
    set(VARFILE_DIR win)
    set(VARFILE_SUFFIX bat)
    set(MXPY_RPATH_BIN_SITE ${MXPY_RPATH_BIN})
elseif(APPLE)
    set(VARFILE_DIR osx)
    set(VARFILE_SUFFIX sh)
else()
    set(VARFILE_DIR linux)
    set(VARFILE_SUFFIX sh)
endif()

configure_file(${VARFILE_DIR}/mx_site_vars.in.${VARFILE_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/mx_site_vars.${VARFILE_SUFFIX} @ONLY)
install(
    FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/mx_site_vars.${VARFILE_SUFFIX} 
    ${VARFILE_DIR}/mx_vars.${VARFILE_SUFFIX} 
    DESTINATION etc
)

configure_file(python/mx_config.in.py ${MX_PYPKG_DIR}/mx_config.py @ONLY)
install(FILES ${MX_PYPKG_DIR}/mx_config.py DESTINATION ${MX_SITEPACKAGES}/mechanica)
