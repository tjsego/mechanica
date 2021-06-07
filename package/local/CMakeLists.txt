# Ensure real paths for install environment variables, so we can use relative paths during building without requiring source to execute
file(REAL_PATH "${MX_SITEPACKAGES}" MX_SITEPACKAGES_REAL)
file(REAL_PATH "${PY_ROOT_DIR}" PY_ROOT_DIR_REAL)

if(WIN32)
    set(VARFILE_DIR win)
    set(VARFILE_SUFFIX bat)
    set(MXPY_RPATH_BIN_SITE ${MXPY_RPATH_BIN})
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
install(FILES ${MX_PYPKG_DIR}/mx_config.py DESTINATION ${MX_SITEPACKAGES_REL}/mechanica)
