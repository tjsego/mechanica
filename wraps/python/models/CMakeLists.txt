add_subdirectory(center)
add_subdirectory(vertex)

install(FILES __init__.py DESTINATION ${MX_SITEPACKAGES}/mechanica/models)
