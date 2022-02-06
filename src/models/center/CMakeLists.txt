MX_MODEL_TREE_INIT()

add_subdirectory(cell_polarity)

MX_MODEL_TREE_PROC(cell_polarity)

install(FILES __init__.py DESTINATION ${MX_SITEPACKAGES}/mechanica/models/center)
