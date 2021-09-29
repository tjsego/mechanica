MX_MODEL_TREE_INIT()
MX_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/cell_polarity.cpp)
MX_MODEL_TREE_HDR(center/cell_polarity/cell_polarity.h)

install(FILES __init__.py DESTINATION ${MX_SITEPACKAGES_REL}/mechanica/models/center/cell_polarity)
install(FILES 
    examples/hollowball.py 
    examples/pancakes.py 
    examples/tubes.py 
    DESTINATION 
    ${MX_SITEPACKAGES_REL}/mechanica/examples/models/center/cell_polarity
)
