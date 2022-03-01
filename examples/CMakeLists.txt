file(GLOB PY_EXAMPLES "*.py")

file(COPY ${PY_EXAMPLES} DESTINATION ${MX_PYPKG_DIR}/examples)
install(FILES ${PY_EXAMPLES} DESTINATION ${MX_SITEPACKAGES}/mechanica/examples)

file(GLOB IPYNB_EXAMPLES "examples/*.ipynb")
file(COPY ${IPYNB_EXAMPLES} DESTINATION ${MX_PYPKG_DIR}/examples/notebooks)
install(FILES ${IPYNB_EXAMPLES} DESTINATION ${MX_SITEPACKAGES}/mechanica/examples/notebooks)
