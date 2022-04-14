find_package(Python 3.7.0...3.9.99 REQUIRED COMPONENTS Interpreter)

macro(MxPyTest_Test src_name src_path)
    add_test(
        NAME ${src_name} 
        COMMAND Python::Interpreter -m pytest ${src_path}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endmacro()

file(GLOB MxPyTest_SRCS MxPyTest_*.py)
foreach(MxPyTest_SRC ${MxPyTest_SRCS})
    cmake_path(GET MxPyTest_SRC FILENAME MxPyTest_SRC_FN)
    cmake_path(GET MxPyTest_SRC_FN STEM LAST_ONLY MxPyTest_SRC_NAME)
    MxPyTest_Test(${MxPyTest_SRC_NAME} ${MxPyTest_SRC})
endforeach()
