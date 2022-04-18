find_library(mechanica_clib 
    PATHS ${MX_INSTALL_ROOT} 
    PATH_SUFFIXES bin lib 
    NAMES mechanica_c libmechanica_c
    REQUIRED
)

set(MxCTest_LIBS ${mechanica_clib})
if(UNIX)
    list(APPEND MxCTest_LIBS m)
endif()

set(MxCTest_INCLUDES 
    ${MX_INSTALL_ROOT}/include/mechanica 
    ${MX_INSTALL_ROOT}/include/mechanica/wraps/C 
)

macro(MxCTest_Build src_name)
    add_executable(${src_name} ${src_name}.c)
    target_link_libraries(${src_name} ${MxCTest_LIBS})
    target_include_directories(${src_name} PUBLIC ${MxCTest_INCLUDES})
endmacro()

file(GLOB MxCTest_SRCS MxCTest_*.c)
foreach(MxCTest_SRC ${MxCTest_SRCS})
    cmake_path(GET MxCTest_SRC FILENAME MxCTest_SRC_FN)
    cmake_path(GET MxCTest_SRC_FN STEM LAST_ONLY MxCTest_SRC_NAME)
    MxCTest_Build(${MxCTest_SRC_NAME})
    add_test(NAME ${MxCTest_SRC_NAME} COMMAND ${MxCTest_SRC_NAME})
endforeach()
