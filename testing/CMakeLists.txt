cmake_minimum_required (VERSION 3.20.3)
project(MechanicaTests)
include(CTest)

option(MX_WITH_C        "Test C language interface for Mechanica."      ON)
option(MX_WITH_PYTHON   "Test Python language interface for Mechanica." ON)

if(MX_WITH_C)
    add_subdirectory(C)
endif()

if(MX_WITH_PYTHON)
    add_subdirectory(python)
endif()
