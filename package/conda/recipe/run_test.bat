@echo off

set MXBUILD_CONFIG=Release

set MXTESTS_ROOT=%cd%/testing
set MXTESTS_BUILDDIR=%MXTESTS_ROOT%/build

mkdir %MXTESTS_BUILDDIR%

cd %MXTESTS_ROOT%

cmake -DCMAKE_BUILD_TYPE:STRING=%MXBUILD_CONFIG% ^
      -G "Ninja" ^
      -DCMAKE_PREFIX_PATH:PATH="%PREFIX%" ^
      -DCMAKE_FIND_ROOT_PATH:PATH="%LIBRARY_PREFIX%" ^
      -DCMAKE_C_COMPILER:PATH="%LIBRARY_PREFIX%"\bin\clang-cl.exe ^
      -DCMAKE_CXX_COMPILER:PATH="%LIBRARY_PREFIX%"\bin\clang-cl.exe ^
      -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld ^
      -DPython_EXECUTABLE:PATH=%PYTHON% ^
      -DMX_INSTALL_ROOT=%LIBRARY_PREFIX% ^
      -S . ^
      -B "%MXTESTS_BUILDDIR%"
if errorlevel 1 exit 2

cmake --build "%MXTESTS_BUILDDIR%" --config %MXBUILD_CONFIG%
if errorlevel 1 exit 3

cd %MXTESTS_BUILDDIR%
ctest --output-on-failure
