@echo off

set current_dir=%cd%

if not exist "%MXENV%" exit 1

if not exist "%MXINSTALLDIR%" exit 1

set MXTESTS_BUILDDIR=%~dp0build

mkdir %MXTESTS_BUILDDIR%

cd %~dp0

cmake -DCMAKE_BUILD_TYPE:STRING=%MXBUILD_CONFIG% ^
      -G "Ninja" ^
      -DCMAKE_PREFIX_PATH:PATH="%MXENV%;%MXINSTALLDIR%;%MXINSTALLDIR%/lib" ^
      -DCMAKE_FIND_ROOT_PATH:PATH=%MXENV%\Library ^
      -DCMAKE_C_COMPILER:PATH=%MXENV%\Library\bin\clang-cl.exe ^
      -DCMAKE_CXX_COMPILER:PATH=%MXENV%\Library\bin\clang-cl.exe ^
      -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld ^
      -DPython_EXECUTABLE:PATH=%MXENV%\python.exe ^
      -S . ^
      -B "%MXTESTS_BUILDDIR%"
if errorlevel 1 exit 2

cmake --build "%MXTESTS_BUILDDIR%" --config %MXBUILD_CONFIG%
if errorlevel 1 exit 3

cd %current_dir%
