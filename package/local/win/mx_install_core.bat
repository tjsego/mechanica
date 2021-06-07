@echo off

if not exist "%MXSRCDIR%" exit 1
if not exist "%MXENV%" exit 2

set current_dir=%cd%

mkdir %MXBUILDDIR%
mkdir %MXINSTALLDIR%

cd %MXBUILDDIR%

cmake -DCMAKE_BUILD_TYPE:STRING=%MXBUILD_CONFIG% ^
      -G "Ninja" ^
      -DCMAKE_PREFIX_PATH:PATH=%MXENV% ^
      -DCMAKE_FIND_ROOT_PATH:PATH=%MXENV%\Library ^
      -DCMAKE_INSTALL_PREFIX:PATH=%MXINSTALLDIR% ^
      -DCMAKE_C_COMPILER:PATH=%MXENV%\Library\bin\clang-cl.exe ^
      -DCMAKE_CXX_COMPILER:PATH=%MXENV%\Library\bin\clang-cl.exe ^
      -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld ^
      -DPython_EXECUTABLE:PATH=%MXENV%\python.exe ^
      -DPThreads_ROOT:PATH=%MXENV%\Library ^
      "%MXSRCDIR%"
if errorlevel 1 exit 3

cmake --build . --config %MXBUILD_CONFIG% --target install
if errorlevel 1 exit 4

cd %current_dir%
