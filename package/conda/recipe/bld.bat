set MXBUILD_CONFIG=Release

set MXPACKAGELOCALOFF=1
set MXPACKAGECONDA=1

mkdir mx_build_conda
cd mx_build_conda

cmake -DCMAKE_BUILD_TYPE:STRING="%MXBUILD_CONFIG%" ^
      -G "Ninja" ^
      -DCMAKE_PREFIX_PATH="%PREFIX%" ^
      -DCMAKE_FIND_ROOT_PATH="%LIBRARY_PREFIX%" ^
      -DCMAKE_INSTALL_PREFIX:PATH="%LIBRARY_PREFIX%" ^
      -DMX_INSTALL_PREFIX_PYTHON:PATH="%SP_DIR%" ^
      -DCMAKE_C_COMPILER:PATH="%LIBRARY_PREFIX%\bin\clang-cl.exe" ^
      -DCMAKE_CXX_COMPILER:PATH="%LIBRARY_PREFIX%\bin\clang-cl.exe" ^
      -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld ^
      -DPThreads_ROOT:PATH="%LIBRARY_PREFIX%" ^
      -DPython_EXECUTABLE=%PYTHON% ^
      "%SRC_DIR%"
if errorlevel 1 exit 1

cmake --build . --config "%MXBUILD_CONFIG%" --target install
if errorlevel 1 exit 1
