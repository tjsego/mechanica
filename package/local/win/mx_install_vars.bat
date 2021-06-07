@echo off

rem build configuration
set MXBUILD_CONFIG=Release

rem path to source root
set MXSRCDIR=%~dp0..\..\..

rem path to build root
set MXBUILDDIR=%MXSRCDIR%\..\mechanica_build

rem path to install root
set MXINSTALLDIR=%MXSRCDIR%\..\mechanica_install

rem path to environment root
set MXENV=%MXINSTALLDIR%\mx_env
