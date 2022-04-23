@echo off

if not exist "%MXINSTALLDIR%" exit 1

set MXTESTS_TESTSDIR=%~dp0build
set PATH=%MXINSTALLDIR%/bin;%PATH%
cd %MXTESTS_TESTSDIR%
ctest --output-on-failure
