@echo off

set current_dir=%cd%

if not exist "%MXSRCDIR%" exit 1

call %MXSRCDIR%\package\local\win\mx_install_core
if errorlevel 1 exit 2

cd %current_dir%
