@echo off

set current_dir=%cd%

call %~dp0win\mx_install_vars

if not exist "%MXSRCDIR%" exit 1

call %MXSRCDIR%\package\local\win\mx_install_env

call conda activate %MXENV%>NUL

call %MXSRCDIR%\package\local\win\mx_install_all
if errorlevel 1 exit 2

cd %current_dir%
