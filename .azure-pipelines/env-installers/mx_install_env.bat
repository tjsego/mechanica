@echo off

set MXENV=%~dp0mx_env

call conda create --yes --prefix %MXENV%
if errorlevel 1 exit 1

call conda env update --prefix %MXENV% --file %~dp0mx_rtenv.yml
if errorlevel 1 exit 1
