@echo off

if not exist "%MXSRCDIR%" exit 1

call conda create --yes --prefix %MXENV%
if errorlevel 1 exit 2

call conda env update --prefix %MXENV% --file %MXSRCDIR%\package\local\win\mx_env.yml
if errorlevel 1 exit 3
