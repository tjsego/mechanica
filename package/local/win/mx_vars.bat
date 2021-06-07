@echo off

call %~dp0mx_site_vars

if not exist "%MXPYSITEDIR%" exit 1

set PYTHONPATH=%MXPYSITEDIR%;%PYTHONPATH%
