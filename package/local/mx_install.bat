@echo off

set current_dir=%cd%

call %~dp0win\mx_install_vars

if not exist "%MXSRCDIR%" exit 1

call %MXSRCDIR%\package\local\win\mx_install_env

call conda activate %MXENV%>NUL

rem Install CUDA support if requested
if defined MX_WITHCUDA (
    if %MX_WITHCUDA% == 1 (
        rem Validate specified compute capability
        if not defined CUDAARCHS (
            echo No compute capability specified (e.g., "export CUDAARCHS=35;50")
            exit 1
        )

        echo Detected CUDA support request
        echo Installing additional dependencies...

        set MXCUDAENV=%MXENV%
        call conda install -y -c nvidia cuda-toolkit>NUL
    )
)

call %MXSRCDIR%\package\local\win\mx_install_all
if errorlevel 1 exit 2

cd %current_dir%
