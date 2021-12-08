@echo off

set current_dir=%cd%

call %~dp0win\mx_install_vars

if not exist "%MXSRCDIR%" exit 1

call %MXSRCDIR%\package\local\win\mx_install_env

if not defined MX_WITHCUDA goto DoInstall

rem Install CUDA support if requested
if %MX_WITHCUDA% == 1 (
    rem Validate specified compute capability
    if not defined CUDAARCHS (
        echo No compute capability specified
        exit 1
    ) 
    echo Detected CUDA support request
    echo Installing additional dependencies...

    goto SetupCUDA
)

goto DoInstall

:SetupCUDA

    set MXCUDAENV=%MXENV%
    call conda install -y -c nvidia -p %MXENV% cuda>NUL
    goto DoInstall

:DoInstall
    call conda activate %MXENV%>NUL

    call %MXSRCDIR%\package\local\win\mx_install_all
    if errorlevel 1 exit 2

    cd %current_dir%
