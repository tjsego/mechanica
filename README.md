Mechanica
=========
Mechanica is an interactive, particle-based physics, chemistry and biology
modeling and simulation environment. Mechanica provides the ability to create, 
simulate and explore models, simulations and virtual experiments of soft condensed 
matter physics at mulitple scales using a simple, intuitive interface. Mechanica 
is designed with an emphasis on problems in complex subcellular, cellular and tissue 
biophysics. Mechanica enables interactive work with simulations on heterogeneous 
computing architectures, where models and simulations can be built and interacted 
with in real-time during execution of a simulation, and computations can be 
selectively offloaded onto available GPUs on-the-fly. 
Mechanica is part of the 
[Tellurium](<http://tellurium.analogmachine.org>) project. 

Mechanica is a native compiled C++ shared library that's designed to be used for model 
and simulation specification in compiled C++ code. Mechanica includes an extensive 
Python API that's designed to be used for model and simulation specification in 
executable Python scripts, an IPython console and a Jupyter Notebook. 

## Build Status ##

|Platform | Status |
| :-----: | :----: |
| Linux   | [![Build Status](https://dev.azure.com/Mechanica-sim/Mechanica/_apis/build/status/mechanica.testing.develop?branchName=test%2Faz_develop&stageName=Local%20build%20for%20Linux)](https://dev.azure.com/Mechanica-sim/Mechanica/_build/latest?definitionId=1&branchName=test%2Faz_develop)   |
| MacOS   | [![Build Status](https://dev.azure.com/Mechanica-sim/Mechanica/_apis/build/status/mechanica.testing.develop?branchName=test%2Faz_develop&stageName=Local%20build%20for%20Mac)](https://dev.azure.com/Mechanica-sim/Mechanica/_build/latest?definitionId=1&branchName=test%2Faz_develop)     |
| Windows | [![Build Status](https://dev.azure.com/Mechanica-sim/Mechanica/_apis/build/status/mechanica.testing.develop?branchName=test%2Faz_develop&stageName=Local%20build%20for%20Windows)](https://dev.azure.com/Mechanica-sim/Mechanica/_build/latest?definitionId=1&branchName=test%2Faz_develop) |

# Installation #

## Pre-Built Binaries ##

Pre-built binaries of the latest Mechanica developments are archived at the 
[Mechanic Azure project](https://dev.azure.com/Mechanica-sim/Mechanica). 
Installing pre-built binaries requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html). 
Packages include a convenience script `mx_install_env` that installs the depencies 
of the Mechanica installation on execution. After installing the depencies 
environment, the Mechanica installation can be used after executing the following steps 
from a terminal. 

On Windows
```bash
call etc/mx_vars
conda activate %MXENV%
```
On Linux and MacOS
```bash
source etc/mx_vars.sh
conda activate $MXENV
```

Launching the provided examples are then as simple as the following

```bash
python examples/cell_sorting.py
```

Likewise Mechanica can be imported in Python scripts and interactive consoles

```python
import mechanica as mx
```

## From Source ##

Supported installation from source uses Git and Miniconda for building and installing 
most dependencies. In addition to requiring [Git](https://git-scm.com/downloads) and 
[Miniconda](https://docs.conda.io/en/latest/miniconda.html), installation from source 
on Windows requires 
[Visual Studio 2019 Build Tools](https://visualstudio.microsoft.com/downloads/), 
and on Linux requires the Mesa packages `libgl1-mesa-dev` and `libegl1-mesa-dev`. 

To execute the standard installation, open a terminal in a directory to install Mechanica
and clone this respository,
```bash
git clone --recurse-submodules https://github.com/tjsego/mechanica
```

From the directory containing the `mechanica` root directory, perform the following.

On Windows 
```bash
call mechanica/package/local/mx_install
```
On Linux
```bash
bash mechanica/package/local/mx_install.sh
```
 
The standard installation will create the directories `mechanica_build` and 
`mechanica_install` next to the `mechanica` root directory, the former containing 
the build files, and the latter containing the installed binaries and conda environment. 
The source and build directories can be safely deleted after installation. 
The conda environment will be installed in the subdirectory `mx_env`. 
To activate the conda environment with the Mechanica Python module, perform the following. 

On Windows
```bash
call mechanica_install/etc/mx_vars
conda activate %MXENV%
```
On Linux
```bash
source mechanica_install/etc/mx_vars.sh
conda activate $MXENV
```

Launching the provided examples are then as simple as the following

```bash
python mechanica/examples/cell_sorting.py
```

Likewise Mechanica can be imported in Python scripts and interactive consoles

```python
import mechanica as mx
```

### Customizing the Build ###

Certain aspects of the installation can be readily customized. 
The source directory `mechanica/package/local` contains subdirectories `linux` and 
`win` containing scripts `mx_install_vars.sh` and `mx_install_vars.bat` for Linux and 
Windows, respectively, which declare default installation environment variables. 
These environment variables can be customized to specify where to find, build and install 
Mechanica, as well as the build configuration. 
For example, to install Mechanica from a source directory `MYMXSRC`, build Mechanica 
at path `MYMXBUILD` in debug mode and install into directory `MYMXINSTALL`, perform the following. 

On Windows
```bash
call %MYMXSRC%/package/local/win/mx_install_vars
set MXBUILD_CONFIG=Debug
set MXSRCDIR=%MYMXSRC%
set MXBUILDDIR=%MYMXBUILD%
set MXINSTALLDIR=%MYMXINSTALL%
call %MXSRCDIR%/package/local/win/mx_install_env
conda activate %MXENV%
call %MXSRCDIR%/package/local/win/mx_install_all
```
On Linux
```bash
source $MYMXSRC/package/local/linux/mx_install_vars.sh
export MXBUILD_CONFIG=Debug
export MXSRCDIR=$MYMXSRC
export MXBUILDDIR=$MYMXBUILD
export MXINSTALLDIR=$MYMXINSTALL
bash ${MXSRCDIR}/package/local/linux/mx_install_env.sh
conda activate $MXENV
bash ${MXSRCDIR}/package/local/linux/mx_install_all.sh
```

The default Python version of the installation is 3.7, though Mechanica has also been tested 
on Windows and Linux for Python versions 3.8 and 3.9. 
To specify a different version of Python, simply add a call to 
[update the conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html#updating-or-upgrading-python) 
in the previous commands before calling `mx_install_all`. 

### Enabling Interactive Mechanica ###

Mechanica supports interactive modeling and simulation specification in an 
IPython console and Jupyter Notebook. To enable interactive Mechanica in an 
IPython console, activate the installed environment as previously described and 
install the `ipython` package from the conda-forge channel, 

```bash
conda install -c conda-forge ipython
```

To enable interactive Mechanica in a Jupyter Notebook, activate the installed 
environment as previously described and install the `notebook`, `ipywidgets` and 
`ipyevents` packages from the conda-forge channel, 

```bash
conda install -c conda-forge notebook ipywidgets ipyevents
```
