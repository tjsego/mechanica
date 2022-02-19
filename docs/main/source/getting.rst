.. _getting:

Getting Mechanica
==================

Pre-Built Binaries
-------------------

Pre-built binaries of the latest Mechanica developments are archived at the
`Mechanica Azure project <https://dev.azure.com/Mechanica-sim/Mechanica>`_.
Installing pre-built binaries requires `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.
Packages include a convenience script `mx_install_env` that installs the dependencies
of the Mechanica installation on execution. After installing the dependencies
environment, the Mechanica installation can be used after executing the following steps
from a terminal with the root of the installation as the current directory.

On Windows

.. code-block:: bat

    call etc/mx_vars
    conda activate %MXENV%

On Linux and MacOS

.. code-block:: bash

    source etc/mx_vars.sh
    conda activate $MXENV

Launching the provided examples are then as simple as the following

.. code-block:: bash

    python examples/cell_sorting.py

Likewise Mechanica can be imported in Python scripts and interactive consoles

.. code-block:: python

    import mechanica as mx


Installing From Source
-----------------------

Supported installation from source uses Git and Miniconda for building and installing
most dependencies. In addition to requiring `Git <https://git-scm.com/downloads>`_ and
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, installation from source
on Windows requires
`Visual Studio 2019 Build Tools <https://visualstudio.microsoft.com/downloads/>`_,
on Linux requires the Mesa packages `libgl1-mesa-dev` and `libegl1-mesa-dev`,
and on MacOS requires Xcode with 10.9 SDK or greater.

To execute the standard installation, open a terminal in a directory to install Mechanica
and clone the `Mechanica repository <https://github.com/tjsego/mechanica>`_,

.. code-block:: bash

    git clone --recurse-submodules https://github.com/tjsego/mechanica

From the directory containing the `mechanica` root directory, perform the following.

On Windows

.. code-block:: bat

    call mechanica/package/local/mx_install

On Linux

.. code-block:: bash

    bash mechanica/package/local/mx_install.sh

On MacOS, specify the installed MacOS SDK (*e.g.*, for 10.9)

.. code-block:: bash

    export MXOSX_SYSROOT=10.9
    bash mechanica/package/local/mx_install.sh

The standard installation will create the directories `mechanica_build` and
`mechanica_install` next to the `mechanica` root directory, the former containing
the build files, and the latter containing the installed binaries and conda environment.
The source and build directories can be safely deleted after installation.
The conda environment will be installed in the subdirectory `mx_env`.
To activate the conda environment with the Mechanica Python module, perform the following.

On Windows

.. code-block:: bat

    call mechanica_install/etc/mx_vars
    conda activate %MXENV%

On Linux and MacOS

.. code-block:: bash

    source mechanica_install/etc/mx_vars.sh
    conda activate $MXENV

Launching the provided examples are then as simple as the following

.. code-block:: bash

    python mechanica/examples/cell_sorting.py

Likewise Mechanica can be imported in Python scripts and interactive consoles

.. code-block:: python

    import mechanica as mx


.. _customizing_the_build:

Customizing the Build
^^^^^^^^^^^^^^^^^^^^^^

Certain aspects of the installation can be readily customized.
The source directory `mechanica/package/local` contains subdirectories `linux` and
`win` containing scripts `mx_install_vars.sh` and `mx_install_vars.bat` for Linux/MacOS and
Windows, respectively, which declare default installation environment variables.
These environment variables can be customized to specify where to find, build and install
Mechanica, as well as the build configuration.
For example, to install Mechanica from a source directory ``MYMXSRC``, build Mechanica
at path ``MYMXBUILD`` in debug mode and install into directory ``MYMXINSTALL``, perform the
following.

On Windows

.. code-block:: bat

    call %MYMXSRC%/package/local/win/mx_install_vars
    set MXBUILD_CONFIG=Debug
    set MXSRCDIR=%MYMXSRC%
    set MXBUILDDIR=%MYMXBUILD%
    set MXINSTALLDIR=%MYMXINSTALL%
    call %MXSRCDIR%/package/local/win/mx_install_env
    conda activate %MXENV%
    call %MXSRCDIR%/package/local/win/mx_install_all

On Linux

.. code-block:: bash

    source $MYMXSRC/package/local/linux/mx_install_vars.sh
    export MXBUILD_CONFIG=Debug
    export MXSRCDIR=$MYMXSRC
    export MXBUILDDIR=$MYMXBUILD
    export MXINSTALLDIR=$MYMXINSTALL
    bash ${MXSRCDIR}/package/local/linux/mx_install_env.sh
    conda activate $MXENV
    bash ${MXSRCDIR}/package/local/linux/mx_install_all.sh

On MacOS

.. code-block:: bash

    source $MYMXSRC/package/local/osx/mx_install_vars.sh
    export MXBUILD_CONFIG=Debug
    export MXSRCDIR=$MYMXSRC
    export MXBUILDDIR=$MYMXBUILD
    export MXINSTALLDIR=$MYMXINSTALL
    bash ${MXSRCDIR}/package/local/osx/mx_install_env.sh
    conda activate $MXENV
    bash ${MXSRCDIR}/package/local/osx/mx_install_all.sh

The default Python version of the installation is 3.7, though Mechanica has also been tested
on Windows, Linux and MacOS for Python versions 3.8 and 3.9.
To specify a different version of Python, simply add a call to
`update the conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html#updating-or-upgrading-python>`_
in the previous commands before calling `mx_install_all`.


Enabling Interactive Mechanica
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mechanica supports interactive modeling and simulation specification in an
IPython console and Jupyter Notebook. To enable interactive Mechanica in an
IPython console, activate the installed environment as previously described and
install the ``ipython`` package from the conda-forge channel,

.. code-block:: bash

    conda install -c conda-forge ipython

To enable interactive Mechanica in a Jupyter Notebook, activate the installed
environment as previously described and install the ``notebook``, ``ipywidgets`` and
``ipyevents`` packages from the conda-forge channel,

.. code-block:: bash

    conda install -c conda-forge notebook ipywidgets ipyevents


Enabling GPU Acceleration
^^^^^^^^^^^^^^^^^^^^^^^^^^
Mechanica supports GPU acceleration on Windows and Linux using CUDA. To enable GPU
acceleration, simply tell Mechanica to build with CUDA support and specify the compute
capability of all available GPUs in the typical way *before* calling `mx_install`.

On Windows

.. code-block:: bat

    set MX_WITHCUDA=1
    set CUDAARCHS=35;50
    call mechanica/package/local/mx_install

On Linux

.. code-block:: bash

    export MX_WITHCUDA=1
    export CUDAARCHS=35;50
    bash mechanica/package/local/mx_install.sh

.. note::

    Mechanica currently supports offloading computations onto CUDA-supporting GPU devices
    of compute capability 3.5 or greater and installed drivers of at least 456.38 on Windows, and
    450.80.02 on Linux.


Setting Up a Development Environment
-------------------------------------

The Mechanica codebase includes convenience scripts to quickly set up a
development environment for building models and extensions in C++. The same
environment deployed in `Installing From Source`_ can be used to build a customized
version of Mechanica. Set up for setting up a development environment is as simple
as getting the Mechanica source code, and installing the pre-configured conda
environment. As such, all requirements described in `Installing From Source`_ are
also applicable for building a custom version of Mechanica.

To set up a development environment, clone the
`Mechanica repository <https://github.com/tjsego/mechanica>`_, open a terminal
in the directory containing the `mechanica` root directory and perform the following.

On Windows

.. code-block:: bat

    call mechanica/package/local/win/mx_install_vars
    call mechanica/package/local/win/mx_install_env

On Linux

.. code-block:: bash

    bash mechanica/package/local/linux/mx_install_vars.sh
    bash mechanica/package/local/linux/mx_install_env.sh

The standard configuration will set the build and installation directories to
`mechanica_build` and `mechanica_install` next to the `mechanica` root directory,
respectively, the latter containing the conda environment with the build dependencies.
These locations can be customized in the same way as described in `Customizing the Build`_,
or in your favorite IDE. For configuring `CMake <https://cmake.org/>`_, refer to the
script `mx_install_core` in the subdirectory of `package/local/*` that corresponds to
your platform, which is the script behind the automated installation from source.
This script includes all variables and the compiler(s) that correspond to building a
fully customized version of Mechanica.

Mechanica currently supports the `Release`, `Debug` and `RelWithDebInfo` build types. The
computational core of Mechanica and C++ front-end can be found throughout the subdirectory
`src`. Bindings for other supported languages are generated using
`SWIG <http://swig.org/>`_. To develop the interface of any other supported language
(or generate support for a new one), refer to the SWIG script `src/mechanica.i`.
