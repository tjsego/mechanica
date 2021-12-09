.. Mechanica documentation master file, created by
   sphinx-quickstart on Tue Oct 16 13:21:40 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

************************
Mechanica Documentation
************************

.. image:: sphere-net.png
   :width: 50%
   :alt: alternate text
   :align: center

Mechanica is an interactive, particle-based physics, chemistry and biology
modeling and simulation environment. Mechanica provides the ability to create,
simulate and explore models, simulations and virtual experiments of soft condensed
matter physics at mulitple scales using a simple, intuitive interface. Mechanica
is designed with an emphasis on problems in complex subcellular, cellular and tissue
biophysics. Mechanica enables interactive work with simulations on heterogeneous
computing architectures, where models and simulations can be built and interacted
with in real-time during execution of a simulation, and computations can be selectively
offloaded onto available GPUs on-the-fly.
Mechanica is part of the `Tellurium <http://tellurium.analogmachine.org>`_ project.

Mechanica is a native compiled C++ shared library that's designed to be used for model
and simulation specification in compiled C++ code. Mechanica includes an extensive
Python API that's designed to be used for model and simulation specification in
executable Python scripts, an IPython console and a Jupyter Notebook.

**Quick Summary**

This documentation provides information on a number of topics related to Mechanica.
If you're looking for something specific, refer to the following,

* To get Mechanica, refer to :ref:`Getting Mechanica <getting>`.

* To get started with Mechanica, refer to :ref:`Quickstart <quickstart>`.

* To learn about the physics and philosophy of Mechanica, refer to :ref:`Introduction <introduction>`.

* For walkthroughs, examples and other discussions, refer to :ref:`Notes <notes>`.

* For application-specific models and tools, refer to :ref:`Models <models>`.

* To dive into the code, refer to :ref:`Mechanica API Reference <api_reference>`.

.. note::

   Mechanica supports modeling and simulation in multiple programming languages.
   While many variables, classes and methods are named and behave the same across
   all supported languages, inevitably there are some differences.
   Most examples in this documentation demonstrate usage in Python, and specific
   cases where Mechanica behaves differently in a particular language are explicitly
   addressed. In general, assume that a documented example or code snippet in one
   language is the same in another unless stated otherwise. For specific details
   about Mechanica in a particular language, refer to :ref:`Mechanica API Reference`.

**Funding**

Mechanica is funded with generous support by NIBIB U24 EB028887.

**Content**

.. toctree::
   :maxdepth: 1

   getting
   quick_start
   introduction
   notes
   models/models
   api_reference
   models/api_models
   status
   history
   references


Indices and tables
##################


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

