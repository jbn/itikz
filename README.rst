=====
itikz
=====


.. image:: https://img.shields.io/pypi/v/itikz.svg
        :target: https://pypi.python.org/pypi/itikz

.. image:: https://travis-ci.org/jbn/itikz.svg?branch=master
        :target: https://travis-ci.org/jbn/itikz

.. image:: https://img.shields.io/coveralls/github/jbn/itikz.svg
        :target: https://coveralls.io/github/jbn/itikz
Latex with Tikz to SVG conversion. Includes Cell magic for rendering in Jupyter

* Free software: MIT license

Basic Usage
-----------
The installation information below will install the version provided by the original author.
To use the upgraded version here, download it, and copy the itikz subdirectory into
a directory in your PYTHONPATH.

Step by step instructions for Windows are shown in the InstallationInstructions.png file.

Prerequisites:

* a working TeX installation, e.g., https://tug.org/texlive/windows.html
* inkscape   (ensure inkscape is on the path)
* pdf2svg    (https://github.com/jalios/pdf2svg-windows  enure the selected directory is on the path)

Install itikz with `python setup.py install`

To see the directories in your pythonpath, execute the following in python:

.. code:: python

   import sys
   for p in sys.path:
       print( p )

To install the original version instead, run:

.. code:: sh

    pip install itikz

Load it:

.. code:: python

    %load_ext itikz

Use it:

.. code:: tex

    %%itikz --file-prefix implicit-demo- --template pic
    \draw[help lines] grid (5, 5);
    \draw[fill=magenta!10] (1, 1) rectangle (2, 2);
    \draw[fill=magenta!10] (2, 1) rectangle (3, 2);
    \draw[fill=magenta!10] (3, 1) rectangle (4, 2);
    \draw[fill=magenta!10] (3, 2) rectangle (4, 3);
    \draw[fill=magenta!10] (2, 3) rectangle (3, 4);

Getting Started Guide
---------------------

`Getting Started Notebook <https://nbviewer.jupyter.org/github/jbn/itikz/blob/master/Quickstart.ipynb>`__

Youtube Video Introducing the nicematrix package
------------------------------------------------

`<https://www.youtube.com/watch?v=CrWpv7XlAdk>`__

API
---

Python:

..code:: build_commands, svg_from_tex, fetch_or_compile_svg

Can be invoked from Julia using PyCall, tested in jupyter-lab and Pluto.jl
