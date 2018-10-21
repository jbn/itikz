=====
itikz
=====


.. image:: https://img.shields.io/pypi/v/itikz.svg
        :target: https://pypi.python.org/pypi/itikz

.. image:: https://travis-ci.org/jbn/itikz.svg?branch=master
        :target: https://travis-ci.org/jbn/itikz

.. image:: https://img.shields.io/coveralls/github/jbn/itikz.svg
        :target: https://coveralls.io/github/jbn/itikz

Cell magic for PGF/TikZ-to-SVG rendering in Jupyter

* Free software: MIT license

Basic Usage
-----------

Install it:

.. code:: sh

    pip install itikz

Load it:

.. code:: python

    %load_ext itikz

Use it:

.. code:: tex

    %%itikz --file-prefix implicit-demo- --implicit-pic
    \draw[help lines] grid (5, 5);
    \draw[fill=magenta!10] (1, 1) rectangle (2, 2);
    \draw[fill=magenta!10] (2, 1) rectangle (3, 2);
    \draw[fill=magenta!10] (3, 1) rectangle (4, 2);
    \draw[fill=magenta!10] (3, 2) rectangle (4, 3);
    \draw[fill=magenta!10] (2, 3) rectangle (3, 4);

Getting Started Guide
---------------------

`Getting Started Notebook <https://nbviewer.jupyter.org/github/jbn/itikz/blob/master/Quickstart.ipynb>`__
