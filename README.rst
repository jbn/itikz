=====
itikz
=====


.. image:: https://img.shields.io/pypi/v/itikz.svg
        :target: https://pypi.python.org/pypi/itikz

.. image:: https://img.shields.io/travis/jbn/itikz.svg
        :target: https://travis-ci.org/jbn/itikz

.. image:: https://readthedocs.org/projects/itikz/badge/?version=latest
        :target: https://itikz.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Cell magic for PGF/TikZ-to-SVG rendering in Jupyter


* Free software: MIT license
* Documentation: https://itikz.readthedocs.io.

Usage
-----

Install it:

.. code:: sh

    pip install itikz

Load it:

.. code:: python

    %load_ext itikz

Use it (this example from `Austin <https://notgnoshi.github.io/svg-with-tikz/>`__):

.. code:: tex

    %%itikz
    \documentclass[tikz]{standalone}

    \newcommand{\tikzAngleOfLine}{\tikz@AngleOfLine}
        \def\tikz@AngleOfLine(#1)(#2)#3{%
            \pgfmathanglebetweenpoints{%
            \pgfpointanchor{#1}{center}}{%
            \pgfpointanchor{#2}{center}}
        \pgfmathsetmacro{#3}{\pgfmathresult}%
        }

    \newcommand{\tikzMarkAngle}[3]{
        \tikzAngleOfLine#1#2{\AngleStart}
        \tikzAngleOfLine#1#3{\AngleEnd}
        \draw #1+(\AngleStart:0.4cm) arc (\AngleStart:\AngleEnd:0.4cm);
    }

    \begin{document}
        \begin{tikzpicture}
            \coordinate (O) at (0, 0);
            \coordinate (z) at (3, 3);
            \coordinate (a) at (3, 0);

            \draw [->, thick] (-1, 0) -- (5, 0);
            \draw [->, thick] (0, -1) -- (0, 5);

            \draw (O) -- (z);
            \draw (O) -- (z) node[above, midway]{$r$};
            \draw (a) -- (z) node[right, midway]{$a$};
            \draw (O) -- (a) node[below, midway]{$b$};

            \draw (z) node[circle, fill, inner sep=1pt]{} node[right]{$z$};
            \draw (O) node[left, yshift=-0.25cm]{$O$};
            \draw (O) node[xshift=0.55cm, yshift=0.2cm]{$\phi$};

            \tikzMarkAngle{(O)}{(a)}{(z)}
        \end{tikzpicture}
    \end{document}

Todo
----
Add argparse options for optional *non-removal* of artifacts.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
