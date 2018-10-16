# -*- coding: utf-8 -*-
import os
import sys
from shutil import fnmatch
from subprocess import check_output, CalledProcessError
from hashlib import md5
from IPython.display import SVG
from IPython.core.magic import Magics, magics_class, line_cell_magic


__author__ = """John Bjorn Nelson"""
__email__ = 'jbn@abreka.com'
__version__ = '0.0.3'


def fetch_or_compile_svg(src, prefix='', cleanup=True):
    output_path = prefix + md5(src.encode()).hexdigest()
    svg_path = output_path + ".svg"

    if not os.path.exists(svg_path):
        tex_path = output_path + ".tex"
        pdf_path = output_path + ".pdf"

        with open(tex_path, "w") as fp:
            fp.write(src)

        try:
            check_output(["pdflatex", tex_path])
            check_output(["pdf2svg", pdf_path, svg_path])
        except CalledProcessError as e:
            for del_file in [tex_path, pdf_path, svg_path]:
                if os.path.exists(del_file):
                    os.unlink(del_file)
            print(e.output.decode(), file=sys.stderr)
            return

        if cleanup:
            keep_files = {svg_path, tex_path}
            for file_path in fnmatch.filter(os.listdir(), output_path + "*"):
                if file_path not in keep_files:
                    os.unlink(file_path)

    with open(svg_path, "r") as fp:
        return SVG(fp.read())


@magics_class
class MyMagics(Magics):

    @line_cell_magic
    def itikz(self, line, cell=None):
        src = cell

        if cell is None:
            d, k = self.shell.user_ns, line.strip()

            if not line or k not in d:
                print("Line magic usage: `%itikz variable`", sys.stderr)
                return

            src = d[k]

        return fetch_or_compile_svg(src)


def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    # You can register the class itself without instantiating it.  IPython will
    # call the default constructor on it.
    ipython.register_magics(MyMagics)
