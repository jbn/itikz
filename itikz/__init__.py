# -*- coding: utf-8 -*-
import os
import sys
from shutil import fnmatch
from subprocess import check_output, CalledProcessError
from hashlib import md5
from IPython.display import SVG
from IPython.core.magic import register_cell_magic


__author__ = """John Bjorn Nelson"""
__email__ = 'jbn@abreka.com'
__version__ = '0.0.2'


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


@register_cell_magic
def itikz(line, cell):
    return fetch_or_compile_svg(cell)
