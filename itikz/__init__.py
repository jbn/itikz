# -*- coding: utf-8 -*-
import argparse
import os
import shlex
import sys
import tempfile
from hashlib import md5
from shutil import fnmatch
from string import Template
from subprocess import check_output, CalledProcessError
from IPython.display import SVG
from IPython.core.magic import Magics, magics_class, line_cell_magic


__author__ = """John Bjorn Nelson"""
__email__ = 'jbn@abreka.com'
__version__ = '0.0.7'


IMPLICIT_PIC_TMPL = Template(r"""\documentclass[tikz]{standalone}
$extras
\begin{document}
\begin{tikzpicture}[scale=$scale]
$src
\end{tikzpicture}
\end{document}""")


IMPLICIT_STANDALONE = Template(r"""\documentclass{standalone}
$extras
\begin{document}
$src
\end{document}""")


def parse_args(line):
    parser = argparse.ArgumentParser(description='Tikz to tex to SVG')

    parser.add_argument('k', type=str, nargs='?',
                        help='the variable in IPython with the string source')

    parser.add_argument('--temp-dir', dest='temp_dir', action='store_true',
                        default=False,
                        help='emit artifacts to system temp dir')

    parser.add_argument('--file-prefix', dest='file_prefix',
                        default='', help='emit artifacts with a path prefix')

    parser.add_argument('--implicit-pic', dest='implicit_pic',
                        action='store_true', default=False,
                        help='wrap source in implicit tikzpicture document')

    parser.add_argument('--implicit-standalone', dest='implicit_standalone',
                        action='store_true', default=False,
                        help='wrap source in implicit document')

    parser.add_argument('--scale', dest='scale',
                        default='1',
                        help='Set tikzpicture scale in --implicit-pic tmpl')

    parser.add_argument('--tikz-libraries', dest='tikz_libraries',
                        default='',
                        help='Comma separated list of tikz libraries to use')

    parser.add_argument('--tex-packages', dest='tex_packages',
                        default='',
                        help='Comma separated list of tex packages to use')

    return parser, parser.parse_args(shlex.split(line))


def get_cwd(args):
    if args.temp_dir or os.environ.get('ITIKZ_TEMP_DIR'):
        cwd = os.path.join(tempfile.gettempdir(), 'itikz')
        os.makedirs(cwd, exist_ok=True)
        return cwd
    else:
        return None


def fetch_or_compile_svg(src, prefix='', working_dir=None, cleanup=True):
    output_path = prefix + md5(src.encode()).hexdigest()
    if working_dir is not None:
        output_path = os.path.join(working_dir, output_path)
    svg_path = output_path + ".svg"

    if not os.path.exists(svg_path):
        tex_path = output_path + ".tex"
        pdf_path = output_path + ".pdf"

        with open(tex_path, "w") as fp:
            fp.write(src)

        try:
            check_output(["pdflatex", tex_path], cwd=working_dir)
            check_output(["pdf2svg", pdf_path, svg_path], cwd=working_dir)
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
        parser, args = parse_args(line)

        if cell is None:
            d = self.shell.user_ns

            if args.k is None or args.k not in d:
                parser.print_usage(file=sys.stderr)
                return

            src = d[args.k]

        if args.implicit_pic and args.implicit_standalone:
            print("Can't use --implicit-standalone and --implicit-pic",
                  file=sys.stderr)
        elif args.implicit_pic:
            src = IMPLICIT_PIC_TMPL.substitute(build_template_args(src, args))
        elif args.implicit_standalone:
            tmpl_args = build_template_args(src, args)
            src = IMPLICIT_STANDALONE.substitute(tmpl_args)

        return fetch_or_compile_svg(src, args.file_prefix, get_cwd(args))


def build_template_args(src, args):
    extras = []

    if args.tex_packages:
        extras.append(r"\usepackage{" + args.tex_packages + "}")

    if args.tikz_libraries:
        extras.append(r"\usetikzlibrary{" + args.tikz_libraries + "}")

    extras = "\n".join(extras)

    return dict(src=src,
                scale=float(args.scale),
                extras=extras)



def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    # You can register the class itself without instantiating it.  IPython will
    # call the default constructor on it.
    ipython.register_magics(MyMagics)
