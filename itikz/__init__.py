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
__version__ = '0.1.0'


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


JINJA2_ENABLED = True
try:
    import jinja2
except ImportError:  # pragma: no cover
    JINJA2_ENABLED = False


def parse_args(line):
    parser = argparse.ArgumentParser(prog="%%itikz",
                                     description='Tikz to tex to SVG',
                                     add_help=False)

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

    parser.add_argument('--as-jinja', dest='as_jinja',
                        action='store_true', default=False,
                        help="Interpret the source as a jinja2 template")

    parser.add_argument('--print-jinja', dest='print_jinja',
                        action='store_true', default=False,
                        help="Print interpolated jinja2 source then bail.")

    # Override help: the default does a sys.exit()
    parser.add_argument('-h', '--help', dest='print_help',
                        action='store_true', default=False,
                        help='show this help message')

    return parser, parser.parse_args(shlex.split(line))


def get_cwd(args):
    if args.temp_dir or os.environ.get('ITIKZ_TEMP_DIR'):
        cwd = os.path.join(tempfile.gettempdir(), 'itikz')
        os.makedirs(cwd, exist_ok=True)
        return cwd  # Override as cwd
    else:
        return None  # No override.


def fetch_or_compile_svg(src, prefix='', working_dir=None, cleanup=True):
    src_hash = md5(src.encode()).hexdigest()
    output_path = prefix + src_hash
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
            cleanup_artifacts(working_dir, src_hash)
            print(e.output.decode(), file=sys.stderr)
            return

        if cleanup:
            cleanup_artifacts(working_dir, src_hash, svg_path, tex_path)

    with open(svg_path, "r") as fp:
        return SVG(fp.read())


def cleanup_artifacts(working_dir, src_hash, *retaining):
    glob = "*{}*".format(src_hash)

    for file_name in fnmatch.filter(os.listdir(working_dir), glob):
        file_path = os.path.join(working_dir or '', file_name)
        if file_path not in retaining:
            os.unlink(file_path)


def load_and_interpolate_jinja2(src, ns):
    # The FileSystemLoader should operate in the current working directory.
    # By assumption, extended jinja templates aren't temporary files --
    # the user wrote them by hand. They are part of code you would want in
    # your repository!
    fs_loader = jinja2.FileSystemLoader(os.getcwd())
    tmpl_env = jinja2.Environment(loader=fs_loader)

    # The final template -- the one that may extend a custom template --
    # may be in the current directory or in a temporary one. So, it's
    # passed as a string.
    tmpl = tmpl_env.from_string(src)

    return tmpl.render(**ns)


@magics_class
class TikZMagics(Magics):

    @line_cell_magic
    def itikz(self, line, cell=None):
        src = cell
        parser, args = parse_args(line)

        if args.print_help:
            parser.print_help(file=sys.stderr)
            return

        ipython_ns = self.shell.user_ns

        if cell is None:

            if args.k is None or args.k not in ipython_ns:
                parser.print_usage(file=sys.stderr)
                return

            src = ipython_ns[args.k]

        if args.implicit_pic and args.implicit_standalone:
            print("Can't use --implicit-standalone and --implicit-pic",
                  file=sys.stderr)
            return None

        # Jinja processing comes BEFORE implicit pic or standalone processing!
        if args.as_jinja:

            if not JINJA2_ENABLED:
                print("Please install jinja2", file=sys.stderr)
                print("$ pip install jinja2", file=sys.stderr)
                return

            src = load_and_interpolate_jinja2(src, ipython_ns)

            if args.print_jinja:
                print(src)
                return

        if args.implicit_pic:
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


def load_ipython_extension(ipython):  # pragma: no cover
    ipython.register_magics(TikZMagics)
