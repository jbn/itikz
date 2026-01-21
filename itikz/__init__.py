import argparse
import re
import os
import shlex
import sys
import tempfile
from hashlib import md5
from shutil import fnmatch
from shutil import copy as shutil_copy
from string import Template
from subprocess import check_output, CalledProcessError
from subprocess import run as run_subprocess
from IPython.display import SVG, Image
from IPython.core.magic import Magics, magics_class, line_cell_magic

from pathlib import Path
import tempfile

import logging
logger = logging.getLogger()
logger.setLevel( logging.DEBUG )

__author__ = """John Bjorn Nelson"""
__email__ = 'jbn@abreka.com'
__version__ = '0.1.7'


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

IMPLICIT_ARTICLE = Template(r"""\documentclass{article}
\pagenumbering{gobble}
$extras
\begin{document}
$src
\end{document}""")


JINJA2_ENABLED = True
try:
    import jinja2
except ImportError:  # pragma: no cover
    JINJA2_ENABLED = False


CAIROSVG_ENABLED = True
try:
    import cairosvg
except ImportError:  # pragma: no cover
    CAIROSVG_ENABLED = False


def parse_args(line):
    parser = argparse.ArgumentParser(prog="%%itikz",
                                     description='Tikz to tex to SVG',
                                     add_help=False)

    parser.add_argument('k', type=str, nargs='?',
                        help='the variable in IPython with the string source')

    parser.add_argument('--keep-file', dest='keep_file',
                        default=None,
                        help='directory to keep tex and svg file')

    parser.add_argument('--temp-dir', dest='temp_dir', action='store_true',
                        default=False,
                        help='emit artifacts to system temp dir')

    parser.add_argument('--file-prefix', dest='file_prefix',
                        default='', help='emit artifacts with a path prefix')

    parser.add_argument('--template', dest='implicit_template',
                        default=None,
                        help='wrap source in implicit document: pic, standalone or article')

    parser.add_argument('--nexec', dest='nexec',
                        default=1,
                        help='set number of engine executions in --nexec')

    parser.add_argument('--scale', dest='scale',
                        default='1',
                        help='Set tikzpicture scale in --template pic')

    parser.add_argument('--tikz-libraries', dest='tikz_libraries',
                        default='',
                        help='Comma separated list of tikz libraries to use')

    parser.add_argument('--tex-packages', dest='tex_packages',
                        default='',
                        help='Comma separated list of tex packages to use')

    parser.add_argument('--tex-program', dest='tex_program',
                        default='pdflatex',
                        help='Name of alternate LaTeX program (e.g., lualatex)')

    parser.add_argument('--use-xetex', dest='use_xetex', action='store_true',
                        default=False,
                        help='use xetex rather than pdflatex')

    parser.add_argument('--use-dvi', dest='use_dvi', action='store_true',
                        default=False,
                        help='use dvi or xdv for conversion to svg')
    parser.add_argument('--crop', dest='crop', action='store_true',
                        default=False,
                        help='use inkscape to crop the svg')

    parser.add_argument('--as-jinja', dest='as_jinja',
                        action='store_true', default=False,
                        help="Interpret the source as a jinja2 template")

    parser.add_argument('--print-jinja', dest='print_jinja',
                        action='store_true', default=False,
                        help="Print interpolated jinja2 source then bail.")

    parser.add_argument('--rasterize', dest='rasterize',
                        action='store_true', default=False,
                        help="Rasterize the svg with cairosvg")

    parser.add_argument('--debug', dest='debug',
                        action='store_true', default=False,
                        help="turn on debug mode")
    parser.add_argument('--full-error', dest='full_err',
                        action='store_true', default=False,
                        help="Emit the full error message")

    # Override help: the default does a sys.exit()
    parser.add_argument('-h', '--help', dest='print_help',
                        action='store_true', default=False,
                        help='show this help message')

    return parser, parser.parse_args(shlex.split(line))

# ==========================================================================
def get_wd(s, root=None, add_itikz=True ):
    '''build the working directory'''
    if isinstance(root, bool):
        root = None
    if root is None:
        tmp = os.environ.get( 'ITIKZ_TEMP_DIR' )
        if not tmp:
            import platform
            root = Path("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
        else:
            root = Path(tmp)
    if isinstance(s, bool):
        s = None

    if s is None:
        if add_itikz:
            root = root / 'itikz'
        root.mkdir(parents=True, exist_ok=True)
        return root

    s      = Path(s)
    l      = len( s.parts )
    sa     = Path(s).absolute()

    if sa.is_dir():
        d = sa / 'itikz' if add_itikz else sa
    elif l > 1 and s.parent.is_dir():
        d = sa / 'itikz' if add_itikz else sa
    else:
        d = root / s
        d = d / 'itikz' if add_itikz else d

    d.mkdir(parents=True, exist_ok=True)
    return d

def get_wf(s, root, sfx='.tex'):
    '''construct the full path name of a file with suffix s'''
    s = Path(s)
    p = s.parts
    if len(p) > 1:
        r = get_wd(s.parent, root, add_itikz=False)
        return r / (s.name + sfx )
    if s.is_absolute():
        return root / s.parent / (s.name + sfx)
    return root / ( "/".join( s.parts ) + sfx )

def build_commands_dict( tex_program=["pdflatex"], svg_converter=[["pdf2svg"],".pdf"], use_xetex=False, use_dvi=False, crop=False, nexec=1):
    tex_program,svg_converter,svg_crop = build_commands( tex_program, svg_converter, use_xetex, use_dvi, crop, nexec)
    return  { "tex_program":tex_program,"svg_converter":svg_converter,"svg_crop":svg_crop  }

def build_commands( tex_program=["pdflatex"], svg_converter=[["pdf2svg"],".pdf"], use_xetex=False, use_dvi=False, crop=False, nexec=1):
    '''
build_commands( tex_program=["pdflatex"], svg_converter=[["pdf2svg"],".pdf"], use_xetex=False, use_dvi=False, crop=False, nexec=1):
    '''
    if isinstance( tex_program, (list,)) is False:
        tex_program = [tex_program]

    if tex_program[0]  == "pdflatex":
        if use_xetex is True:
            if use_dvi is True:
                if nexec > 1:
                    #print("EXA  case 1")
                    ###_tex_program = ["xelatex", "--no-pdf", "--etex" ]
                    _tex_program = ["xelatex", "--no-pdf" ]
                    _svg_converter = [["dvisvgm", "--font-format=woff2", "--exact"], ".xdv"]
                else:
                    #print("EXA  case 2")
                    #_tex_program = ["latexmk", "--quiet", "--silent", "--xelatex", "--etex" ]
                    _tex_program = ["latexmk", "-quiet", "-silent", "-xelatex" ]
                    _svg_converter = [["dvisvgm", "--font-format=woff2", "--exact"], ".xdv"]
            else:
                if nexec > 1:
                    #print("EXA  case 3")
                    #_tex_program = ["xelatex", "--etex" ]
                    _tex_program = ["xelatex" ]
                    _svg_converter = [["pdf2svg"], ".pdf"]
                else:
                    #print("EXA  case 4")
                    #_tex_program = ["latexmk", "--quiet", "--silent", "--xelatex", "--etex" ]
                    _tex_program = ["latexmk", "-quiet", "-silent", "-xelatex" ]
                    _svg_converter = [["pdf2svg"], ".pdf"]
        else:
            if use_dvi is True:
                #print("EXA  case 5")
                #_tex_program = ["latexmk", "--quiet", "--silent", "--etex", "-dvi" ]
                _tex_program = ["latexmk", "-quiet", "-silent", "-dvi" ]
                _svg_converter = [["dvisvgm", "--font-format=woff2", "--exact"], ".dvi"]
            else:
                #print("EXA  case 6")
                #_tex_program = ["latexmk", "--quiet", "--silent", "--etex", "-pdf" ]
                _tex_program = ["latexmk", "-quiet", "-silent", "-pdf" ]
                _svg_converter = [["pdf2svg"], ".pdf"]
    else:
       #print("EXA  case 7")
       _tex_program = tex_program
       _svg_converter = svg_converter
    if crop:
        _svg_crop = (["inkscape", "--batch-process", "--export-plain-svg", "-D", "--export-margin=1", "-o"])
    else:
        _svg_crop = None

    return _tex_program, _svg_converter,_svg_crop

def fetch_or_compile_svg(src, prefix='', working_dir=None, full_err=False, debug=False, tex_program=["pdflatex"], svg_converter=[["pdf2svg"],".pdf"], svg_crop=None, nexec=1, keep_file=None):
    '''
fetch_or_compile_svg(src, prefix='', working_dir=None, full_err=False, debug=False, tex_program=["pdflatex"], svg_converter=[["pdf2svg"],".pdf"], svg_crop=None, nexec=1, keep_file=None):
    '''
    svg = svg_from_tex(src, prefix, working_dir, full_err, debug, tex_program, svg_converter, svg_crop, nexec, keep_file)
    if svg is not None:
        return SVG(svg)
    return None

def svg_file_from_tex(src, prefix='', working_dir=None, full_err=False, debug=False, tex_program=["pdflatex"], svg_converter=[["pdf2svg"],".pdf"], svg_crop=None, nexec=1, keep_file=None):
    '''
svg_file_from_tex(src, prefix='', working_dir=None, full_err=False, debug=False, tex_program=["pdflatex"], svg_converter=[["pdf2svg"],".pdf"], svg_crop=None, nexec=1, keep_file=None):
    '''
    ##EXA working_dir = get_working_dir(working_dir)
    working_dir = get_wd(working_dir, root=None, add_itikz=working_dir is None)

    src_hash = md5(src.encode()).hexdigest()
    output_prefix  =  prefix +src_hash

    tex_file = working_dir / (output_prefix + '.tex')
    pdf_file = working_dir / (output_prefix + '.pdf')
    svg_file = working_dir / (output_prefix + '.svg')

    if debug or not svg_file.exists():
        if debug:
            print(">>>> tex file: ", tex_file)
            print(">>>> tex code:\n", src)
        with open(tex_file, "w") as fp:
            #print( "EXA ***writing to ", tex_file, "\nsrc:   ", src)
            try:
                fp.write(src)
            except:
                print("failed to write tex source", file=sys.stderr)
                cleanup_artifacts(working_dir, src_hash)
                return
        #print("EXA file exists: ", tex_file.exists(), tex_file )

        try:
            tex_program.append( str(tex_file) )
            if debug:
                print(">>>> tex_PROGRAM: ", ' '.join(tex_program), working_dir)
            for _ in range(nexec-1):
                run_subprocess(tex_program, cwd=working_dir)
            check_output(tex_program, cwd=working_dir)

            svg_program = svg_converter[0] + [str(pdf_file), str(svg_file)]
            if debug:
                print(">>>> svg_PROGRAM: ", ' '.join(svg_program))
            check_output(svg_program, cwd=working_dir)

            if svg_crop is not None:
                crop_program = svg_crop + [str(svg_file), str(svg_file)]
                if debug:
                    print(">>>> svg_crop_PROGRAM: ", ' '.join(crop_program))
                check_output( crop_program, cwd=working_dir)

            # Could also convert SVG to PNG here
            # inkscape -z -e test.png -w 1024 -h 1024 test.svg

            if keep_file is not None:
                tex_keep_file = get_wf(keep_file, Path.cwd(), sfx=".tex")
                svg_keep_file = get_wf(keep_file, Path.cwd(), sfx=".svg")

                try:
                    if debug:
                        print(">>>> Save Files: ", ' '.join(["cp", str(tex_file), str(tex_keep_file)]))
                        print("                 ", ' '.join(["cp", str(svg_file), str(svg_keep_file)]))
                    shutil_copy( tex_file, tex_keep_file )
                    shutil_copy( svg_file, svg_keep_file )
                except:
                    print( "Could not copy files")

        except CalledProcessError as e:
            if keep_file is not None:
                tex_keep_file = get_wf(keep_file, Path.cwd(), sfx=".tex")
                svg_keep_file = get_wf(keep_file, Path.cwd(), sfx=".svg")
                try:
                    if debug:
                        print(">>>> Save Files: ", ' '.join(["cp", str(tex_file), str(tex_keep_file)]))
                        print("                 ", ' '.join(["cp", str(svg_file), str(svg_keep_file)]))
                    shutil_copy( tex_file, tex_keep_file )
                    shutil_copy( svg_file, svg_keep_file )
                except:
                    print( "Could not copy files")

            cleanup_artifacts(working_dir, src_hash)
            err_msg = e.output.decode()

            if not full_err:  # tail -n 20
                err_msg = "\n".join(err_msg.splitlines()[-20:])

            print(err_msg, file=sys.stderr)
            return

        cleanup_artifacts(working_dir, src_hash, svg_file, tex_file)

    return tex_file,svg_file

def html_img_from_tex(src, prefix='', working_dir=None, full_err=False, debug=False, tex_program=["pdflatex"], svg_converter=[["pdf2svg"],".pdf"], svg_crop=None, nexec=1, keep_file=None):

    _,svg_file = svg_file_from_tex(src, prefix, working_dir, full_err, debug, tex_program, svg_converter, svg_crop, nexec, keep_file)
    return f'<img src="{svg_file}">'

def svg_from_tex(src, prefix='', working_dir=None, full_err=False, debug=False, tex_program=["pdflatex"], svg_converter=[["pdf2svg"],".pdf"], svg_crop=None, nexec=1, keep_file=None):
    '''
svg_from_tex(src, prefix='', working_dir=None, full_err=False, debug=False, tex_program=["pdflatex"], svg_converter=[["pdf2svg"],".pdf"], svg_crop=None, nexec=1, keep_file=None):
    '''

    tex_file,svg_file = svg_file_from_tex(src, prefix, working_dir, full_err, debug, tex_program, svg_converter, svg_crop, nexec, keep_file)

    with open(svg_file, "r") as fp:
        return fp.read()

def cleanup_artifacts(working_dir, src_hash, *retaining):
    files = working_dir.glob(f'**/*{src_hash}*')
    for f in files:
        if f not in retaining:
            f.unlink()

def load_and_interpolate_jinja2(src, ns):
    # The FileSystemLoader should operate in the current working directory.
    # By assumption, extended jinja templates aren't temporary files --
    # the user wrote them by hand. They are part of code you would want in
    # your repository!
    fs_loader = jinja2.FileSystemLoader(Path.cwd())
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

        if args.implicit_template is not None:
            if args.implicit_template == "pic":
                src = IMPLICIT_PIC_TMPL.substitute(build_template_args(src, args))
            elif args.implicit_template == "standalone":
                tmpl_args = build_template_args(src, args)
                src = IMPLICIT_STANDALONE.substitute(tmpl_args)
            elif args.implicit_template == "article":
                tmpl_args = build_template_args(src, args)
                src = IMPLICIT_ARTICLE.substitute(tmpl_args)
            else:
                print( "no implicit template '", args.implicit_template, "'" )
                print( "   known templates are one of: 'pic, standalone, article', else use jinja" )
                return

        if args.debug:
            print(">>>> program args: ", args.tex_program, args.use_dvi, args.use_xetex)

        nexec = int(args.nexec)
        tex_program,svg_converter,svg_crop =  build_commands( args.tex_program, [["pdf2svg"],".pdf"], args.use_xetex, args.use_dvi, args.crop, nexec)

        #print("EXA **** working dir<<<<", get_cwd(args), ">>>>")
        working_dir = None if args.temp_dir == False else args.temp_dir
        svg = fetch_or_compile_svg(src, args.file_prefix, working_dir,
                                   args.full_err, args.debug, tex_program, svg_converter, svg_crop, nexec, args.keep_file)

        if svg is None:
            return None

        if args.rasterize:
            if not CAIROSVG_ENABLED:
                print("Please install cairosvg", file=sys.stderr)
                print("$ pip install cairosvg", file=sys.stderr)
                return

            png_bytes = cairosvg.svg2png(bytestring=svg.data.encode())

            return Image(data=png_bytes)

        return svg


def build_template_args(src, args):
    extras = []

    pattern = re.compile( r'(\[[^]]*])(.*)' )
    if args.tex_packages:
        for package in args.tex_packages.split(','):
            match = pattern.match( package )
            if match:
                extras.append( r"\usepackage" + match.group(1) + "{" + match.group(2)+ "}" )
            else:
                extras.append(r"\usepackage{" + package + "}")

    if args.tikz_libraries:
        extras.append(r"\usetikzlibrary{" + args.tikz_libraries + "}")

    extras = "\n".join(extras)

    return dict(src=src,
                scale=float(args.scale),
                extras=extras)


def load_ipython_extension(ipython):  # pragma: no cover
    ipython.register_magics(TikZMagics)
