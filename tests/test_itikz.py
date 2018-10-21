import os
import pytest
import itikz
import tempfile
from IPython.display import SVG


@pytest.mark.skipif(not itikz.JINJA2_ENABLED, reason="jinja2 not installed")
def test_load_and_interpolate_jinja2(monkeypatch, tmpdir):
    # Create the parent_tmpl.md template.
    monkeypatch.chdir(tmpdir)
    parent_src = """# Example\n{% block content -%}{% endblock -%}"""
    parent = tmpdir / "parent_tmpl.md"
    parent.write(parent_src)

    # Use it.
    pkg_name = "itikz"
    child_src = ('{% extends "parent_tmpl.md" %}'
                 "{% block content -%}{{pkg_name}}{% endblock -%}""")
    output = itikz.load_and_interpolate_jinja2(child_src, locals())

    assert output == "# Example\nitikz"


def test_get_cwd(mocker, monkeypatch):
    args = mocker.MagicMock()

    # Case 1: No temp_dir implies the no (`None`) overriding.
    args.temp_dir = False
    assert itikz.get_cwd(args) is None

    # Case 2: The environmental variable demands a temp_dir.
    monkeypatch.setenv('ITIKZ_TEMP_DIR', '1')
    assert itikz.get_cwd(args).endswith("itikz")
    monkeypatch.delenv('ITIKZ_TEMP_DIR')

    # Case 2: Use the system temp_dir.
    args.temp_dir = True
    assert itikz.get_cwd(args).endswith("itikz")


def test_parse_args_does_not_sysexit(capsys):
    itikz.parse_args("-h")
    _, err = capsys.readouterr()
    assert err == ""


def test_build_template_args(mocker):

    # Case 1: No overrides.
    args = itikz.parse_args("")[1]
    res = itikz.build_template_args("code", args)
    assert res == dict(src="code", scale=1, extras="")

    # Case 2: Override scale.
    args = itikz.parse_args("--scale 2")[1]
    res = itikz.build_template_args("code", args)
    assert res == dict(src="code", scale=2.0, extras="")

    # Case 3: Override tex packages.
    args = itikz.parse_args("--scale 2 --tex-packages a,b")[1]
    res = itikz.build_template_args("code", args)
    assert res == dict(src="code", scale=2.0, extras=r"\usepackage{a,b}")

    # Case 4: Override tikz libraries.
    invocation = "--scale 2 --tex-packages a,b --tikz-libraries c,d"
    args = itikz.parse_args(invocation)[1]
    res = itikz.build_template_args("code", args)
    assert res == dict(src="code", scale=2.0,
                       extras="\\usepackage{a,b}\n\\usetikzlibrary{c,d}")


RECTANGLE_TIKZ = r"""
\documentclass[tikz]{standalone}
\begin{document}
\begin{tikzpicture}
\draw[fill=blue] (0, 0) rectangle (1, 1);
\end{tikzpicture}
\end{document}
""".strip()


BAD_TIKZ = "HELLO WORLD"


def test_fetch_or_compile_svg_good_input(tmpdir, monkeypatch):
    expected_md5 = "15d53b05d3a27e1545c9a3688be5e3b4"
    res = itikz.fetch_or_compile_svg(RECTANGLE_TIKZ, 'test_', str(tmpdir))

    for ext in 'tex', 'svg':
        path = tmpdir.join("test_{}.{}".format(expected_md5, ext))
        assert os.path.exists(str(path))

    for ext in 'log', 'aux', 'pdf':
        path = tmpdir.join("test_{}.{}".format(expected_md5, ext))
        assert not os.path.exists(str(path))

    assert isinstance(res, SVG)


def test_fetch_or_compile_svg_bad_input(tmpdir, capsys):
    expected_md5 = "361fadf1c712e812d198c4cab5712a79"
    res = itikz.fetch_or_compile_svg(BAD_TIKZ, 'test_', str(tmpdir))

    for ext in 'tex', 'svg', 'log', 'aux', 'pdf':
        path = tmpdir.join("test_{}.{}".format(expected_md5, ext))
        assert not os.path.exists(str(path))

    assert res is None

    _, err = capsys.readouterr()
    assert 'error' in err.lower()


@pytest.fixture
def itikz_magic(mocker, monkeypatch):
    obj = itikz.TikZMagics()
    shell = mocker.MagicMock()
    shell.user_ns = {}
    monkeypatch.setattr(obj, 'shell', shell)
    return obj


def test_magic_print_help(itikz_magic, capsys):
    assert itikz_magic.itikz("--help") is None
    _, err = capsys.readouterr()
    assert err.startswith("usage: %%itikz")


def test_magic_print_help_on_no_input(itikz_magic, capsys):
    assert itikz_magic.itikz('') is None
    _, err = capsys.readouterr()
    assert err.startswith("usage: %%itikz")


def test_magic_cell_usage(itikz_magic):
    expected_md5 = "15d53b05d3a27e1545c9a3688be5e3b4"
    tmp_dir = os.path.join(tempfile.gettempdir(), 'itikz')

    res = itikz_magic.itikz("--temp-dir --file-prefix test_", RECTANGLE_TIKZ)

    for ext in 'tex', 'svg':
        path = os.path.join(tmp_dir, "test_{}.{}".format(expected_md5, ext))
        assert os.path.exists(str(path))

    for ext in 'log', 'aux', 'pdf':
        path = os.path.join(tmp_dir, "test_{}.{}".format(expected_md5, ext))
        assert not os.path.exists(str(path))

    assert isinstance(res, SVG)


def test_magic_line_usage(itikz_magic):
    expected_md5 = "15d53b05d3a27e1545c9a3688be5e3b4"
    tmp_dir = os.path.join(tempfile.gettempdir(), 'itikz')
    itikz_magic.shell.user_ns['env_src'] = RECTANGLE_TIKZ

    res = itikz_magic.itikz("--temp-dir --file-prefix test_ env_src")

    for ext in 'tex', 'svg':
        path = os.path.join(tmp_dir, "test_{}.{}".format(expected_md5, ext))
        assert os.path.exists(str(path))

    for ext in 'log', 'aux', 'pdf':
        path = os.path.join(tmp_dir, "test_{}.{}".format(expected_md5, ext))
        assert not os.path.exists(str(path))

    assert isinstance(res, SVG)


def test_magic_no_simultanous_standalone_and_pic(itikz_magic, capsys):
    res = itikz_magic.itikz("--implicit-pic --implicit-standalone",
                            RECTANGLE_TIKZ)

    assert res is None
    _, err = capsys.readouterr()
    assert err.startswith("Can't use --implicit")


def test_magic_jinja_without_jinja(itikz_magic, capsys, monkeypatch):
    monkeypatch.setattr(itikz, 'JINJA2_ENABLED', False)
    res = itikz_magic.itikz("--as-jinja", RECTANGLE_TIKZ)

    assert res is None
    _, err = capsys.readouterr()
    assert err.startswith("Please install jinja2")


def test_magic_jinja_print_src(itikz_magic, capsys):
    src = "The answer is: {{n}}"
    itikz_magic.shell.user_ns['n'] = 42
    res = itikz_magic.itikz("--as-jinja --print-jinja", src)

    assert res is None
    out, _ = capsys.readouterr()
    assert out.startswith("The answer is: 42")


def test_implicit_pic(itikz_magic):
    src = r"\node[draw] at (0,0) {Hello World};"
    res = itikz_magic.itikz("--implicit-pic --temp-dir", src)
    assert isinstance(res, SVG)


def test_implicit_standalone(itikz_magic):
    pic = r"\node[draw] at (0,0) {Hello World};"
    src = "\\begin{tikzpicture}\n" + pic + "\n\\end{tikzpicture}\n"
    cmd = "--implicit-standalone --tex-packages=tikz --temp-dir"
    res = itikz_magic.itikz(cmd, src)
    assert isinstance(res, SVG)
