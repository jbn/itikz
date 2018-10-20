import os
import pytest
import itikz
from IPython.display import SVG


@pytest.mark.skip
def test_cell_magic():
    pass


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

def test_fetch_or_compile_svg(tmpdir, monkeypatch):
    expected_md5 = "15d53b05d3a27e1545c9a3688be5e3b4"
    res = itikz.fetch_or_compile_svg(RECTANGLE_TIKZ, 'test_', str(tmpdir))

    for ext in 'tex', 'svg':
        path = tmpdir.join("test_{}.{}".format(expected_md5, ext))
        assert os.path.exists(str(path))

    for ext in 'log', 'aux', 'pdf':
        path = tmpdir.join("test_{}.{}".format(expected_md5, ext))
        assert not os.path.exists(str(path))

    assert isinstance(res, SVG)
