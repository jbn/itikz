import re
import jinja2

la_colors=r"""% ======================================================= colors
\definecolor{la_white}{RGB}{233,235,223} %#E9EBDF
\definecolor{la_dark}{RGB}{59,54,81}     %#3B3651
\definecolor{la_gray}{RGB}{96,112,139}   %#60708B
\definecolor{la_tan}{RGB}{152,159,122}   %#989F7A
"""

template = r"""\documentclass[tikz{% for a in class_args %},{{a}}{% endfor %}]{standalone}
\pagestyle{empty}
{% for p in tex_packages %}
{{p}}
{% endfor %}
{% for p in tikz_libraries %}
\usetikzlibrary{{p}}
{% endfor %}
{{extension}}

\begin{document}
{{preamble}}
\begin{tikzpicture}{% for p in tikz_args %}{{p}}{% endfor %}
   {{tikz_code}}
\end{tikzpicture}
\end{document}
"""

pattern = re.compile( r'(\[[^]]*])(.*)' )

def tikz_source( code,
                 class_args=None, tex_packages=None, tikz_libraries=None, extension="% no_extension", 
                 preamble="% preamble", tikz_args=None):
    def split(arg):
        if arg is None:
            return []
        l = []
        for a in arg.split(","):
            match = pattern.match( a )
            if match:
                l.append( r"\usepackage" + match.group(1) + "{" + match.group(2)+ "}" )
            else:
                l.append(r"\usepackage{" + a + "}")
        return l


    class_args     = [] if class_args     is None else [class_args]
    tex_packages   = split(tex_packages)
    tikz_libraries = [] if tikz_libraries is None else ["{"+tikz_libraries+"}"]
    tikz_args      = [] if tikz_args      is None else ["["+tikz_args+"]"]

    src=jinja2.Template( template )\
              .render( class_args=class_args,
                       tex_packages=tex_packages,
                       tikz_libraries=tikz_libraries,
                       extension=extension,
                       preamble=preamble,
                       tikz_args=tikz_args,
                       tikz_code=code
                     )

    return src
