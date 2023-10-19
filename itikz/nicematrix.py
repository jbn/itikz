import numpy as np
import sympy as sym
import jinja2
import itikz
from IPython.core.display import SVG
from IPython.core.display import HTML

# ================================================================================================================================
extension = r''' '''
# -----------------------------------------------------------------
preamble = r''' '''
# =================================================================
BACKSUBST_TEMPLATE = r'''\documentclass[notitlepage,table,svgnames]{article}
\pagestyle{empty}
\usepackage[margin=0cm,paperwidth=90in]{geometry}
\usepackage{mathtools}
\usepackage{amssymb}
\usepackage{cascade}
\usepackage{systeme}
\usepackage{nicematrix}

\begin{document}\begin{minipage}{\textwidth}
{%% if fig_scale %%}
\scalebox{ {{fig_scale}} }{%
{%% endif %%}
% ----------------------------------------------------------------------------------------
{{preamble}}%
%====================================================================
{%% if show_system %%}
{{system_txt}}
{%% endif %%}
%====================================================================
{%% if show_cascade %%}
{%% if show_system %%}
\vspace{1cm}

{%% endif %%}
{%% for line in cascade_txt -%%}
{{line}}
{%% endfor -%%}
{%% endif %%}
%====================================================================
{%% if show_solution %%}
{%% if show_system or show_cascade %%}
\vspace{1cm}

{%% endif %%}
{{solution_txt}}
{%% endif %%}
%====================================================================
{%% if fig_scale %%}
}
{%% endif %%}
\end{minipage}\end{document}
'''
# =================================================================
EIGPROBLEM_TEMPLATE = r'''\documentclass[notitlepage,table,svgnames]{article}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{mathtools}
\usepackage{nicematrix}
\usepackage{xcolor}

\begin{document}\begin{minipage}{\textwidth}
{%% if fig_scale %%}
{{fig_scale}}
{%% endif %%}
% ----------------------------------------------------------------------------------------
{{preamble}}%
%=========================================================================================
\begin{tabular}{{table_format}} \toprule
{%% if sigmas %%}% sigma -----------------------------------------------------------------
$\color{{color}}{\sigma}$ & {{sigmas}}  {{rule_format}}
{%% endif %%}% lambda --------------------------------------------------------------------
$\color{{color}}{\lambda}$ & {{lambdas}} {{rule_format}}
$\color{{color}}{m_a}$ & {{algebraic_multiplicities}}  {{rule_format}} \addlinespace[1mm]
%  eigenvectors --------------------------------------------------------------------------
{\parbox{2cm}{\textcolor{{color}}{basis for $\color{{color}}{E_\lambda}$}}} &
{{eigenbasis}} {%% if orthonormal_basis %%}
%  orthonormal eigenvectors --------------------------------------------------------------
 {{rule_format}} \addlinespace[2mm]
{\parbox{2cm}{\textcolor{{color}}{orthonormal basis for $E_\lambda$}}} &
{{orthonormal_basis}}
{%% endif -%%}
 \addlinespace[2mm] \midrule \addlinespace[2mm]
% ------------------------------------------------------------- lambda
$\color{{color}}{ {{matrix_names[0]}} =}$ & {{lambda_matrix}} \\ \addlinespace[2mm]
% ------------------------------------------------------------- E or Q
{%% if evecs_matrix %%}%
$\color{{color}}{ {{matrix_names[1]}} = }$ & {{evecs_matrix}} \\  \addlinespace[2mm] %\bottomrule
{%% endif -%%}
% ------------------------------------------------------------- U
{%% if left_singular_matrix %%}%
$\color{{color}}{ {{matrix_names[2]}} = }$ & {{left_singular_matrix}} \\  \addlinespace[2mm] \bottomrule
{%% endif -%%}
\end{tabular}
{%% if fig_scale %%}
}
{%% endif %%}
\end{minipage}\end{document}
'''
# =================================================================
GE_TEMPLATE = r'''\documentclass[notitlepage]{article}
\pagestyle{empty}
\usepackage[margin=0cm,paperwidth=90in]{geometry}

\usepackage{mathtools}
\usepackage{xltxtra}
\usepackage{pdflscape}
\usepackage{graphicx}
\usepackage[table,svgnames]{xcolor}
\usepackage{nicematrix,tikz}
\usetikzlibrary{calc,fit,decorations.markings}

\newcommand*{\colorboxed}{}
\def\colorboxed#1#{%
  \colorboxedAux{#1}%
}
\newcommand*{\colorboxedAux}[3]{%
  % #1: optional argument for color model
  % #2: color specification
  % #3: formula
  \begingroup
    \colorlet{cb@saved}{.}%
    \color#1{#2}%
    \boxed{%
      \color{cb@saved}%
      #3%
    }%
  \endgroup
}

% ---------------------------------------------------------------------------- extension
{{extension}}
\begin{document}\begin{minipage}{\textwidth}
\begin{landscape}
{%% if fig_scale %%}
{{fig_scale}}
{%% endif %%}
% ---------------------------------------------------------------------------- preamble
{{preamble}}%
% ============================================================================ NiceArray
$\begin{NiceArray}[vlines-in-sub-matrix = I]{{mat_format}}{{mat_options}}%
{%% if codebefore != [] -%%}
\CodeBefore [create-cell-nodes]
    {%% for entry in codebefore: -%%}
    {{entry}}
    {%% endfor -%%}%
\Body
{%% endif -%%}
{{mat_rep}}
\CodeAfter %[ sub-matrix / extra-height=2mm, sub-matrix / xshift=2mm ]
% --------------------------------------------------------------------------- submatrix delimiters
    {%% for loc in submatrix_locs: -%%}
          \SubMatrix({{loc[1]}})[{{loc[0]}}]
    {%% endfor -%%}
    {%% for txt in submatrix_names: -%%}
          {{txt}}
    {%% endfor -%%}
% --------------------------------------------------------------------------- pivot outlines
\begin{tikzpicture}
    \begin{scope}[every node/.style = draw]
    {%% for loc in pivot_locs: -%%}
        \node [draw,{{loc[1]}},fit = {{loc[0]}}]  {} ;
    {%% endfor -%%}
    \end{scope}
%
% --------------------------------------------------------------------------- explanatory text
    {%% for loc,txt,c in txt_with_locs: -%%}
        \node [right,align=left,color={{c}}] at {{loc}}  {\qquad {{txt}} } ;
    {%% endfor -%%}
%
% --------------------------------------------------------------------------- row echelon form path
    {%% for t in rowechelon_paths %%} {{t}}
    {%% endfor -%%}
\end{tikzpicture}
\end{NiceArray}$
{%% if fig_scale %%}
}
{%% endif %%}
\end{landscape}
\end{minipage}\end{document}
'''
# ================================================================================================================================
# Index Computations and formating associated with Matrices laid out on a grid.
# ================================================================================================================================
class MatrixGridLayout:
    ''' Basic Computations of the Matrix Grid Layout and the resulting Tex Representation
        Indexing is zero-based.

        The matrices are first written into an array, with an associated format string:
            array_format_string_list()
            array_of_tex_entries()
            decorate_tex_entries()
        Each array row is converted to a string
            tex_repr
        The display is further enhanced by adding
            nm_add_option
            nm_submatrix_locs
    '''

    def __init__(self, matrices, extra_cols=None, extra_rows = None ):
        '''save the matrices, determine the matrix grid dimensions and the number of rows in the first row of the grid
           Note that the number of cols in the first grid row and/or the number of rows in the first grid col can be ragged
        '''
        # Fix up matrices
        if not isinstance( matrices, list):  # allow using this class for a single matrix [[None, A]]
            matrices = [[ None, matrices ]]

        self.matrices = []
        cnt_layers    = 0
        for (i,layer) in enumerate(matrices): # ensure the matrices are not passed as lists
            cnt_layers += 1
            l = []
            for (j,mat) in enumerate(layer):
                if isinstance( mat, list ):
                    l.append( np.array(mat) )
                else:
                    l.append( mat )
            self.matrices.append(l)

        self.nGridRows        = cnt_layers # len(self.matrices)
        self.nGridCols        = len(self.matrices[0])

        self._set_shapes()
        self.array_names      = []

        self.mat_row_height = [ max(map( lambda s: s[0], self.array_shape[i, :])) for i in range(self.nGridRows)]
        self.mat_col_width  = [ max(map( lambda s: s[1], self.array_shape[:, j])) for j in range(self.nGridCols)]

        self.adjust_positions( extra_cols, extra_rows )
        self.txt_with_locs    = []
        self.rowechelon_paths = []
        self.codebefore       = []
        self.preamble         = '\n' + r" \NiceMatrixOptions{cell-space-limits = 1pt}"+'\n'
        self.extension        = '%\n'

    def adjust_positions( self, extra_cols=None, extra_rows=None ):
        '''insert extra rows and cols between matrices'''
        self.extra_cols          = MatrixGridLayout._set_extra( extra_cols, self.nGridCols )
        self.extra_rows          = MatrixGridLayout._set_extra( extra_rows, self.nGridRows )

        self.cs_extra_cols       = np.cumsum( self.extra_cols )
        self.cs_extra_rows       = np.cumsum( self.extra_rows )

        self.cs_mat_row_height   = np.cumsum( np.hstack([[0], self.mat_row_height]))
        self.cs_mat_col_width    = np.cumsum( np.hstack([[0], self.mat_col_width ]))

        self.tex_shape = (self.cs_mat_row_height[-1]+self.cs_extra_rows[-1],
                          self.cs_mat_col_width [-1]+self.cs_extra_cols[-1])

    def _set_shapes(self):
        '''compute the shapes of the arrays in the grid, obtain the maximal number of rows/cols'''
        self.array_shape = np.empty((self.nGridRows, self.nGridCols), tuple)

        for i in range(self.nGridRows):
            for j in range(self.nGridCols):
                try:
                    self.array_shape[i,j] = (self.matrices[i][j]).shape
                except:
                    self.array_shape[i,j] = (0,0)

        if self.nGridRows > 1:
            self.n_Col_0 = [s[1] for s in self.array_shape[1:,0]]
        else:
            self.n_Col_0 = [self.array_shape[0,1][1]]

        if self.nGridCols > 1:
            self.m_Row_0 = [s[0] for s in self.array_shape[0,1:]]
        else:
            self.m_Row_0 = [s[1] for s in self.array_shape[0,1:]]

    @staticmethod
    def _set_extra( extra, n ):
        if isinstance(extra, int):
            extra = np.hstack([ np.repeat( 0, n), [extra] ])
        elif extra is None:
            extra = np.repeat( 0, n+1 )
        else:
            assert( len(extra) == (n+1))
        return extra

    def describe(self):
        #self.grid_shape = [len(self.mat_row_height), len(self.mat_col_width)]
        #                = [self.nGridRows, self.nCOLMats]

        print( f"Layout {self.nGridRows} x {self.nGridCols} grid:")

        print( f".  insert extra_cols:            {self.extra_cols}")
        print( f".  col_start                   = {self.cs_mat_col_width  + self.cs_extra_cols}")
        print( f".  row_start                   = {self.cs_mat_row_height + self.cs_extra_rows}")
        print()

        print( "Consistent Matrix Sizes in the grid")
        for i in self.mat_row_height:
            for j in self.mat_col_width:
                print( f'  {(i,j)}', end='')
            print()

        print("Actual TopLeft:BottomRight Indices")
        for i in range(self.nGridRows):
            for j in range(self.nGridCols):
                tl,br,_ = self._top_left_bottom_right(i,j)
                print( f'  {tl}:{br}', end='')
            print()

    def element_indices( self, i,j, gM, gN ):
        '''return the actual indices of element (i,j) in the matrix at grid position (gM,gN)'''
        last_row  = self.cs_mat_row_height[gM+1] + self.cs_extra_rows[gM]
        last_col  = self.cs_mat_col_width [gN+1] + self.cs_extra_cols[gN]

        A_shape   = self.array_shape[gM][gN]
        return (last_row - (A_shape[0] -i),
                last_col - (A_shape[1] - j)
               )

    def _top_left_bottom_right( self, gM, gN ):
        ''' given the grid position obtain the actual indices of the top left corner for the matrix'''
        A_shape = self.array_shape[gM,gN]

        row_offset = self.cs_extra_rows[gM]+self.cs_mat_row_height[gM]+self.mat_row_height[gM]-A_shape[0]
        col_offset = self.cs_extra_cols[gN]+self.cs_mat_col_width [gN]+self.mat_col_width [gN]-A_shape[1]
        return (row_offset, col_offset), \
               (row_offset+A_shape[0]-1, col_offset+A_shape[1]-1), \
               A_shape

    #def tex_repr( self, blockseps = r' \noalign{\vskip2mm} '):
    def tex_repr( self, blockseps = r'[2mm]'):
        '''Create a list of strings from the array of TeX strings, one for each line in the grid ready to print in the LaTeX document'''
        self.tex_list =[' & '.join( self.a_tex[k,:]) for k in range(self.a_tex.shape[0])]
        for i in range( len(self.tex_list) -1):
            self.tex_list[i] += r' \\'

        for i in (self.cs_mat_row_height[1:-1] + self.cs_extra_rows[1:-1] - self.extra_rows[1:-1]):
            self.tex_list[i-1] += blockseps

        if self.extra_rows[-1] != 0: # if there are final extra rows, we need another sep
            self.tex_list[ self.tex_shape[0] - self.extra_rows[-1] - 1] += blockseps

    def array_of_tex_entries(self, formater=str):
        '''Create a matrix of TeX strings from the grid entries'''

        a_tex = np.full( self.tex_shape,"", dtype=object)

        for i in range(self.nGridRows):
            for j in range(self.nGridCols):
                tl,br,shape = self._top_left_bottom_right(i,j)
                A = self.matrices[i][j]
                for ia in range( shape[0]):
                    for ja in range(shape[1]):
                        a_tex[ tl[0]+ia, tl[1]+ja ] = formater( A[ia,ja])
        self.a_tex = a_tex

    def decorate_tex_entries( self, gM, gN, decorate, entries=None):
        '''apply decorate to the list of i,j TeX entries to grid matrix at (gM,gN)'''
        try: # avoid writing code for A == None case
            tl,br,shape = self._top_left_bottom_right(gM,gN)

            if entries is None:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        self.a_tex[tl[0]+i, tl[1]+j] = decorate(self.a_tex[tl[0]+i, tl[1]+j])
            else:
                for (i,j) in entries:
                    self.a_tex[tl[0]+i, tl[1]+j]     = decorate(self.a_tex[tl[0]+i, tl[1]+j])
        except:
            pass

    @staticmethod
    def matrix_array_format( N, p_str='I', vpartitions=None):
        '''format string for a matrix with N columns'''
        if vpartitions is None:
            return N*"r"
            #return f"*{N}r"
        s     = ""
        cur   = 0
        for p in vpartitions:
            s_r = (p-cur)*"r"
            s += f"{s_r}{p_str}"
            #s += f"*{p-cur}r{p_str}"
            cur = p
        if cur < N:
            s += (N-cur)*"r"
            #s += f"*{N-cur}r"
        return s

    #def array_format_string_list( self, partitions={}, spacer_string=r'@{\qquad\ }', p_str='I', last_col_format = "l@{\qquad\;\;}") :
    def array_format_string_list( self, partitions={}, spacer_string=r'@{\hspace{9mm}}',
                                  p_str='I', last_col_format=r'l@{\hspace{2cm}}' ):
        '''Construct the format string. Partitions is a dict { gridcolumn: list of partitions}'''

        for i in range(self.nGridCols):   # make sure we have a partion entry for each column of matrices
            if i not in partitions:
                partitions[i]=None

        # format for initial extra cols
        l   = self.extra_cols[0]
        fmt = l*'r'

        last = self.nGridCols - 1
        for i in range(self.nGridCols):
            # we now iterate over (matrix, extra) pairs
            # -----------  matrix -----------------------
            N    = self.mat_col_width[i]
            fmt += spacer_string + MatrixGridLayout.matrix_array_format( N, p_str=p_str, vpartitions=partitions[i] )

            # ------------ spacer ----------------------
            l = self.extra_cols[i+1]
            if l > 0:
                if i == last:
                    if l > 1:
                        fmt += spacer_string + (l-1)*'r'+last_col_format
                    else:
                        fmt += spacer_string + last_col_format
                else:
                    fmt += spacer_string + l*'r'

        self.format = fmt

    def tl_shape_below( self, gM, gN ):
        '''obtain tl and shape of free space below grid matrix at (gM, gN)'''
        tl,br,_ = self._top_left_bottom_right( gM, gN )
        free_tl = (br[0]+1, tl[1])
        shape   = (self.tex_shape[0]-free_tl[0], self.tex_shape[1] - tl[1])
        return free_tl, shape

    def tl_shape_above( self, gM, gN ):
        '''obtain tl and shape of free space above grid matrix at (gM, gN)'''
        tl,br,_ = self._top_left_bottom_right( gM, gN )
        free_tl = (tl[0]-self.extra_rows[gM], tl[1])
        shape   = (self.extra_rows[gM], self.tex_shape[1]-free_tl[1])
        return free_tl, shape

    def tl_shape_left( self, gM, gN ):
        '''obtain tl and shape of free space above grid matrix at (gM, gN)'''
        tl,_,_ = self._top_left_bottom_right( gM, gN )
        return (tl[0],tl[1]-self.extra_cols[gN]), \
               (self.tex_shape[0]-tl[0],self.extra_cols[gN])

    def tl_shape_right( self, gM, gN ):
        '''obtain tl and shape of free space above grid matrix at (gM, gN)'''
        tl,_,shape = self._top_left_bottom_right( gM, gN )
        return (tl[0],tl[1]+shape[1]), \
               (self.tex_shape[0]-tl[0],self.extra_cols[gN+1])

    def add_row_above( self, gM, gN, m, formater=str, offset=0 ):
        '''add tex entries to the tex array'''
        tl,shape = self.tl_shape_above( gM, gN )
        for (j,v) in enumerate(m):
            self.a_tex[tl[0]+offset, tl[1]+j] = formater( v )

    def add_row_below( self, gM, gN, m, formater=str, offset=0 ):
        '''add tex entries to the tex array'''
        tl,shape = self.tl_shape_below( gM, gN )
        for (j,v) in enumerate(m):
            self.a_tex[tl[0]+offset, tl[1]+j] = formater( v )

    def add_col_right( self, gM, gN, m, formater=str, offset=0 ):
        '''add tex entries to the tex array'''
        tl,shape = self.tl_shape_right( gM, gN )
        for (i,v) in enumerate(m):
            self.a_tex[tl[0]+i, tl[1]+offset] = formater( v )

    def add_col_left( self, gM, gN, m, formater=str, offset=0 ):
        '''add tex entries to the tex array'''
        tl,shape = self.tl_shape_left( gM, gN )
        for (i,v) in enumerate(m):
            self.a_tex[tl[0]+i, tl[1]+offset-1] = formater( v )

    def nm_submatrix_locs(self, name='A', color='blue', name_specs=None, line_specs=None ):
        '''nicematrix style location descriptors of the submatrices'''
        # name_specs = [ spec*]; spec = [ (gM, gN), position, text ]
        # line_specs = [ spec*]; spec = [ (gM, gN), h_lines, vlines ]
        self.submatrix_name = name
        smat_args = np.full( (self.nGridRows,self.nGridCols),"", dtype=object)
        if line_specs is not None:
            for pos,h_lines,v_lines in line_specs:
                if h_lines is not None:
                    if isinstance( h_lines, int):
                        smat_args[pos] = f',hlines={h_lines}'
                    else:
                        smat_args[pos] = ',hlines={'+ ','.join([str(s) for s in h_lines]) + '}'
                if v_lines is not None:
                    if isinstance( v_lines, int):
                        smat_args[pos] += f',vlines={v_lines}'
                    else:
                        smat_args[pos] += ',vlines={'+ ','.join([str(s) for s in v_lines]) + '}'

        locs = []
        for i in range(self.nGridRows):
            for j in range(self.nGridCols):
                if self.array_shape[i,j][0] != 0:
                    tl,br,_ = self._top_left_bottom_right(i,j)
                    smat_arg = f"name={name}{i}x{j}"+smat_args[i,j]
                    locs.append( [ smat_arg, f"{{{tl[0]+1}-{tl[1]+1}}}{{{br[0]+1}-{br[1]+1}}}"] )

        self.locs = locs

        if name_specs is not None:
            array_names = []
            ar=r"\tikz \draw[<-,>=stealth,COLOR,thick] ($ (NAME.north east) + (0.02,0.02) $) -- +(0.6cm,0.3cm)    node[COLOR, above right=-3pt]{TXT};".replace('COLOR',color)
            al=r"\tikz \draw[<-,>=stealth,COLOR,thick] ($ (NAME.north west) + (-0.02,0.02) $) -- +(-0.6cm,0.3cm) node[COLOR, above left=-3pt] {TXT};".replace('COLOR',color)
            a =r"\tikz \draw[<-,>=stealth,COLOR,thick] ($ (NAME.north) + (0,0) $) -- +(0cm,0.6cm) node[COLOR, above=1pt] {TXT};".replace('COLOR',color)

            bl=r"\tikz \draw[<-,>=stealth,COLOR,thick] ($ (NAME.south west) + (-0.02,-0.02) $) -- +(-0.6cm,-0.3cm)  node[COLOR, below left=-3pt]{TXT};".replace('COLOR',color)
            br=r"\tikz \draw[<-,>=stealth,COLOR,thick] ($ (NAME.south east) + (0.02,-0.02) $) -- +(0.6cm,-0.3cm)   node[COLOR, below right=-3pt]{TXT};".replace('COLOR',color)
            b =r"\tikz \draw[<-,>=stealth,COLOR,thick] ($ (NAME.south) + (0,0) $) -- +(0cm,-0.6cm) node[COLOR, below=1pt] {TXT};".replace('COLOR',color)

            for (gM,gN),pos,txt in name_specs:
                nm = f"{name}{gM}x{gN}"
                t  = None
                if    pos == 'a':  t =  a.replace('NAME',nm).replace('TXT',txt)
                elif  pos == 'al': t = al.replace('NAME',nm).replace('TXT',txt)
                elif  pos == 'ar': t = ar.replace('NAME',nm).replace('TXT',txt)
                elif  pos == 'b':  t =  b.replace('NAME',nm).replace('TXT',txt)
                elif  pos == 'bl': t = bl.replace('NAME',nm).replace('TXT',txt)
                elif  pos == 'br': t = br.replace('NAME',nm).replace('TXT',txt)
                if t is not None:
                    array_names.append( t )

            self.array_names = array_names

    def nm_background(self, gM,gN, loc_list, color='red!15', pt=0):
        '''add background color to a list of entries'''
        tl,_,_ = self._top_left_bottom_right( gM, gN )
        for entry in loc_list:
            cmd_1 = f'\\tikz \\node [fill={color}, inner sep = {pt}pt, fit = '
            if not isinstance( entry, list):
                cmd_2  =  f'({tl[0]+entry[0]+1}-{tl[1]+entry[1]+1}-medium)'
            else:
                cmd_2  =  f'({tl[0]+entry[0][0]+1}-{tl[1]+entry[0][1]+1}-medium) ({tl[0]+entry[1][0]+1}-{tl[1]+entry[1][1]+1}-medium)'
            cmd_3 = ' ] {} ;'
            self.codebefore.append( cmd_1 + cmd_2 + cmd_3 )


    def nm_text(self, txt_list, color='violet'):
        '''add text add each layer (requires a right-most extra col)'''
        assert( self.extra_cols[-1] != 0 )

        # find the indices of the first row in the last col (+1 for nicematrix indexing)
        txt_with_locs = []
        for (g,txt) in enumerate(txt_list):
            A_shape   = self.array_shape[g][self.nGridCols-1]

            first_row = self.cs_mat_row_height[g] + self.cs_extra_rows[g] + (self.mat_row_height[g] - A_shape[0])+1
            txt_with_locs.append(( f'({first_row}-{self.tex_shape[1]-1}.east)', txt, color) )
        self.txt_with_locs = txt_with_locs

    def nm_add_rowechelon_path( self, gM,gN, pivots, case='hh', color='violet,line width=0.4mm', adj=0.1 ):
        tl,_,shape = self._top_left_bottom_right( gM, gN )

        def coords(i,j):
            if i >= shape[0]:
                if gN == 0 and j == 0:                           # HACK alert: first col in leftmost matrix!
                    x = r'\x1'
                else:
                    x = r'\x2' if j >= shape[1] else r'\x4'
                #x = r'\x2' if j >= shape[1] else r'\x1'
                y = r'\y2'
                p = f'({x},{y})'
            elif j >= shape[1]:
                x = r'\x2' if i >= shape[0] else r'\x2'
                y = r'\y4'
                p = f'({x},{y})'
            elif j == 0:
                x = r'\x1'
                y = r'\y1' if i == 0 else r'\y3'
                p = f'({x},{y})'
            else:
                x = f'{i+1+tl[0]}'
                y = f'{j+1+tl[1]}'
                p = f'({x}-|{y})'

            if j != 0 and j < shape[1] and adj != 0:
                p = f'($ {p} + ({adj:2},0) $)'

            return p

        cur = pivots[0]
        ll = [cur] if (case == 'vv') or (case == 'vh') else []
        for nxt in pivots[1:]:                  # at top right
            if cur[0] != nxt[0]:                # down 1
                cur = (cur[0]+1, cur[1])
                ll.append( cur )
            if nxt[1] != cur[1]:                # over
                cur = (cur[0], nxt[1])
                ll.append( cur )
            if cur != nxt:
                ll.append(nxt)                  # down  to top right
            cur = nxt

        if len(ll) == 0 and case == 'hv':
            ll = [ (pivots[0][0]+1,pivots[0][0] ), (shape[0], pivots[0][1] )]

        if (case == 'hh') or (case == 'vh'):
            if cur[0] != shape[0]:             # down 1
                cur = (cur[0]+1, cur[1])
                ll.append( cur )
            ll.append( (cur[0], shape[1]))     # over to right
        else:
            ll.append( (shape[0], cur[1]))     # down to bottom

        corners = f'let \\p1 = ({self.submatrix_name}{gM}x{gN}.north west), \\p2 = ({self.submatrix_name}{gM}x{gN}.south east), '

        if (case == 'vv') or (case == 'vh'):
            p3 = f'\\p3 = ({ll[1][0]+tl[0]+1}-|{ll[1][1]+tl[1]+1}), '
        else:
            p3 = f'\\p3 = ({ll[0][0]+tl[0]+1}-|{ll[0][1]+tl[1]+1}), '

        if (case=='vh') or (case=='hh'):                #   last dir: ->
            i,j = ll[-2]
            p4 = f'\\p4 = ({i+tl[0]+1}-|{j+tl[1]+1}) in '
        else:                                           #   last dir: |
            i,j = ll[-1]
            #if len(pivots) == 1 and cur[0] == 0 and gN == 0:    # was this my previous "FIX"???
            #    p4 = f'\\p4 = ({self.submatrix_name}{gM}x{gN}.south west) in '
            #else:
            #    p4 = f'\\p4 = ({i+tl[0]+1}-|{j+tl[1]+1}) in '
            p4 = f'\\p4 = ({i+tl[0]+1}-|{j+tl[1]+1}) in '

        cmd = '\\tikz \\draw['+color+'] ' + corners + p3 + p4  + ' -- '.join( [coords(*p) for p in ll] ) + ';'
        self.rowechelon_paths.append( cmd )

    def apply( self, func,  *args, **kwargs ):
        func( self, *args, **kwargs )

    def nm_latexdoc( self, template = GE_TEMPLATE, fig_scale=None ):
        if fig_scale is not None:
            fig_scale = r'\scalebox{'+str(fig_scale)+'}{%'
        return jinja2.Template( template, block_start_string='{%%', block_end_string='%%}',
                                          comment_start_string='{##', comment_end_string='##}' ).render( \
                preamble        = self.preamble,
                fig_scale       = fig_scale,
                extension       = self.extension,
                mat_rep         = '\n'.join( self.tex_list ),
                mat_format      = '{'+self.format+'}',
                mat_options     = '[create-extra-nodes,create-medium-nodes]',
                submatrix_locs  = self.locs,
                submatrix_names = self.array_names,
                pivot_locs      = [],
                txt_with_locs   = self.txt_with_locs,
                rowechelon_paths= self.rowechelon_paths,
                codebefore      = self.codebefore,
        )
# -----------------------------------------------------------------------------------------------------
def make_decorator( text_color='black', bg_color=None, text_bg=None, boxed=None, box_color=None, bf=None, move_right=False, delim=None ):
    '''decorate terms:
        text_color :  apply text color,                   default = 'black'
        text_bg :     apply background color,             default = None
        boxed   :     put a box around the entry,         default = False (None)
        box_color:    put a colored box around the entry, default = False (None)
        bf :          make the entry boldface,            default = False (None)
        move_right :  apply \\mathrlap,                   default = False (None)
        delim :       put delimiter around text,          default = None,    e.g.,  surround with in '$'
    '''
    box_decorator         = "\\boxed<{a}>"
    coloredbox_decorator  = "\\colorboxed<{color}><{a}>"
    color_decorator       = "\\Block[draw={text_color},fill={bg_color}]<><{a}>"
    txt_color_decorator   = "\\color<{color}><{a}>"
    bg_color_decorator    = "\\colorbox<{color}><{a}>"
    bf_decorator          = "\\mathbf<{a}>"
    rlap_decorator        = "\\mathrlap<{a}>"
    delim_decorator       = "<{delim}{a}{delim}>"

    x = '{a}'
    if bf is not None:
        x = bf_decorator.format(a=x)
    if boxed is not None:
        x = box_decorator.format( a=x )
    if box_color is not None:
        x = coloredbox_decorator.format( a=x, color=box_color )
    if bg_color is not None:
        x = bg_color_decorator.format(a=x, color=bg_color)
    if text_bg is not None:
        x = color_decorator.format(a=x, text_color= text_color, bg_color=text_bg)
    elif text_color != 'black':
        x = txt_color_decorator.format( color=text_color, a=x)

    if move_right:
        x = rlap_decorator.format(a=x)
    if delim is not None:
        x = delim_decorator.format( delim=delim, a=x )

    x = x.replace('<','{{').replace('>','}}')

    return lambda a: x.format(a=a)

# ================================================================================================================================
def str_rep_from_mat( A, formater=str):
    '''str_rep_from_mat( A, formater=str)
    convert matrix A to a string using formater
    '''
    M,N=A.shape
    return np.array( [[formater(A[i,j]) for j in range(N)] for i in range(M)] )

def str_rep_from_mats( A, b, formater=str ):
    '''str_rep_from_mats( A, b, formater=str)
    convert matrix A and vector b to a string using formater, return the augmented matrix
    '''
    sA = str_rep_from_mat(A, formater)
    sb = np.array(b).reshape(-1,1)
    return np.hstack( [sA, sb] )
# ================================================================================================================================
def mk_ge_names(n, lhs='E', rhs=['A','b'], start_index=1 ):
    '''utility to generate array names for ge'''
    names = np.full( shape=(n,2),fill_value='', dtype=object)

    def pe(i):
        if start_index is None:
            return ' '.join([f' {lhs}' for k in range(i,0,-1)])
        else:
            return ' '.join([f' {lhs}_{k+start_index-1}' for k in range(i,0,-1)])
    def pa(e_prod,i):
        if i > 0 and rhs[-1] == 'I':
            rhs[-1] = ''
        return r' \mid '.join( [e_prod+' '+k for k in rhs ])

    for i in range(n):
        if start_index is None:
            names[i,0] = f'{lhs}'
        else:
            names[i,0] = f'{lhs}_{start_index+i-1}'

        e_prod = pe(i)
        names[i,1] = pa(e_prod,i)

    if len(rhs) > 1:
        for i in range(n):
            names[i,1] = r'\left( '+ names[i,1] + r' \right)'

    for i in range(n):
        for j in range(2):
            names[i,j] = r'\mathbf{ ' + names[i,j] + ' }'

    terms = [ [(0,1),'ar', '$' + names[0,1] + '$']]
    for i in range(1,n):
        terms.append( [(i,0), 'al', '$' + names[i,0] + '$'])
        terms.append( [(i,1), 'ar', '$' + names[i,1] + '$'])
    return terms
# --------------------------------------------------------------------------------------------------------------------------------
def _ge( matrices, Nrhs=0, formater=str, pivot_list=None, bg_for_entries=None,
         variable_colors=['red','blue'], pivot_text_color='red',
         ref_path_list=None, comment_list=None, variable_summary=None, array_names=None,
         start_index=1, func=None, fig_scale=None, tmp_dir="tmp", keep_file=None ):
    '''basic GE layout (development version):
    matrices:         [ [None, A0], [E1, A1], [E2, A2], ... ]
    Nrhs:             number of right hand side columns determines the placement of a partition line, if any
                      can also be a list of widths to be partioned...
    pivot_list:       [ pivot_spec, pivot_spec, ... ] where pivot_spec = [grid_pos, [pivot_pos, pivot_pos, ...]]
    bg_for_entries:   [ bg_spec, ...] where bg_spec = [gM,gN, [ entries ], color, pt ]
    variable_colors:  [ basic_var_color, free_var_color]
    ref_path_list:    [ path_spec, path_spec, ... ] where path_spec = [grid_pos, [pivot_pos], directions ] where directions='vv','vh','hv' or 'hh'
    comment_list:     [ txt, txt, ... ] must have a txt entry for each layer. Multiline comments are separated by \\
    variable_summary: [ basic, ... ]  a list of true/false values specifying whether a column has a pivot or not
    array_names:      list of names for the two columns: [ 'E', ['A','b','I']
    start_index:      first subscript for the elementary operation matrices (can be None)
    func:             a function to be applied to the MatrixGridLayout object prior to generating the latex document
    '''
    extra_cols = None if comment_list     is None else 1
    extra_rows = None if variable_summary is None else 2

    m = MatrixGridLayout(matrices, extra_rows=extra_rows, extra_cols = extra_cols )

    # compute the format spec for the arrays and set up the entries (defaults to a single partition line)
    if isinstance( Nrhs, np.ndarray ):  # julia passes arrays rather than lists   :-()
        Nrhs = list(Nrhs.flatten())

    if not isinstance( Nrhs, list):
        partitions = {} if Nrhs == 0 else { 1: [m.mat_col_width[-1]-Nrhs]}
    else:
        nrhs = Nrhs.copy()                        # partitions just specifies each col
        cuts = [m.mat_col_width[-1] - sum(nrhs)]  # cut after the main matrix
        for cut in nrhs[0:-1]:                    # we don't need a cut at the end of the matrix
            cuts.append( cuts[-1]+cut )           # cut location is cut cols from previous cut
        partitions = { 1: cuts}

    m.array_format_string_list( partitions=partitions )
    m.array_of_tex_entries(formater=formater)   # could overwride the entry to TeX string conversion here

    if pivot_list is not None:
        red_box = make_decorator( text_color=pivot_text_color, boxed=True, bf=True )
        for spec in pivot_list:
            m.decorate_tex_entries( *spec[0], red_box, entries=spec[1] )

    if func is not None:
        m.apply( func )

    if comment_list is not None:
        m.nm_text( comment_list )

    if variable_summary is not None:
        blue    = make_decorator(text_color=variable_colors[1], bf=True)
        red     = make_decorator(text_color=variable_colors[0], bf=True)
        typ     = []
        var     = []
        for (i,basic) in enumerate(variable_summary):
            if basic is True:
                typ.append(red(r'\Uparrow'))
                var.append(red( f'x_{i+1}'))
            elif basic is False:
                typ.append(blue(r'\uparrow'))
                var.append(blue( f'x_{i+1}'))
            else:
                typ.append(blue(''))
                var.append(blue(''))
        m.add_row_below(m.nGridRows-1,1,typ,           formater=lambda a: a )
        m.add_row_below(m.nGridRows-1,1,var, offset=1, formater=lambda a: a )

    if array_names is not None:
        name_specs = mk_ge_names( m.nGridRows, *array_names, start_index )
    else:
        name_specs = None

    m.nm_submatrix_locs('A',color='blue',name_specs=name_specs) # this defines the submatrices (the matrix delimiters)
    m.tex_repr()                                                # converts the array of TeX entries into strings with separators and spacers

    if bg_for_entries is not None:
        for spec in bg_for_entries:
            if all(isinstance(elem, list) for elem in spec):
                for s in spec:
                    m.nm_background( *s )
            else:
                m.nm_background( *spec )

    if ref_path_list is not None:
        for spec in ref_path_list:
            m.nm_add_rowechelon_path( *spec )

    m_code = m.nm_latexdoc(template = GE_TEMPLATE, fig_scale=fig_scale )

    tex_file,svg_file = itikz.svg_file_from_tex(
        m_code, prefix='ge_', working_dir=tmp_dir, debug=False,
        **itikz.build_commands_dict(use_xetex=True,use_dvi=False,crop=True),
        nexec=1, keep_file=keep_file )


    return m,tex_file,svg_file
# -----------------------------------------------------------------------------------------------
def ge( matrices, Nrhs=0, formater=str, pivot_list=None, bg_for_entries=None,
        variable_colors=['red','blue'], pivot_text_color='red',
        ref_path_list=None, comment_list=None, variable_summary=None, array_names=None,
        start_index=1, func=None, fig_scale=None, tmp_dir="tmp", keep_file=None ):
    '''basic GE layout (development version):
    matrices:         [ [None, A0], [E1, A1], [E2, A2], ... ]
    Nrhs:             number of right hand side columns determines the placement of a partition line, if any
                      can also be a list of widths to be partioned...
    pivot_list:       [ pivot_spec, pivot_spec, ... ] where pivot_spec = [grid_pos, [pivot_pos, pivot_pos, ...]]
    bg_for_entries:   [ bg_spec, ...] where bg_spec = [gM,gN, [ entries ], color, pt ]
    variable_colors:  [ basic_var_color, free_var_color]
    ref_path_list:    [ path_spec, path_spec, ... ] where path_spec = [grid_pos, [pivot_pos], directions ] where directions='vv','vh','hv' or 'hh'
    comment_list:     [ txt, txt, ... ] must have a txt entry for each layer. Multiline comments are separated by \\
    variable_summary: [ basic, ... ]  a list of true/false values specifying whether a column has a pivot or not
    array_names:      list of names for the two columns: [ 'E', ['A','b','I']
    start_index:      first subscript for the elementary operation matrices (can be None)
    func:             a function to be applied to the MatrixGridLayout object prior to generating the latex document
    '''

    m, tex_file, svg_file = _ge( matrices, Nrhs=Nrhs, formater=formater,
                                 pivot_list=pivot_list, bg_for_entries=bg_for_entries,
                                 variable_colors=variable_colors,pivot_text_color=pivot_text_color,
                                 ref_path_list=ref_path_list, comment_list=comment_list,
                                 variable_summary=variable_summary, array_names=array_names,
                                 start_index=start_index, func=func, fig_scale=fig_scale,
                                 tmp_dir=tmp_dir, keep_file=keep_file)

    with open(svg_file, "r") as fp:
        svg = fp.read()
    if svg is not None:
        return SVG(svg), m

    return None, m
# -----------------------------------------------------------------------------------------------
def _to_svg_str( matrices, Nrhs=0, formater=str, pivot_list=None, bg_for_entries=None,
        variable_colors=['red','blue'], pivot_text_color='red',
        ref_path_list=None, comment_list=None, variable_summary=None, array_names=None,
        start_index=1, func=None, fig_scale=None, tmp_dir="tmp", keep_file=None ):
    '''basic GE layout (development version):
    matrices:         [ [None, A0], [E1, A1], [E2, A2], ... ]
    Nrhs:             number of right hand side columns determines the placement of a partition line, if any
                      can also be a list of widths to be partioned...
    pivot_list:       [ pivot_spec, pivot_spec, ... ] where pivot_spec = [grid_pos, [pivot_pos, pivot_pos, ...]]
    bg_for_entries:   [ bg_spec, ...] where bg_spec = [gM,gN, [ entries ], color, pt ]
    variable_colors:  [ basic_var_color, free_var_color]
    ref_path_list:    [ path_spec, path_spec, ... ] where path_spec = [grid_pos, [pivot_pos], directions ] where directions='vv','vh','hv' or 'hh'
    comment_list:     [ txt, txt, ... ] must have a txt entry for each layer. Multiline comments are separated by \\
    variable_summary: [ basic, ... ]  a list of true/false values specifying whether a column has a pivot or not
    array_names:      list of names for the two columns: [ 'E', ['A','b','I']
    start_index:      first subscript for the elementary operation matrices (can be None)
    func:             a function to be applied to the MatrixGridLayout object prior to generating the latex document
    '''

    m, tex_file, svg_file = _ge( matrices, Nrhs=Nrhs, formater=formater,
                                 pivot_list=pivot_list, bg_for_entries=bg_for_entries,
                                 variable_colors=variable_colors,pivot_text_color=pivot_text_color,
                                 ref_path_list=ref_path_list, comment_list=comment_list,
                                 variable_summary=variable_summary, array_names=array_names,
                                 start_index=start_index, func=func, fig_scale=fig_scale,
                                 tmp_dir=tmp_dir, keep_file=keep_file)

    with open(svg_file, "r") as fp:
        svg = fp.read()

    return svg
# ================================================================================================================================
def _q_gram_schmidt( v_list ):
    w = []
    for j in range( len( v_list )):
        w_j = v_list[j]
        for k in range( j-1 ):
            w_j = w_j - w[k].dot( v_list[j]) * w[k]
        w.append(1/w_j.norm() * w_j)
    return w

def compute_qr_matrices( A, W ):
    ''' given the matrix A and the corresponding matrix W with orthogonal columns,
    compute the list of list of sympy matrices in a QR layout
    '''
    A    = sym.Matrix(A)
    W    = sym.Matrix(W)
    WtW  = W.T @ W
    WtA  = W.T @ A
    S    = sym.Matrix.diag( list( map(lambda x: 1/sym.sqrt(x), sym.Matrix.diagonal( WtW))))

    Qt = S*W.T
    R  = S*WtA

    matrices =  [ [ None,  None,   A,    W ],
                  [ None,   W.T, WtA,  WtW ],
                  [ S,       Qt,   R, None ] ]
    return matrices

def _qr(matrices, formater=str, array_names=True, fig_scale=None, tmp_dir="tmp", keep_file=None):
    m = MatrixGridLayout( matrices, extra_rows = [1,0,0,0])
    m.preamble = preamble + '\n' + r" \NiceMatrixOptions{cell-space-limits = 2pt}"+'\n'

    N = matrices[0][2].shape[1]

    m.array_format_string_list()
    m.array_of_tex_entries(formater=formater)

    brown    = make_decorator(text_color='brown', bf=True )
    def qr_dec_known_zeros( WtA, WtW ):
        l_WtA = [(1,2), [(i,j) for i in range(WtA.shape[0]) for j in range(WtA.shape[0]) if i >  j ]]
        l_WtW = [(1,3), [(i,j) for i in range(WtW.shape[0]) for j in range(WtW.shape[0]) if i != j ]]
        return  [l_WtA, l_WtW]

    for spec in qr_dec_known_zeros( matrices[1][2], matrices[1][3]):
        #[ [(1,2), [(1,0),(2,0),(2,1)]], [(1,3), [(1,0),(2,0),(2,1), (0,1),(0,2),(1,2)]] ]:
        m.decorate_tex_entries( *spec[0], brown, entries=spec[1] )

    red      = make_decorator(text_color='red',  bf=True)
    red_rgt  = make_decorator(text_color='red',  bf=True, move_right=True)
    m.add_row_above(0,2, [red(f'v_{i+1}')   for i in range(N)] + [red(f'w_{i+1}') for i in range(N)], formater= lambda a: a )
    m.add_col_left( 1,1, [red_rgt(f'w^t_{i+1}') for i in range(N)], formater= lambda a: a )

    if array_names:
        dec = make_decorator(bf=True, delim='$')
        m.nm_submatrix_locs( 'QR', color='blue', name_specs=[
            [(0,2), 'al', dec('A')],
            [(0,3), 'ar', dec('W')],
            # ----------------------
            [(1,1), 'al', dec('W^t')],
            [(1,2), 'al', dec('W^t A')],
            [(1,3), 'ar', dec('W^t W')],
            # ----------------------
            [(2,0), 'al', dec(r'S = \left( W^t W \right)^{-\tfrac{1}{2}}')],
            [(2,1), 'br', dec(r'Q^t = S W^t')],
            [(2,2), 'br', dec('R = S W^t A')]
        ])
    else:
        m.nm_submatrix_locs()

    m.tex_repr( blockseps = r'\noalign{\vskip3mm} ')

    m_code = m.nm_latexdoc( fig_scale=fig_scale )


    tex_file,svg_file = itikz.svg_file_from_tex(
            m_code, prefix='qr_', working_dir=tmp_dir, debug=False,
            **itikz.build_commands_dict(use_xetex=True,use_dvi=False,crop=True),
            nexec=1, keep_file=keep_file )

    return m,tex_file,svg_file
# -----------------------------------------------------------------------------------------------
def qr(matrices, formater=str, array_names=True, fig_scale=None, tmp_dir="tmp", keep_file=None):
    m,tex_file,svg_file = _qr(matrices, formater=formater, array_names=array_names, fig_scale=fig_scale, tmp_dir=tmp_dir, keep_file=keep_file)

    with open(svg_file, "r") as fp:
        svg = fp.read()
    if svg is not None:
        return SVG(svg), m

    return None, m

def gram_schmidt_qr( A_, W_, formater=sym.latex, fig_scale=None, tmp_dir="tmp" ):
    A = sym.Matrix( A_ )
    W = sym.Matrix( W_ )

    WtW  = W.T @ W
    WtA  = W.T @ A
    S    = WtW**(-1)
    for i in range(S.shape[0]):
        S[i,i]=sym.sqrt(S[i,i])

    Qt = S*W.T
    R  = S*WtA

    matrices =  [ [ None,  None,   A,    W ],
                  [ None,   W.T, WtA,  WtW ],
                  [ S,       Qt,   R, None ] ]
    h,m = qr( matrices, formater=formater, array_names=True, fig_scale=fig_scale, tmp_dir=tmp_dir )
    return h,m

# ==================================================================================================
# BACKSUBSTITUTION
# ==================================================================================================
class BacksubstitutionCascade:
    def __init__(self, ref_A, ref_rhs = None ):
        self.ref_syseq( ref_A, ref_rhs=ref_rhs)

    @classmethod
    def from_ref_Ab(cls, ref_Ab):
        """create `cls` from augmented row echelon form matrix Ab"""
        ref_Ab = sym.Matrix( ref_Ab )
        return cls( ref_Ab[:,0:-1], ref_Ab[:,-1] )

    def ref_syseq(self, ref_A, ref_rhs = None ):
        self.ref_A   = sym.Matrix(ref_A)
        self.ref_rhs = None if ref_rhs is None else sym.Matrix( ref_rhs ).reshape( self.ref_A.shape[0], 1 )

        if ref_rhs is None:
            self.rref_A, self.pivot_cols = self.ref_A.rref()
            self.rref_rhs = None
        else:
            Ab = self.ref_A.row_join( self.ref_rhs )
            rref_Ab, pivot_cols_Ab = Ab.rref()
            self.rref_A   = rref_Ab[:,0:-1]
            self.rref_rhs = rref_Ab[:,-1]
            if pivot_cols_Ab[-1] == self.rref_A.shape[1]:
                self.pivot_cols = pivot_cols_Ab[:-1]
            else:
                self.pivot_cols = pivot_cols_Ab

        #return cls( ref_Ab[:,0:-1], ref_Ab[:,-1] )

        self.free_cols = [ i for i in range(ref_A.shape[1]) if i not in self.pivot_cols]
        self.rank      = len(    self.pivot_cols)

    def ref_rhs( self, rhs ):
        self.ref_rhs = None if rhs is None else sym.Matrix( rhs )

    def ref_Ab( self, Ab ):
        Ab = sym.Matrix( Ab )
        self.ref_syseq( Ab[:,0:-1], Ab[:,-1] )

    @staticmethod
    def _bs_equation( ref_A, pivot_row, pivot_col, rhs=None, name="x" ):
        """given a row, generate the right hand terms from A_ref for the back substitution algorithm"""
        t = sym.Integer(0) if rhs is None else rhs[pivot_row]
        for j in range(pivot_col+1, ref_A.shape[1]):
            t = t - ref_A[pivot_row,j]*sym.Symbol(f"{name}_{j+1}")

        if t.is_zero: return f" 0 "

        factor = 1/ref_A[pivot_row,pivot_col]

        if   factor ==  1: return(sym.latex(t))
        elif factor == -1: return f"- \\left( {sym.latex(t)} \\right)"
        else:              return f"{sym.latex(factor)} \\left( {sym.latex(t)} \\right)"

    def _gen_back_subst_eqs( self ):
        """generate the equations for the back substitution algorithm"""

        alpha = r'\alpha'
        x     = 'x'
        bs    = []
        if len(self.free_cols) > 0:
            bs.append( ',\\;'.join([f"x_{i+1} = {alpha}_{i+1}" for i in self.free_cols] ))
            start = self.rank-1
        else:
            bs.append( f"x_{self.rank} = {BacksubstitutionCascade._bs_equation(self.ref_A,self.rank-1,self.pivot_cols[-1], self.ref_rhs, name=alpha )}")
            start = self.rank-2

        for i in range(start,-1, -1):
            bs.append( [
                f"x_{self.pivot_cols[i]+1} = {BacksubstitutionCascade._bs_equation(     self.ref_A,i,self.pivot_cols[i], self.ref_rhs, name=x )}",
                f"x_{self.pivot_cols[i]+1} = {BacksubstitutionCascade._bs_equation(self.rref_A,i,self.pivot_cols[i], self.rref_rhs, name=alpha )}"
            ])
        return bs

    def particular_solution(self):
        p = sym.Matrix.zeros( self.rref_A.shape[1], 1)
        ps = self.ref_A[0:self.rank,self.pivot_cols]**-1 * self.ref_rhs[0:self.rank,0]
        for i,c in enumerate(self.pivot_cols):
            p[c,0] = ps[i,0]
        return p

    def homogeneous_solution(self):
        hs = sym.Matrix.zeros( self.rref_A.shape[1], len(self.free_cols))

        for (i,col) in enumerate(self.free_cols):
            hs[col,i] = 1
        for (i,col) in enumerate(self.pivot_cols):
            hs[col, :] = -self.rref_A[i,self.free_cols]
        return hs

    def _gen_solution_eqs( self ):
        """generate the solution equations"""
        #lft = r" \left( \begin{array}{r} "
        #rgt = r" \end{array} \right)"
        lft = r'\begin{pNiceArray}{r}'
        rgt = r'\end{pNiceArray}'

        x = lft + r" \\ ".join( [f" x_{i+1}" for i in range(self.ref_A.shape[1]) ] ) + rgt

        if self.ref_rhs is None:
            p    = ""
            plus = ""
        else:
            ps   = self.particular_solution()
            p    = lft + r" \\ ".join( [ sym.latex(ps[i,0])  for i in range(self.ref_A.shape[1]) ] ) + rgt
            plus = " + " if len(self.free_cols) > 0 else ""

        if len(self.free_cols) > 0:
            hs = self.homogeneous_solution()
            h_txt = []
            for j,jv in enumerate( self.free_cols ):
                h = f"\\alpha_{jv+1} "  + lft +  r" \\ ".join( [ sym.latex(hs[i,j])  for i in range(self.ref_A.shape[1]) ] ) + rgt
                h_txt.append( h )

            h_txt = " + ".join(h_txt)
        else:
            h_txt=""

        return "$ " + x + " = " + p + plus + h_txt + " $"

    @staticmethod
    def gen_system_eqs( A, b ):
        """generate the system equations"""
        var = set()
        def mk( j, v ):
            var.add(f"x_{j}")
            try:
                if v > 0:
                    if v ==  1: s = [" + ",                 f"x_{j}" ]
                    else:       s = [" + ", sym.latex( v ), f"x_{j}" ]
                elif v < 0:
                    if v == -1: s = [" - ",                 f"x_{j}" ]
                    else:       s = [" - ", sym.latex(-v ), f"x_{j}" ]
                else:
                    s = [""]
            except:
                vs = sym.latex(v)
                if vs.find("+") < 0 or vs.find("-") < 0:
                    s = [ " + ", "( ", vs.replace("+",r"\+").replace("-",r"\-"), " ) ", f"x_{j}" ]
                else:
                    s = [ " + ", sym.latex(v), f"x_{j}" ]

            return s

        A   = sym.Matrix( A )
        b   = sym.Matrix( b ).reshape( A.shape[0], 1 )
        eqs = []
        for i in range( A.shape[0] ):
            terms = []
            for j in range( A.shape[1] ):
                if not A[i,j].is_zero:
                    terms.extend( mk( j+1, A[i,j] ))

            if len(terms) == 0: terms = [" 0 "]
            if terms[0] == " + ":
                terms = terms[1:]
            eqs.append( "".join( terms ) + " = " + sym.latex( b[i,0] ) )
        return r"\sysdelim.\}\systeme[" + "".join(sorted(var)) + "]{ " + ",".join( eqs ) + "}"

    @staticmethod
    def _mk_cascade( bs ):
        def mk_args( bs ):
            mbs = [ f"   {{$\\boxed{{ {bs[0]}  }}$}}%" ]
            for term in bs[1:]:
                mbs.append( [f"   {{${term[0]}$}}%", f"   {{$\;\Rightarrow\; \\boxed{{ {term[1]} }}$}}%"])
            return mbs

        mbs   = mk_args(bs)
        num_c = len(mbs)-1
        lines = num_c*[ r"{\ShortCascade%"]
        lines.append( mbs[0] )
        for i in range(num_c):
            lines.extend( mbs[i+1] )
            lines.append( r"}%")

        return lines

    def nm_latex_doc( self, A=None, b=None, show_system=False, show_cascade=True, show_solution=False, fig_scale=None ):
        if show_system:
           if A is None or b is None:
               system_txt = BacksubstitutionCascade.gen_system_eqs( self.ref_A, self.ref_rhs )
           else:
               system_txt = BacksubstitutionCascade.gen_system_eqs( A, b )
        else:
           system_txt = None

        if show_cascade:
            bsA         = self._gen_back_subst_eqs()
            cascade_txt = BacksubstitutionCascade._mk_cascade(bsA)  # why at times is this a tuple with a list entry?????  FIX
        else:
            cascade_txt = None

        if show_solution:
            solution_txt = self._gen_solution_eqs()
        else:
            solution_txt = None

        return jinja2.Template( BACKSUBST_TEMPLATE, block_start_string='{%%', block_end_string='%%}',
                 comment_start_string='{##', comment_end_string='##}' ).render( \
                 preamble       = r" \NiceMatrixOptions{cell-space-limits = 1pt}"+'\n',
                 show_system    = show_system,  system_txt   = system_txt,
                 show_cascade   = show_cascade, cascade_txt  = cascade_txt[0] if type(cascade_txt) is tuple else cascade_txt,
                 show_solution  = show_solution,solution_txt = solution_txt,
                 fig_scale      = fig_scale
               )

    def show(self, A=None, b=None, show_system=False, show_cascade=True, show_solution=False, fig_scale=None, keep_file=None, tmp_dir="tmp" ):
        code = self.nm_latex_doc( A=A, b=b, show_system=show_system, show_cascade=show_cascade, show_solution=show_solution, fig_scale=fig_scale)

        h = itikz.fetch_or_compile_svg(
                code, prefix='backsubst_', working_dir=tmp_dir, debug=False,
                **itikz.build_commands_dict(use_xetex=True,use_dvi=False,crop=True),
                nexec=1, keep_file=keep_file )
        return h

# ==================================================================================================
# EigenProblem Tables
# ==================================================================================================
class EigenProblemTable:
    ''' Basic Computations of the EigenDecomposition tables and the resulting Tex Representation
        Indexing is zero-based.
    '''

    def __init__(self, eig, formater=None, eig_digits=None, sigma_digits=None, vec_digits=None, sz=None ):
        '''
        eig:  dictionary with entries
              'lambda'        :    distinct eigenvalues
              'sigma'         :    distinct singular values
              'ma'            :    corresponding algebraic multiplicites
              'evecs'         :    list of mg vectors for each eigenvalue
              'qvecs'         :    list of mg orthonormal vectors for each eigenvalue
              'uvecs'         :    list of mg orthonormal vectors for each singular value
              'sz'            :    size (M,N) of the matrix needed for the SVD table
        '''

        self.N     = sum(eig['ma'])
        self.sz    = (self.N,self.N) if sz is None else sz

        self.ncols = 2*len(eig['lambda'])-1
        self.color = "blue"

        self.eig   = eig
        self._round( eig_digits=eig_digits, sigma_digits=sigma_digits, vec_digits=vec_digits )

        if formater is not None:
           self.eig['lambda'] = list( map( formater, eig['lambda']) )

           if 'sigma' in eig.keys():
              self.eig['sigma'] = list( map( formater, eig['sigma']) )

           self.eig['ma']    = eig['ma']

           if 'evecs' in eig.keys():
              self.eig['evecs'] = self._mk_vectors('evecs', formater=formater )
           if 'qvecs' in eig.keys():
              self.eig['qvecs'] = self._mk_vectors('qvecs', formater=formater )
           if 'uvecs' in eig.keys():
              self.eig['uvecs'] = self._mk_vectors('uvecs', formater=formater )


        self.tbl_fmt   = self._mk_table_format()
        self.rule_fmt  = self._mk_rule_format()

    def _round( self, eig_digits=None, sigma_digits=None, vec_digits=None ):
        if eig_digits is not None and 'lambda' in self.eig.keys():
            f = lambda x: round(x) if eig_digits==0 else round(x,eig_digits)
            self.eig['lambda'] = list( map( f, self.eig['lambda'] ))
        if sigma_digits is not None and 'sigma' in self.eig.keys():
            f = lambda x: round(x) if sigma_digits==0 else round(x,sigma_digits)
            self.eig['sigma'] = list( map( f, self.eig['sigma'] ))
        if vec_digits is not None:
            f = lambda x: round(x) if vec_digits==0 else round(x,vec_digits)
            if 'evecs' in self.eig.keys():
                self.eig['evecs'] = self._mk_vectors('evecs', f)
            if 'qvecs' in self.eig.keys():
                self.eig['qvecs'] = self._mk_vectors('qvecs', f)
            if 'uvecs' in self.eig.keys():
                self.eig['uvecs'] = self._mk_vectors('uvecs', f)


    def _mk_table_format( self ):
        fmt = '{@{}l' + self.ncols*'c' + '@{}}'
        return fmt

    def _mk_rule_format( self ):
        fmt = ''
        for i in range(1, len(self.eig['lambda'])+1):
            fmt += ' \\cmidrule{' + f'{2*i}-{2*i}' + '}'
        return fmt

    def _mk_values( self, key='lambda', formater=str ):
        l = list( map( formater, self.eig[key]) )
        if key == 'sigma' and self.eig[key][-1] == '0':
            l[-1] = formater(' ')
        l = list( map( lambda x: '$'+x+'$', l) )

        ll=[l[0]]
        for i in l[1:]:
            ll.append('')
            ll.append( i)
        return ll

    def mk_values( self, key, formater=str ):
        line = self._mk_values(key=key,  formater=formater)
        return " & ".join( line ) + r' \\' # + self.rule_fmt

    def _mk_vectors(self, key, formater=str ):
        groups = []
        for vecs in self.eig[key]:
            l_lambda   = []
            for vec in vecs:
                l_lambda.append( np.array( [ formater(v) for v in vec], dtype=object ))
            groups.append( l_lambda )
        return groups

    def mk_vectors(self, key, formater=str, add_height=0 ):
        groups = []
        nl = r' \\ ' if add_height == 0 else r' \\'+ f'[{add_height}mm] '
        for vecs in self.eig[key]:
            l_lambda   = []
            for vec in vecs:
                l = [ formater(v) for v in vec]
                l_lambda.append( r'$\begin{pNiceArray}{r}' + nl.join(l) + r' \end{pNiceArray}$')
            groups.append( ', '.join(l_lambda ))
        return " & & ".join( groups )

    def _mk_diag_matrix( self, key='lambda', mm=8, formater=str ):
        space   = '@{\\hspace{' + str(mm) + 'mm}}'
        pre     = r'\multicolumn{' + f'{len(self.eig["ma"])}' + '}{c}{\n'+\
                  r'$\begin{pNiceArray}{' + space.join( self.N*['c'] ) + '}'
        post    = r'\end{pNiceArray}$}'

        Lambda  = np.full( self.sz, formater(0), dtype=object)
        lambdas = []
        for i,v in enumerate( self.eig['ma'] ):
            l       = self.eig[key][i]
            lambdas += v*[l]

        for i,v in enumerate(lambdas):
            try:
                Lambda[i,i] = formater(v)
            except:
                break

        return pre,Lambda,post

    def _mk_evecs_matrix( self, key='evecs', formater=str, mm=0 ):
        sz      = self.sz[0] if key == 'uvecs' else self.sz[1]
        space   = '@{\\hspace{' + str(mm) + 'mm}}' if mm > 0 else ''
        pre     = r'\multicolumn{' + f'{len(self.eig["ma"])}' + '}{c}{\n'+\
                  r'$\begin{pNiceArray}{' + space.join( sz*['r'] ) + '}'
        post    = r'\end{pNiceArray}$}'

        S  = np.empty( (sz,sz), dtype=object)
        j  = 0
        for vecs in self.eig[key]:
            for vec in vecs:
                for i,v in enumerate(vec):
                    S[i,j] = formater(v)
                j     += 1
        return pre,S,post

    def _fmt_matrix( self, pre, m, post, add_height=0 ):
        nl = r' \\ ' if add_height == 0 else r' \\'+ f'[{add_height}mm] '

        mat = []
        for i in range( m.shape[0]):
            mat.append( ' & '.join( m[i,: ]))
        mat = pre + nl.join( mat ) + r' \\ ' + post
        return mat

    def mk_diag_matrix( self, key, formater=str, mm=8, extra_space='', add_height=0 ):
        pre, m, post = self._mk_diag_matrix(key=key, formater=formater, mm=mm )
        for i in range(m.shape[0]):
            m[i,0]        = extra_space+m[i,0]
            m[i,self.N-1] = m[i,self.N-1]+extra_space
        return self._fmt_matrix( pre, m, post, add_height=add_height )

    def mk_evecs_matrix( self, key, formater=str, mm=8,extra_space='', add_height=0  ):
        pre, m, post = self._mk_evecs_matrix(key=key, formater=formater, mm=mm )
        sz = self.sz[0] if key == 'uvecs' else self.sz[1]
        if m.shape[1] == sz:
            for i in range(m.shape[0]):
                m[i,0] = extra_space+m[i,0]
                m[i,m.shape[1]-1]=m[i,m.shape[1]-1]+extra_space

            return self._fmt_matrix( pre, m, post, add_height )
        else:
            return m

    def decorate_values(self, l, decorate, i=None ):
        if i is not None:
            l[i] = decorate( l[i] )
        else:
            for i in range(0,len(l)):
                l[i] = decorate(l[i])

    def decorate_matrix(self, m, decorate, row=None, col=None ):
        m = m.reshape( m.shape[0],-1 )
        rows = range( m.shape[0] ) if row is None else [ row ]
        cols = range( m.shape[1] ) if col is None else [ col ]
        for i in rows:
            for j in cols:
                m[i,j]=decorate( m[i,j])

    def nm_latex_doc( self, formater=sym.latex, case="S", color='blue',
                      mmLambda=8, mmS=4, spaceLambda=r' \;\; ', spaceS=r' \;\; ',
                      fig_scale = None
        ):
        tbl_fmt   = self._mk_table_format()

        # ------------------------------------------------------ values
        sigmas  = self.mk_values('sigma',  formater=formater ) if case == 'SVD' else None
        lambdas = self.mk_values('lambda', formater=formater)
        mas     = self.mk_values('ma',     formater=formater)

        # ------------------------------------------------------ vectors
        evecs = self.mk_vectors('evecs', formater=formater) + r' \\'

        qvecs = None
        uvecs = None

        if case == 'Q':
            qvecs  = self.mk_vectors('qvecs', formater=formater, add_height=1) + r' \\'
        elif case == 'SVD':
            qvecs  = self.mk_vectors('qvecs', formater=formater, add_height=1) + r' \\'
            try:
                uvecs  = self.mk_vectors('uvecs', formater=formater, add_height=1) + r' \\'
            except:
                uvecs  = None

        # ------------------------------------------------------ matrices
        left_singular_matrix = None

        if case == 'SVD':
            lambda_matrix = self.mk_diag_matrix( 'sigma',  formater=formater, mm=mmLambda)
        else:
            try:
                lambda_matrix = self.mk_diag_matrix( 'lambda', formater=formater, mm=mmLambda)
            except:
                lambda_matrix = None

        if case == 'S':
            try:
                evecs_matrix = self.mk_evecs_matrix( 'evecs', formater=formater, mm=mmS )
            except:
                evecs_matrix  = None
        else:
            evecs_matrix = self.mk_evecs_matrix( 'qvecs', formater=formater, mm=mmS )
            if case == 'SVD':
                try:
                    left_singular_matrix = self.mk_evecs_matrix( 'uvecs', formater=formater, mm=mmS )
                except:
                    left_singular_matrix = None

        if   case == 'S': matrix_names=[r'\Lambda', 'S']
        elif case == 'Q': matrix_names=[r'\Lambda', 'Q']
        else:             matrix_names=[r'\Sigma',  'V', 'U']

        # ------------------------------------------------------ figure scaling
        if fig_scale is not None:
            fig_scale = r'\scalebox{'+str(fig_scale)+'}{%'

        return jinja2.Template( EIGPROBLEM_TEMPLATE, block_start_string='{%%', block_end_string='%%}',
                                          comment_start_string='{##', comment_end_string='##}' ).render( \
                   preamble                 = r" \NiceMatrixOptions{cell-space-limits = 1pt}"+'\n',
                   fig_scale                = fig_scale,
                   matrix_names             = matrix_names,
                   table_format = tbl_fmt, color = '{'+color+'}',
                   rule_format              = self.rule_fmt,
                   sigmas                   = sigmas,
                   lambdas                  = lambdas,
                   algebraic_multiplicities = mas,
                   eigenbasis               = evecs,
                   orthonormal_basis        = qvecs,
                   left_singular_matrix     = left_singular_matrix,
                   lambda_matrix            = lambda_matrix,
                   evecs_matrix             = evecs_matrix
               )
# --------------------------------------------------------------------------------------------------
def eig_tbl(A, normal=False, eig_digits=None,vec_digits=None):
    A = sym.Matrix(A)
    eig = {
        'lambda': [],
        'ma':     [],
        'evecs':  [],
    }
    if normal:
        eig['qvecs'] = []

    res = A.eigenvects()
    for e,m,vecs in res:
        eig['lambda'].insert(0,e)
        eig['ma'].insert(0,m)
        eig['evecs'].insert(0,vecs)
        if normal:
            vvecs = _q_gram_schmidt( vecs )
            eig['qvecs'].insert(0, vvecs )

    return EigenProblemTable( eig,eig_digits=eig_digits, vec_digits=vec_digits )

def show_eig_tbl(A, Ascale=None, normal=False, eig_digits=None, vec_digits=None, formater=sym.latex, mmS=10, mmLambda=8, fig_scale=1.0, color='blue', keep_file=None, tmp_dir="tmp" ):
    E = eig_tbl(A, normal=normal, eig_digits=eig_digits,vec_digits=vec_digits)
    if Ascale is not None:
        E.eig[ 'lambda' ] = [ e/Ascale for e in E.eig[ 'lambda' ]]

    c = 'Q' if normal else 'S'

    svd_code = E.nm_latex_doc( formater=formater, case=c, mmS=mmS, mmLambda=mmLambda, fig_scale=fig_scale, color=color)

    h = itikz.fetch_or_compile_svg(
            svd_code, prefix='svd_', working_dir=tmp_dir, debug=False,
            **itikz.build_commands_dict(use_xetex=True,use_dvi=False,crop=True),
            nexec=1, keep_file=keep_file )
    return h
# --------------------------------------------------------------------------------------------------
def svd_tbl(A, Ascale=None, eig_digits=None, sigma_digits=None, vec_digits=None):
    A   = sym.Matrix(A)
    eig = {
        'sigma':  [],
        'lambda': [],
        'ma':     [],
        'evecs':  [],
        'qvecs':  [],
        'uvecs':  []
    }
    def mySVD(A):
        A = sym.Matrix(A)

        def sort_eig_vec(sym_eig_vec):
            sort_eig_vecs = sorted(sym_eig_vec, key=lambda x: x[0], reverse=True)

            for i in sort_eig_vecs:
                e = i[0] if Ascale is None else i[0] / (Ascale*Ascale)
                sigma = sym.sqrt(e)
                eig['sigma'].append(sigma)

                eig['lambda'].append( e )
                eig['ma'].append( i[1] )
                eig['evecs'].append( i[2] )
                vvecs = _q_gram_schmidt( i[2] )
                eig['qvecs'].append( vvecs )

                if not sigma.is_zero:
                    s_inv = 1/sigma if Ascale is None else 1/(sigma*Ascale)
                    eig['uvecs'].append( [ s_inv * A * v for v in vvecs] )

        sort_eig_vec((A.transpose() * A).eigenvects())
        ns = A.transpose().nullspace()
        if len(ns) > 0:
            ns_on_basis = _q_gram_schmidt( ns )
            eig['uvecs'].append( ns_on_basis )

    mySVD(A)
    return EigenProblemTable( eig, sz=A.shape, eig_digits=eig_digits, sigma_digits=sigma_digits, vec_digits=vec_digits )

def show_svd_table(A, Ascale=None, eig_digits=None, sigma_digits=None, vec_digits=None,
                   formater=sym.latex, mmS=10, mmLambda=8, fig_scale=1.0, color='blue', keep_file=None, tmp_dir="tmp" ):
    E = svd_tbl(A, Ascale=Ascale, eig_digits=eig_digits, sigma_digits=sigma_digits, vec_digits=vec_digits)
    svd_code = E.nm_latex_doc( formater=formater, case='SVD', mmS=mmS, mmLambda=mmLambda, fig_scale=fig_scale, color=color)

    h = itikz.fetch_or_compile_svg(
            svd_code, prefix='svd_', working_dir=tmp_dir, debug=False,
            **itikz.build_commands_dict(use_xetex=True,use_dvi=False,crop=True),
            nexec=1, keep_file=keep_file )
    return h
# ==================================================================================================
def html_string(txt,sz=20,color="darkred",justify="left",height="15"):
    return HTML(f"""<div style=\"float:center;width:100%;text-align:${justify};\">
               <strong style=\"height:${height}px;color:${color};font-size:${sz}pt;\">
               ${txt}
               </strong></div>""")
# --------------------------------------------------------------------------------------------------
def html_strings(txt1,txt2,sz1=20,sz2=20,color="darkred",justify="left",height="15"):
    return HTML("""<div style=\"float:center;width:100%;text-align:${justify};\">
<strong style=\"height:${height}px;color:${color};font-size:${sz1}pt;\">${txt1}</strong><br>
<strong style=\"height:${height}px;color:${color};font-size:${sz2}pt;\">${txt2}</strong><br>
</div>""")


#def foo(x):
#    print( "foo input: ", x)
#    if x is None:
#        print( "None == ", None)
#    if x is not None:
#        print( "not None != ", None)
