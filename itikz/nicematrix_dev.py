import numpy as np
import sympy as sym
import jinja2
import itikz

# ================================================================================================================================
extension = r''' '''
# -----------------------------------------------------------------
preamble = r''' '''
# =================================================================
EIGPROBLEM_TEMPLATE = r'''\documentclass[notitlepage,table,svgnames]{article}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{mathtools}
\usepackage{nicematrix}
\usepackage{xcolor}

\begin{document}
{% if fig_scale %}
{{fig_scale}}
{% endif %}
%====================================================================
\begin{tabular}{{table_format}} \toprule
{% if sigmas %}% sigma -----------------------------------------------------------------
$\color{{color}}{\sigma}$ & {{sigmas}}  {{rule_format}}
{% endif %}% lambda --------------------------------------------------------------------
$\color{{color}}{\lambda}$ & {{lambdas}} {{rule_format}}
$\color{{color}}{m_a}$ & {{algebraic_multiplicities}}  {{rule_format}} \addlinespace[1mm]
%  eigenvectors ------------------------------------------------------------------------
{\parbox{2cm}{\textcolor{{color}}{basis for $\color{{color}}{E_\lambda}$}}} &
{{eigenbasis}} {% if orthonormal_basis %}
%  orthonormal eigenvectors ------------------------------------------------------------
 {{rule_format}} \addlinespace[2mm]
{\parbox{2cm}{\textcolor{{color}}{orthonormal basis for $E_\lambda$}}} &
{{orthonormal_basis}}
{% endif -%}
 \addlinespace[2mm] \midrule \addlinespace[2mm]
% ------------------------------------------------------------- lambda
$\color{{color}}{ {{matrix_names[0]}} =}$ & {{lambda_matrix}} \\ \addlinespace[2mm]
% ------------------------------------------------------------- Q
$\color{{color}}{ {{matrix_names[1]}} = }$ & {{evecs_matrix}} \\  \addlinespace[2mm] \bottomrule
\end{tabular}
{% if fig_scale %}
}
{% endif %}
\end{document}
'''
# =================================================================
GE_TEMPLATE = r'''\documentclass[notitlepage]{article}
\pagestyle{empty}
%\usepackage[paperheight=9in,paperwidth=56in,top=1in,bottom=1in,right=1in,left=1in,heightrounded,showframe]{geometry}

\usepackage{mathtools}
\usepackage{xltxtra}
\usepackage{pdflscape}
\usepackage{graphicx}
\usepackage[table,svgnames]{xcolor}
\usepackage{nicematrix,tikz}
\usetikzlibrary{calc,fit,decorations.markings}
% ---------------------------------------------------------------------------- extension
{{extension}}
\begin{document}
\begin{landscape}
{% if fig_scale %}
{{fig_scale}}
{% endif %}
% ---------------------------------------------------------------------------- preamble
{{preamble}}%
% ============================================================================ NiceArray
$\begin{NiceArray}[vlines-in-sub-matrix = I]{{mat_format}}{{mat_options}}
{% if codebefore != [] -%}
\CodeBefore [create-cell-nodes]
    {% for entry in codebefore: -%}
    {{entry}}
    {% endfor -%}%
\Body
{% endif %}%
{{mat_rep}}
\CodeAfter %[ sub-matrix / extra-height=2mm, sub-matrix / xshift=2mm ]
% --------------------------------------------------------------------------- submatrix delimiters
    {% for loc in submatrix_locs: -%}
          \SubMatrix({{loc[1]}})[{{loc[0]}}]
    {% endfor -%}
    {% for txt in submatrix_names: -%}
          {{txt}}
    {% endfor -%}
% --------------------------------------------------------------------------- pivot outlines
\begin{tikzpicture}
    \begin{scope}[every node/.style = draw]
    {% for loc in pivot_locs: -%}
        \node [draw,{{loc[1]}},fit = {{loc[0]}}]  {} ;
    {% endfor -%}
    \end{scope}
%
% --------------------------------------------------------------------------- explanatory text
    {% for loc,txt,c in txt_with_locs: -%}
        \node [right,align=left,color={{c}}] at {{loc}}  {\qquad {{txt}} } ;
    {% endfor -%}
%
% --------------------------------------------------------------------------- row echelon form path
    {% for t in rowechelon_paths %} {{t}}
    {% endfor -%}
\end{tikzpicture}
\end{NiceArray}$
{% if fig_scale %}
}
{% endif %}
\end{landscape}
\end{document}
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
        self.stylenames       = []
        self.codebefore       = []
        self.preamble         = '%\n'
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
                    s =  decorate(self.a_tex[tl[0]+i, tl[1]+j])
                    self.a_tex[tl[0]+i, tl[1]+j] = s
        except:
            pass

    def matrix_array_format( N, p_str='I', vpartitions=None):
        '''format string for a matrix with N columns'''
        if vpartitions is None:
            return f"*{N}r"
        s     = ""
        cur   = 0
        for p in vpartitions:
            s += f"*{p-cur}r{p_str}"
            cur = p
        if cur < N:
            s += f"*{N-cur}r"
        return s

    #def array_format_string_list( self, partitions={}, spacer_string=r'@{\qquad\ }', p_str='I', last_col_format = "l@{\qquad\;\;}") :
    def array_format_string_list( self, partitions={}, spacer_string=r'@{\hspace{9mm}}', p_str='I', last_col_format=r'l@{\hspace{2cm}}' ):
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
                x = r'\x2' if j >= shape[1] else r'\x4'
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
            p4 = f'\\p4 = ({i+tl[0]+1}-|{j+tl[1]+1}) in '

        cmd = '\\tikz \\draw['+color+'] ' + corners + p3 + p4  + ' -- '.join( [coords(*p) for p in ll] ) + ';'
        self.rowechelon_paths.append( cmd )


    def _mk_entries( self, gM, gN, *args ):
        tl,_,_ = self._top_left_bottom_right( gM, gN )

        def mk(entry):
            if not isinstance( entry, list):
                return [f'({tl[0]+entry[0]+1}-{tl[1]+entry[1]+1})']
            i1,j1 = entry[0]
            i2,j2 = entry[-1]

            return [f'({tl[0]+i+1}-{tl[1]+j+1})' for i in range(i1,i2+1) for j in range(j1,j2+1)]
        l = [mk(entry) for entry in args]
        return [loc for sublist in l for loc in sublist]

    def nm_background( self, gM, gN, entries, style_name='highlight', color='red!15' ):
        if style_name not in self.stylenames:
            self.stylenames.append( style_name )
            self.preamble = self.preamble + r'\tikzset{' + style_name + f'/.style = {{ fit = #1 , fill={color}, inner sep = 2pt }} }} %' + '\n'

        for entry in entries:
            for k in self._mk_entries( gM,gN, entry):
                self.codebefore.append( f'\\tikz \\node [{style_name} = {k}] {{ }} ;')

    def apply( self, func,  *args, **kwargs ):
        func( self, *args, **kwargs )

    def nm_latexdoc( self, template = GE_TEMPLATE, fig_scale=None ):
        if fig_scale is not None:
            fig_scale = r'\scalebox{'+str(fig_scale)+'}{%'
        return jinja2.Template( template ).render( \
                preamble        = self.preamble,
                fig_scale       = fig_scale,
                extension       = self.extension,
                mat_rep         = '\n'.join( self.tex_list ),
                mat_format      = '{'+self.format+'}',
                mat_options     = '',
                submatrix_locs  = self.locs,
                submatrix_names = self.array_names,
                pivot_locs      = [],
                txt_with_locs   = self.txt_with_locs,
                rowechelon_paths= self.rowechelon_paths,
                codebefore      = self.codebefore,
        )
# -----------------------------------------------------------------------------------------------------
def make_decorator( text_color='black', bg_color=None, text_bg=None, boxed=None, bf=None, move_right=False, delim=None ):
    box_decorator         = "\\boxed<{a}>"
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
    def pa(e_prod):
        return r' \mid '.join( [e_prod+' '+k for k in rhs ])

    for i in range(n):
        if start_index is None:
            names[i,0] = f'{lhs}'
        else:
            names[i,0] = f'{lhs}_{start_index+i-1}'

        e_prod = pe(i)
        names[i,1] = pa(e_prod)

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
def ge( matrices, Nrhs=0, formater=str, pivot_list=None, ref_path_list=None, comment_list=None, variable_summary=None, array_names=None,
        start_index=1, func=None, fig_scale=None, tmp_dir=None, keep_file=None ):
    '''basic GE layout (development version):
    matrices:         [ [None, A0], [E1, A1], [E2, A2], ... ]
    Nrhs:             number of right hand side columns determines the placement of a partition line, if any
                      can also be a list of widths to be partioned...
    pivot_list:       [ pivot_spec, pivot_spec, ... ] where pivot_spec = [grid_pos, [pivot_pos, pivot_pos, ...]]
    ref_path_list:    [ path_spec, path_spec, ... ] where path_spec = [grid_pos, [pivot_pos], directions ] where directions='vv','vh','hv' or 'hh'
    comment_list:     [ txt, txt, ... ] must have a txt entry for each layer. Multiline comments are separated by \\
    variable_summary: [ basic, ... ]  a list of true/false values specifying whether a column has a pivot or not
    array_names:      list of names for the two columns: [ 'E', ['A','b','I']
    start_index:      first subscript for the elementart operation matrices (can be None)
    func:             a function to be applied to the MatrixGridLayout object prior to generating the latex document
    '''
    extra_cols = None if comment_list     is None else 1
    extra_rows = None if variable_summary is None else 2

    m = MatrixGridLayout(matrices, extra_rows=extra_rows, extra_cols = extra_cols )

    # compute the format spec for the arrays and set up the entries (defaults to a single partition line)
    if not isinstance( Nrhs, list):
        partitions = {} if Nrhs == 0 else { 1: [m.mat_col_width[-1]-Nrhs]}
    else:
        nrhs = Nrhs.copy(); nrhs.reverse()       # partitions just specifies each col
        cuts = [m.mat_col_width[-1] - sum(nrhs)]
        for cut in nrhs[1:]:
            cuts.append( cuts[-1]+cut )
        partitions = { 1: cuts}

    m.array_format_string_list( partitions=partitions )
    m.array_of_tex_entries(formater=formater)   # could overwride the entry to TeX string conversion here

    if pivot_list is not None:
        red_box = make_decorator( text_color='red', boxed=True, bf=True )
        for spec in pivot_list:
            m.decorate_tex_entries( *spec[0], red_box, entries=spec[1] )

    if func is not None:
        m.apply( func )

    if comment_list is not None:
        m.nm_text( comment_list )

    if variable_summary is not None:
        blue    = make_decorator(text_color='blue', bf=True)
        red     = make_decorator(text_color='red',  bf=True)
        typ     = []
        var     = []
        for (i,basic) in enumerate(variable_summary):
            if basic:
                typ.append(red(r'\Uparrow'))
                var.append(red( f'x_{i+1}'))
            else:
                typ.append(blue(r'\uparrow'))
                var.append(blue( f'x_{i+1}'))
        m.add_row_below(m.nGridRows-1,1,typ,           formater=lambda a: a )
        m.add_row_below(m.nGridRows-1,1,var, offset=1, formater=lambda a: a )

    if array_names is not None:
        name_specs = mk_ge_names( m.nGridRows, *array_names, start_index )
    else:
        name_specs = None

    m.nm_submatrix_locs('A',color='blue',name_specs=name_specs) # this defines the submatrices (the matrix delimiters)
    m.tex_repr()                                                # converts the array of TeX entries into strings with separators and spacers

    if ref_path_list is not None:
        for spec in ref_path_list:
            m.nm_add_rowechelon_path( *spec )

    m_code = m.nm_latexdoc(template = GE_TEMPLATE, fig_scale=fig_scale )

    h = itikz.fetch_or_compile_svg(
        m_code, prefix='ge_', working_dir=tmp_dir, debug=False,
        **itikz.build_commands_dict(use_xetex=True,use_dvi=False,crop=True),
        nexec=1, keep_file=keep_file )

    return h, m

# ================================================================================================================================
def qr(matrices, formater=str, array_names=True, fig_scale=None, tmp_dir=None, keep_file=None):
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


    h = itikz.fetch_or_compile_svg(
            m_code, prefix='qr_', working_dir=tmp_dir, debug=False,
            **itikz.build_commands_dict(use_xetex=True,use_dvi=False,crop=True),
            nexec=1, keep_file=keep_file )
    return h, m

def gram_schmidt_qr( A_, W_ ):
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
    h,m = qr( matrices, formater=sym.latex, array_names=True, tmp_dir="tmp" )
    return h,m

# ==================================================================================================
# EigenProblem Tables
# ==================================================================================================
class EigenProblemTable:
    ''' Basic Computations of the EigenDecomposition tables and the resulting Tex Representation
        Indexing is zero-based.
    '''

    def __init__(self, eig, formater=None ):
        '''
        eig:  dictionary with entries
              'lambda'        :    distinct eigenvalues
              'sigma'         :    distinct singular values
              'ma'            :    corresponding algebraic multiplicites
              'evecs'         :    list of mg vectors for each eigenvalue
              'qvecs'         :    list of mg orthonormal vectors for each eigenvalue
        '''

        self.N     = sum(eig['ma'])
        self.ncols = 2*len(eig['lambda'])-1
        self.color = "blue"

        self.eig   = eig
        if formater is not None:
           f_eig = {}
           f_eig['lambda'] = list( map( formater, eig['lambda']) )

           if 'sigma' in eig.keys():
              f_eig['sigma'] = list( map( formater, eig['sigma']) )

           f_eig['ma']    = eig['ma']

           if 'evecs' in eig.keys():
              f_eig['evecs'] = self._mk_vectors('evecs', formater=formater )
           if 'qvecs' in eig.keys():
              f_eig['qvecs'] = self._mk_vectors('qvecs', formater=formater )
           self.eig = f_eig

        self.tbl_fmt   = self._mk_table_format()
        self.rule_fmt  = self._mk_rule_format()

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

        Lambda  = np.full( (self.N,self.N), formater(0), dtype=object)
        lambdas = []
        for i,v in enumerate( self.eig['ma'] ):
            lambdas += v*[self.eig[key][i]]
        for i,v in enumerate(lambdas):
            Lambda[i,i] = formater(v)

        return pre,Lambda,post

    def _mk_evecs_matrix( self, key='evec', formater=str, mm=0 ):
        space = '@{\\hspace{' + str(mm) + 'mm}}' if mm > 0 else ''
        pre     = r'\multicolumn{' + f'{len(self.eig["ma"])}' + '}{c}{\n'+\
                  r'$\begin{pNiceArray}{' + space.join( self.N*['r'] ) + '}'
        post    = r'\end{pNiceArray}$}'

        S  = np.empty( (self.N,self.N), dtype=object)
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
            m[i,0] = extra_space+m[i,0]
            m[i,self.N-1]=m[i,self.N-1]+extra_space
        return self._fmt_matrix( pre, m, post, add_height=add_height )

    def mk_evecs_matrix( self, key, formater=str, mm=8,extra_space='', add_height=0  ):
        pre, m, post = self._mk_evecs_matrix(key=key, formater=formater, mm=mm )
        for i in range(m.shape[0]):
            m[i,0] = extra_space+m[i,0]
            m[i,self.N-1]=m[i,self.N-1]+extra_space

        return self._fmt_matrix( pre, m, post, add_height )

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

        if case != 'S':
            qvecs  = self.mk_vectors('qvecs', formater=formater, add_height=1) + r' \\'
        else:
            qvecs = None

        # ------------------------------------------------------ matrices
        if case == 'SVD':
            lambda_matrix = self.mk_diag_matrix( 'sigma',  formater=formater, mm=mmLambda)
        else:
            lambda_matrix = self.mk_diag_matrix( 'lambda', formater=formater, mm=mmLambda)

        if case == 'S':
            evecs_matrix = self.mk_evecs_matrix( 'evecs', formater=formater, mm=mmS )
        else:
            evecs_matrix = self.mk_evecs_matrix( 'qvecs', formater=formater, add_height=1, mm=mmS )

        if   case == 'S': matrix_names=[r'\Lambda', 'S']
        elif case == 'Q': matrix_names=[r'\Lambda', 'Q']
        else:             matrix_names=[r'\Sigma_r', 'V_r']

        if fig_scale is not None:
            fig_scale = r'\scalebox{'+str(fig_scale)+'}{%'

        return jinja2.Template( EIGPROBLEM_TEMPLATE ).render(
                   fig_scale                = fig_scale,
                   matrix_names             = matrix_names,
                   table_format = tbl_fmt, color = '{'+color+'}',
                   rule_format              = self.rule_fmt,
                   sigmas                   = sigmas,
                   lambdas                  = lambdas,
                   algebraic_multiplicities = mas,
                   eigenbasis               = evecs,
                   orthonormal_basis        = qvecs,
                   lambda_matrix            = lambda_matrix,
                   evecs_matrix             = evecs_matrix
               )
# --------------------------------------------------------------------------------------------------
# E = EigenProblemTable(  # requires a dictionary
#        {
#            'lambda': [3,           2, 1],
#            'ma':     [2,           1, 1],
#            'sigma':  [sym.sqrt(3), sym.sqrt(2), 1],
#            'evecs':  [[sym.Matrix([1, 2, 1,0]), sym.Matrix([3, -1, 1,0])],   # list of lists of vectors, one for each eigenvalue
#                       [sym.Matrix([2, 1, 2,1])],
#                       [sym.Matrix([1, 1, 0,0])]],
#            'qvecs':  [[sym.Matrix([1, 2, 1,0])/sym.sqrt(6), sym.Matrix([3, -1, 1,0])/sym.sqrt(11)],
#                       [sym.Matrix([2, 1, 2,1])/sym.sqrt(10)],
#                       [sym.Matrix([1, 1, 0,0])/sym.sqrt(2)]]
#        }
#
#  )

# ==================================================================================================
# New Examples
# ==================================================================================================
#    k  = sym.Symbol('k'); h = sym.Symbol('h')
#    Ab = sym.Matrix([[1,2,4,1],[2,k,8,h],[3,7,3,1]]); matrices = [[None, Ab]]; pivots = []; txt=[]
#    # we could use row ops, but we want a computational layout (and hence the E matrices!):
#    #    A=A.elementary_row_op('n->n+km', k=-3, row1=2,row2=0 );A
#    #    A=A.elementary_row_op('n<->m',row1=1,row2=2);A
#
#    E1=sym.eye(3);E1[1:,0]=[-2,-3]; A1=E1*Ab;                               matrices.append([E1,A1]); pivots.append((1,1));txt.append('Pivot at (1,1)')
#    E2=sym.eye(3);E2=E2.elementary_row_op('n<->m',row1=1,row2=2); A2=E2*A1; matrices.append([E2,A2]); pivots.append(None); txt.append('Rows 2 <-> 3')
#    E3=sym.eye(3);E3[2,1]=4-k; A3=E3*A2;                                    matrices.append([E3,A3]); pivots.append((2,2));txt.append('Pivot at (2,2)')
#    pivots.append((3,3)); txt.append('In Row Echelon Form')
#
# m3 = nM.MatrixGridLayout(matrices, extra_cols=1)
# m3.array_format_string_list( partitions={ 1:[3]} )
# m3.array_of_tex_entries()
# red_box = nM.make_decorator( text_color='red', boxed=True, bf=True )
# m3.decorate_tex_entries( 0,1, red_box, entries=[(0,0)] )
# m3.decorate_tex_entries( 1,1, red_box, entries=[(0,0),(1,1)] )
# m3.decorate_tex_entries( 2,1, red_box, entries=[(0,0),(1,1)] )
# m3.decorate_tex_entries( 3,1, red_box, entries=[(0,0),(1,1),(2,2)] )
#
# m3.nm_text(txt)
#
# m3.nm_submatrix_locs()
# m3.tex_repr( blockseps = r'\noalign{\vskip2mm}')
#
# m3_code = m3.nm_latexdoc(template = nM.GE_TEMPLATE )
#
# if True:
#     h = itikz.fetch_or_compile_svg(
#         m3_code, prefix='tst_', working_dir='/tmp/itikz', debug=False,
#         **itikz.build_commands_dict(use_xetex=True,use_dvi=False,crop=True),
#         nexec=4, keep_file="/tmp/itikz/m3" )
# h
# ==================================================================================================
# OLD EXAMPLES
# ==================================================================================================
#loc_format( (1,2), parens=('{','}') )
#pivot_locations([(1,3),(2,4)], n_layers=4, M=4, row_offset=10, col_offset=0)
#
#old_submatrix_locations( 2, (3,5), row_offset=1, col_offset=1+3, start_at_layer=0)
#submatrix_locations( layers, row_offset=1, col_offset=1+3, start_at_layer=0)
###########################################################
#pivots =[]; n_layers=0
#A  = np.array([[1.,2,1,  9,9],[3,4,5, 9,9], [5,6,1, 9,9]]); pivots.append((1,1)); n_layers=1
#E1 = np.array([[1,0,0],[-3,1,0],[-5, 0,1]]); A1 = E1 @ A;   pivots.append((2,2)); n_layers=2
#E2 = np.array([[1,0,0],[ 0,1,0],[ 0,-2,1]]); A2 = E2 @ A1;  pivots.append((3,3)); n_layers=3
#
#mat_rep, submatrix_locs, pivot_locs, path_corners,txt_with_locs,mat_format = nM.ge_int_layout( [[None, A], [E1, A1], [E2,A2]], pivots)
#
#print("mat_rep ="); print(mat_rep)
#print("pivot_locs =");pivot_locs
#print("submatrix_locs =");submatrix_locs
