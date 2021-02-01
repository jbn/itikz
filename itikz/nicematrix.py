import numpy as np
import sympy as sym
import jinja2
import itikz

extension = r'''
\ExplSyntaxOn
\makeatletter


\dim_new:N \l__submatrix_extra_height_dim
\dim_new:N \l__submatrix_left_xshift_dim
\dim_new:N \l__submatrix_right_xshift_dim

\keys_define:nn { SubMatrix }
  {
    extra-height .dim_set:N = \l__submatrix_extra_height_dim ,
    extra-height .value_required:n = true ,
    left-xshift .dim_set:N = \l__submatrix_left_xshift_dim ,
    left-xshift .value_required:n = true ,
    right-xshift .dim_set:N = \l__submatrix_right_xshift_dim ,
    right-xshift .value_required:n = true ,
  }

\NewDocumentCommand { \SubMatrixOptions } { m }
  { \keys_set:nn { SubMatrix } { #1 } }


\NewDocumentCommand \SubMatrix { m m m m ! O { } }
  {
    \keys_set:nn { SubMatrix } { #5 }
    \begin { tikzpicture }
      [
        outer~sep =0 pt ,
        inner~sep = 0 pt ,
        draw = none ,
        fill = none ,
      ]
    \pgf@process
      {
        \pgfpointdiff
          { \pgfpointanchor { nm - \NiceMatrixLastEnv - #3 - medium } { south } }
          { \pgfpointanchor { nm - \NiceMatrixLastEnv - #2 - medium } { north } }
      }
    \dim_set_eq:NN \l_tmpa_dim \pgf@y
    \dim_add:Nn \l_tmpa_dim \l__submatrix_extra_height_dim
    \node
      at
      (
        [ xshift = -0.8 mm - \l__submatrix_left_xshift_dim ]
        $ (#2-medium.north~west) ! .5 ! (#3-medium.south~east-|#2-medium.north~west) $
      )
      {
        \nullfont
        \c_math_toggle_token
        \left #1
        \vcenter { \nullfont \hrule height .5 \l_tmpa_dim depth .5 \l_tmpa_dim width 0 pt }
        \right .
        \c_math_toggle_token
      } ;
    \node
      at
      (
        [ xshift = 0.8 mm + \l__submatrix_right_xshift_dim ]
        $ (#2-medium.north~west-|#3-medium.south~east) ! .5 ! (#3-medium.south~east) $
      )
      {
        \nullfont
        \c_math_toggle_token
        \left .
        \vcenter { \nullfont \hrule height .5 \l_tmpa_dim depth .5 \l_tmpa_dim width 0 pt }
        \right #4
        \c_math_toggle_token
      } ;
    \end { tikzpicture }
  }

\makeatother
\ExplSyntaxOff
'''
# -----------------------------------------------------------------
preamble = r'''
\newcolumntype{I}{!{\OnlyMainNiceMatrix{\vrule}}}
\SubMatrixOptions{extra-height = 1mm}
'''
# =================================================================
def str_rep_from_mat( A, formater=repr):
    '''str_rep_from_mat( A, formater=repr)
    convert matrix A to a string using formater
    '''
    M,N=A.shape
    return np.array( [[formater(A[i,j]) for j in range(N)] for i in range(M)] )

def str_rep_from_mats( A, b, formater=repr ):
    '''str_rep_from_mats( A, b, formater=repr)
    convert matrix A and vector b to a string using formater, return the augmented matrix
    '''
    sA = str_rep_from_mat(A, formater)
    sb = np.array(b).reshape(-1,1)
    return np.hstack( [sA, sb] )
# =================================================================
def tex_from_mat( A, front=0, back=0, formater=repr):
    '''print latex representation of array A, with "front" and "back" empty slots'''
    M,N=A.shape
    tf = ' '.join([ ' & ' for _ in range(front)])
    tb = ' '.join([ ' & ' for _ in range(back)])
    tbe = tb + ' \\\\ \n'

    t  = tbe.join( [ tf + ' & '.join( [ formater(x) for x in A[i,:] ]  ) for i in range(M)] ) + tb
    return t

# -----------------------------------------------------------------
def path_format(i,j):
    return "(row-{i}-|col-{j})".format(i=i,j=j),"(row-{i}-|col-{j})".format(i=i+1,j=j)
# -----------------------------------------------------------------
def loc_format(x,parens=('(', ')') ):
    '''loc_format(x,parens=('(', ')') )
    location format of a node (i-j) or {i-j}
    '''
    return '{s}{i}-{j}{e}'.format(s=parens[0],i=x[0],j=x[1], e=parens[1])

# -----------------------------------------------------------------
# This is not correct if a third or further layer has a different number of rows!
def submatrix_locations(layers, which_layers=None, row_offset=1, col_offset=1, rep=lambda x,y:loc_format((x,y),('{','}'))):
    '''submatrix_locations(n_layers, shape, offset=1, start_at_layer=0, rep=lambda x,y:loc_format((x,y),('{','}')))
    layers:         vertical stack of matrices
    which_layers:   iterable over layers. (default = all)
    shape:          (M,N) of each matrix
    row_offset:     positions to skip from the top
    col_offset:     positions to skip from the left
    start_at_layer: first layer for which to add a submatrix
    rep:            formater for a corner of the submatrix
    '''

    row_sizes = np.array([0,layers[0][1].shape[0]] + [layer[0].shape[0] for layer in layers[1:]])
    col_sizes = np.array([0]+[ m.shape[1] for m in layers[1]])

    row_start = np.cumsum( row_sizes )
    col_start = np.cumsum( col_sizes )

    num_matrices_in_layer = len( layers[1])
    if which_layers is None: which_layers = range(0, len(layers))

    sub_matrices = []
    for lc in range(num_matrices_in_layer):
        for lr in which_layers:
            sub_matrices.append( rep(row_start[lr]+row_offset,col_start[lc]+col_offset)+
                                 rep(row_start[lr+1]+row_offset-1,col_start[lc+1]+col_offset-1) )

    return sub_matrices[1:]
# -----------------------------------------------------------------
def old_submatrix_locations(n_layers, shape, row_offset=1, col_offset=1, start_at_layer=0, rep=lambda x,y:loc_format((x,y),('{','}'))):
    '''old_submatrix_locations(n_layers, shape, offset=1, start_at_layer=0, rep=lambda x,y:loc_format((x,y),('{','}')))
    n_layers:       number of layers
    shape:          (M,N) of each matrix
    row_offset:     positions to skip from the top
    col_offset:     positions to skip from the left
    start_at_layer: first layer for which to add a submatrix
    rep:            formater for a corner of the submatrix
    '''
    M,N = shape
    return [(rep(row_offset+i*M,col_offset)+rep(row_offset+(i+1)*M-1,col_offset+N-1)) for i in range(start_at_layer,n_layers)]

# -----------------------------------------------------------------
def one_pivot_locations( loc, n_layers=1, M=0, row_offset=0, col_offset=0, c=("blue","red"), rep=lambda x,y:loc_format((x,y)) ):
    ''' add pivots in the vertical stack underneath the current pivot
    loc:         are the indices of the first location, colored c[1]
    n_layers:    is the number of matrix layers in the stack
    M:           is the number of rows of a matrix in the stack
    '''
    i,j = loc
    i   = i + row_offset
    j   = j + col_offset

    p = [(rep(i,j),c[1])]
    for l in range(1,n_layers):
        p.append( (rep(l*M+i,j), c[0]))

    return p

# -----------------------------------------------------------------
def path_locations( locs, n_layers=1, M=0, N=0, row_offset=0, col_offset=0, rep=path_format ):
    '''construct all actual pivot locations in a matrix stack'''
    p = []
    l = len(locs)-1
    for i,loc in enumerate(locs):
        if loc is not None:
            i,j = loc
            i   = l*M+i + row_offset
            j   = j + col_offset

            p_add = rep(i,j)
            p.extend( p_add )

    p.append( rep(n_layers*M+1, N+col_offset)[0] )
    return p
# -----------------------------------------------------------------
def pivot_locations( locs, n_layers=1, M=0, row_offset=0, col_offset=0, c=("blue", "red"), rep=lambda x,y:loc_format((x,y)) ):
    '''construct all actual pivot locations in a matrix stack'''
    p = []
    for i,loc in enumerate(locs):
        if loc is not None:
            p = p + ( one_pivot_locations((loc[0]+i*M,loc[1]), n_layers-i, M, row_offset, col_offset, c, rep))
    return p
# -----------------------------------------------------------------
def ge_int_layout( layer_defs, Nrhs=0, pivots=None, txt=None, decorate=False ):
    n_layers = len(layer_defs)
    Ma,Na    = layer_defs[0][1].shape   # matrix sizes
    Me,Ne    = Ma,Ma                    #

    sep            = '% --------------------------------------------\n'
    mat_rep        = sep + tex_from_mat( layer_defs[0][1], front=Me, back=1, formater=(lambda x: f'{x:.0f}'))
    #submatrix_locs = old_submatrix_locations( n_layers, (Me,Ne), row_offset=1, col_offset=1, start_at_layer=1)

    for layer in layer_defs[1:]:
        mat_rep        += ' \\\\ \\noalign{\\vskip2mm} \n % ---------------------------------------------\n' \
                 + tex_from_mat( np.hstack(layer), back=1, formater=lambda x: f'{x:.0f}')
        #submatrix_locs += old_submatrix_locations(n_layers, (Ma,Na), row_offset=1, col_offset=1+Ne, start_at_layer=0)
    submatrix_locs = submatrix_locations(layer_defs, row_offset=1, col_offset=1)

    if pivots is not None and not decorate:
        pivot_locs   = pivot_locations(pivots, n_layers, M=Ma, row_offset=0, col_offset=Ne)
    else:
        pivot_locs = []

    if decorate:
        path_corners = path_locations(pivots, n_layers, M=Ma, N=Na, row_offset=0, col_offset=Ne)
    else:
        path_corners = []

    if txt is not None:
        txt_with_locs = [ (f'({1+i*Ma}-{Ne+Na}.east)','\\quad '+t) for i,t in enumerate(txt) ]
    else:
        txt_with_locs = []

    if Nrhs > 0:
        Na1 = Na - Nrhs
        mat_format = f'{{*{Ne}r@{{\qquad\ }}*{Na1}rI*{Nrhs}r@{{\qquad\;\;}}r}}'
    else:
        mat_format = f'{{*{Ne}r@{{\qquad\ }}*{Na}rr@{{\qquad\;\;}}r}}'

    return mat_rep, submatrix_locs, pivot_locs, path_corners, txt_with_locs,mat_format

# -----------------------------------------------------------------
def convert_layer_defs( layer_defs ):
    '''convert layer_defs to [[None,A],[],[],[]]'''
    n_layers = len(layer_defs)
    A = np.array( layer_defs[0][1] )
    return [[None, A]] + [np.hstack(layer) for layer in layer_defs[1:]]

# -----------------------------------------------------------------
def ge_layout_from_stacked( layer_defs, Nrhs=0, pivots=None, txt=None, decorate=False, formater=repr ):
    n_layers = len(layer_defs)
    A = np.array( layer_defs[0][1] )

    Ma,Na    = A.shape                  # matrix sizes
    Me,Ne    = Ma,Ma                    #

    sep            = '% --------------------------------------------\n'
    mat_rep        = sep + tex_from_mat( A, front=Me, back=1, formater=formater)

    #submatrix_locs = old_submatrix_locations( n_layers, (Me,Ne), row_offset=1, col_offset=1, start_at_layer=1)
    submatrix_locs = submatrix_locations(layer_defs, which_layers=range(1,n_layers), row_offset=1, col_offset=1)

    for layer in layer_defs[1:]:
        M = np.array(layer)
        mat_rep  += ' \\\\ \\noalign{\\vskip2mm} \n % ---------------------------------------------\n' \
                 + tex_from_mat( M, back=1, formater=formater)
        #submatrix_locs += old_submatrix_locations(n_layers, (Ma,Na), row_offset=1, col_offset=1+Ne, start_at_layer=0)

    if pivots is not None and not decorate:
        pivot_locs   = pivot_locations(pivots, n_layers, M=Ma, row_offset=0, col_offset=Ne)
    else:
        pivot_locs = []

    if decorate:
        path_corners = path_locations(pivots, n_layers, M=Ma, N=Na, row_offset=0, col_offset=Ne)
    else:
        path_corners = []

    if txt is not None:
        txt_with_locs = [ (f'({1+i*Ma}-{Ne+Na}.east)','\\quad '+t) for i,t in enumerate(txt) ]
    else:
        txt_with_locs = []

    if Nrhs > 0:
        Na1 = Na - Nrhs
        mat_format = f'{{*{Ne}r@{{\qquad\ }}*{Na1}rI*{Nrhs}r@{{\qquad\;\;}}r}}'
    else:
        mat_format = f'{{*{Ne}r@{{\qquad\ }}*{Na}r@{{\qquad\;\;}}r}}'

    return mat_rep, submatrix_locs, pivot_locs, path_corners, txt_with_locs,mat_format
# -----------------------------------------------------------------
# Needs debugging
def new_ge_layout( layer_defs, Nrhs=0, pivots=None, txt=None, decorate=False, formater=repr ):
    '''generate a ge_layout with a certain number of rhs'''
    defs = convert_layer_defs( layer_defs )
    return ge_layout_from_stacked( defs, Nrhs=Nrhs, pivots=pivots, txt=txt, decorate=decorate, formater=formater )

# -----------------------------------------------------------------
def ge_layout( layer_defs, Nrhs=0, pivots=None, txt=None, decorate=False, formater=repr ):
    n_layers = len(layer_defs)
    Ma,Na    = layer_defs[0][1].shape   # matrix sizes
    Me,Ne    = Ma,Ma                    #

    sep            = '% --------------------------------------------\n'
    mat_rep        = sep + tex_from_mat( layer_defs[0][1], front=Me, back=1, formater=formater)
    #submatrix_locs = old_submatrix_locations( n_layers, (Me,Ne), row_offset=1, col_offset=1, start_at_layer=1)

    for layer in layer_defs[1:]:
        mat_rep        += ' \\\\ \\noalign{\\vskip2mm} \n % ---------------------------------------------\n' \
                 + tex_from_mat( np.hstack(layer), back=1, formater=formater)
        #submatrix_locs += old_submatrix_locations(n_layers, (Ma,Na), row_offset=1, col_offset=1+Ne, start_at_layer=0)

    submatrix_locs = submatrix_locations(layer_defs,row_offset=1, col_offset=1)


    if pivots is not None and not decorate:
        pivot_locs   = pivot_locations(pivots, n_layers, M=Ma, row_offset=0, col_offset=Ne)
    else:
        pivot_locs = []

    if decorate:
        path_corners = path_locations(pivots, n_layers, M=Ma, N=Na, row_offset=0, col_offset=Ne)
    else:
        path_corners = []

    if txt is not None:
        txt_with_locs = [ (f'({1+i*Ma}-{Ne+Na}.east)','\\quad '+t) for i,t in enumerate(txt) ]
    else:
        txt_with_locs = []

    if Nrhs > 0:
        Na1 = Na - Nrhs
        mat_format = f'{{*{Ne}r@{{\qquad\ }}*{Na1}rI*{Nrhs}r@{{\qquad\;\;}}r}}'
    else:
        mat_format = f'{{*{Ne}r@{{\qquad\ }}*{Na}r@{{\qquad\;\;}}r}}'

    return mat_rep, submatrix_locs, pivot_locs, path_corners, txt_with_locs,mat_format

# =================================================================
GE_TEMPLATE = r'''
\documentclass[notitlepage]{article}
%\pagenumbering{gobble}
\pagestyle{empty}

%\documentclass{standalone}
%\usepackage{standalone}

%\usepackage[french]{babel}
\usepackage{mathtools}
\usepackage{xltxtra}
%\usepackage{xcolor}
\usepackage{nicematrix,tikz}
\usetikzlibrary{calc,fit}

{{extension}}
\begin{document}
{{preamble}}
% ================================================================================
$\begin{NiceArray}[create-medium-nodes]{{mat_format}}{{mat_options}}
{{mat_rep}}
\CodeAfter
% ----------------------------------------- submatrix delimiters
  \SubMatrixOptions{right-xshift=2mm, left-xshift=2mm}
    {% for loc in submatrix_locs: -%}
          \SubMatrix({{loc}})
    {% endfor -%}
% ----------------------------------------- pivot outlines
\begin{tikzpicture}
    \begin{scope}[every node/.style = draw]
    {% for loc in pivot_locs: -%}
        \node [draw,{{loc[1]}},fit = {{loc[0]}}]  {} ;
    {% endfor -%}
    \end{scope}

% ----------------------------------------- explanatory text
    {% for loc,txt in txt_with_locs: -%}
        \node [right,align=left] at {{loc}}  {\qquad {{txt}} } ;
    {% endfor -%}

%\node [right,align=left] at (14-8.east) {\quad There are no free variables.\\
%                                        \quad We have obtained a unique solution.} ;

% ----------------------------------------- row echelon form path

\end{tikzpicture}
\end{NiceArray}$

\end{document}
'''
# -----------------------------------------------------------------
def qr_layout(A_,W_,formater=sym.latex):
    A=sym.Matrix(A_);W=sym.Matrix(W_)
    WtW  = W.T @ W
    WtA  = W.T @ A
    S    = WtW**(-1)
    for i in range(S.shape[0]):
        S[i,i]=sym.sqrt(S[i,i])

    Qt = S*W.T
    R  = S*WtA

    def mk_nones( l ):
        return '&'.join( np.repeat(" ",l+1))+"  "
    extra = " & "

    cS = 1; cWt = cS+S.shape[1]; cA=cWt+A.shape[0]; cW=cA+A.shape[1]; cEnd=cW+W.shape[1]
    r1 = 1; r2  = r1+A.shape[0]; r3= r2+W.shape[1]; rEnd=r3+W.shape[1]

    mat_fmt = f'{{*{S.shape[1]}r@{{\qquad\ }}*{A.shape[0]}r@{{\qquad\ }}*{A.shape[1]}rI*{W.shape[1]}r@{{\qquad\;\;}}r}}'

    #sep = "% -----------------------------------------------------------------------------\n"
    #display(sym.BlockMatrix([A,W]))
    def mk_l1():
        l1_nones = mk_nones(S.shape[1]+W.shape[0])
        s = []
        for i in range(A.shape[0]):
            s.append( l1_nones + " & ".join( map(formater, A[i,:]) ) + " &   "
                               + " & ".join( map(formater, W[i,:]) ) + extra )
        startA = S.shape[1]+A.shape[0]+1; endA=A.shape[0]
        submatrix_locs = [f'{{{r1}-{cA}}}{{{r2-1}-{cEnd-1}}}']
        return s, submatrix_locs

    #display(sym.BlockMatrix([W.T,WtA,WtW]))
    def mk_l2():
        l2_nones = mk_nones(S.shape[1])
        s = []
        for i in range(W.shape[1]):
            s.append( l2_nones + " & ".join( map(formater, W[:,i] )  )  + " &   "
                               + " & ".join( map(formater, WtA[i,:]) )  + " &   "
                               + " & ".join( map(formater, WtW[i,:]) ) + extra
            )
        submatrix_locs = [f'{{{r2}-{cWt}}}{{{r3-1}-{cA-1}}}',
                          f'{{{r2}-{cA}}}{{{r3-1}-{cEnd-1}}}'
                         ]
        return s, submatrix_locs

    #display(sym.BlockMatrix([S,Qt,R]))
    def mk_l3():
        l3_nones = mk_nones(W.shape[1])
        s = []
        for i in range(S.shape[0]):
            s.append( " & ".join( map(formater, S[i,:] )  )  + " &   "
                    + " & ".join( map(formater, Qt[i,:]) )  + " &   "
                    + " & ".join( map(formater, R[i,:]) )
                    + l3_nones + extra
            )
        submatrix_locs = [f'{{{r3}-{cS}}}{{{rEnd-1}-{cWt-1}}}',
                          f'{{{r3}-{cWt}}}{{{rEnd-1}-{cA-1}}}',
                          f'{{{r3}-{cA}}}{{{rEnd-1}-{cEnd-1}}}'
                         ]
        return s,submatrix_locs

    layer_1, submatrix_locs_1 = mk_l1()
    layer_2, submatrix_locs_2 = mk_l2()
    layer_3, submatrix_locs_3 = mk_l3()

    layers = [layer_1, layer_2, layer_3]

    s = []
    for l in layers:
        s.append( " \\\\ \n".join(l))

    return " \\\\  \\noalign{\\vskip2mm} \n".join(s),\
           mat_fmt,\
           submatrix_locs_1+submatrix_locs_2+submatrix_locs_3

# ========================================================================================
# NEW: replacement for the previous formating functions
# ========================================================================================
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

        self.mat_row_height = [ max(map( lambda s: s[0], self.array_shape[i, :])) for i in range(self.nGridRows)]
        self.mat_col_width  = [ max(map( lambda s: s[1], self.array_shape[:, j])) for j in range(self.nGridCols)]

        self.adjust_positions( extra_cols, extra_rows )
        self.txt_with_locs    = []

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
        #A_shape   = (self.matrices[gM][gN]).shape
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

    def tex_repr( self, blockseps = r'\noalign{\vskip2mm} '):
        '''Create a list of strings, one for each line in the grid'''
        self.tex_list =[' & '.join( self.a_tex[k,:]) for k in range(self.a_tex.shape[0])]
        for i in range( len(self.tex_list) -1):
            self.tex_list[i] += r' \\'

        sep = ' ' + blockseps
        for i in (self.cs_mat_row_height[1:-1] + self.cs_extra_rows[1:-1] - self.extra_rows[1:-1]):
            self.tex_list[i-1] += sep

        if self.extra_rows[-1] != 0: # if there are final extra rows, we need another sep
            self.tex_list[ self.tex_shape[0] - self.extra_rows[-1] - 1] += sep

    def array_of_tex_entries(self, formater=repr):
        '''Create a matrix of strings from the grid entries'''

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
        '''apply decorate to the list of i,j entries to grid matrix at (gM,gN)'''
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

    def array_format_string_list( self, partitions={}, spacer_string=r'@{\qquad\ }', p_str='I', last_col_format = "l@{\qquad\;\;}") :
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
                    fmt += spacer_string + (l-1)*'r'+last_col_format
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

    def add_row_above( self, gM, gN, m, formater=repr, offset=0 ):
        '''add tex entries to the tex array'''
        tl,shape = self.tl_shape_above( gM, gN )
        for (j,v) in enumerate(m):
            self.a_tex[tl[0]+offset, tl[1]+j] = formater( v )

    def add_row_below( self, gM, gN, m, formater=repr, offset=0 ):
        '''add tex entries to the tex array'''
        tl,shape = self.tl_shape_below( gM, gN )
        for (j,v) in enumerate(m):
            self.a_tex[tl[0]+offset, tl[1]+j] = formater( v )

    def add_col_right( self, gM, gN, m, formater=repr, offset=0 ):
        '''add tex entries to the tex array'''
        tl,shape = self.tl_shape_right( gM, gN )
        for (i,v) in enumerate(m):
            self.a_tex[tl[0]+i, tl[1]+offset] = formater( v )

    def add_col_left( self, gM, gN, m, formater=repr, offset=0 ):
        '''add tex entries to the tex array'''
        tl,shape = self.tl_shape_left( gM, gN )
        for (i,v) in enumerate(m):
            self.a_tex[tl[0]+i, tl[1]+offset-1] = formater( v )

    #def add_row_echelon_path( self, gM, gN, pivot_locs, tikz_opts='[dashed,red]' ):
    #    '''This would work, but produces an unsatisfactory result'''
    #    l = [tikz_opts]
    #    for p in pivot_locs:
    #        i,j = self.element_indices( *p, gM, gN ); i+=1; j+= 1
    #        l.extend([ f'(row-{i}-|col-{j})', f'(row-{i+1}-|col-{j})'])
    #    i+=1; j+= 1
    #    l.append( f'(row-{i}-|col-{j})')
    #    self.row_echelon_paths.append(l)

    def nm_submatrix_locs(self):
        '''nicematrix style location descriptors of the submatrices'''
        locs = []
        for i in range(self.nGridRows):
            for j in range(self.nGridCols):
                if self.array_shape[i,j][0] != 0:
                    tl,br,_ = self._top_left_bottom_right(i,j)
                    locs.append( f"{{{tl[0]+1}-{tl[1]+1}}}{{{br[0]+1}-{br[1]+1}}}" )

        self.locs = locs

    def nm_text(self, txt_list):
        '''add text add each layer (requires a right-most extra col)'''
        assert( self.extra_cols[-1] != 0 )

        # find the indices of the first row in the last col (+1 for nicematrix indexing)
        txt_with_locs = []
        for (g,txt) in enumerate(txt_list):
            #A_shape   = (self.matrices[g][self.nGridCols-1]).shape
            A_shape   = self.array_shape[g][self.nGridCols-1]

            first_row = self.cs_mat_row_height[g] + self.cs_extra_rows[g] + (self.mat_row_height[g] - A_shape[0])+1
            txt_with_locs.append(( f'({first_row}-{self.tex_shape[1]-1}.east)', txt) )
        self.txt_with_locs = txt_with_locs

    def nm_latexdoc( self, template = GE_TEMPLATE, preamble = preamble, extension = extension ):
        return jinja2.Template( template ).render( \
                preamble       = preamble,
                extension      = extension,
                mat_rep        = '\n'.join( self.tex_list ),
                mat_format     = '{'+self.format+'}',
                mat_options    = '',
                submatrix_locs = self.locs,
                pivot_locs     = [],
                txt_with_locs  = self.txt_with_locs)
# -----------------------------------------------------------------------------------------------------
def make_decorator( text_color='black', text_bg=None, boxed=None, bf=None, move_right=False ):
    box_decorator         = "\\boxed<{a}>"
    color_decorator       = "\\Block[draw={text_color},fill={bg_color}]<><{a}>"
    txt_color_decorator   = "\\color<{color}><{a}>"
    bf_decorator          = "\\mathbf<{a}>"
    rlap_decorator        = "\\mathrlap<{a}>"

    x = '{a}'
    if bf is not None:
        x = bf_decorator.format(a=x)
    if boxed is not None:
        x = box_decorator.format( a=x )
    if text_bg is not None:
        x = color_decorator.format(a=x, text_color= text_color, bg_color=text_bg)
    elif text_color != 'black':
        x = txt_color_decorator.format( color=text_color, a=x)
    if move_right:
        x = rlap_decorator.format(a=x)

    x = x.replace('<','{{').replace('>','}}')

    return lambda a: x.format(a=a)

# -----------------------------------------------------------------------------------------------------
def ge( matrices, Nrhs=0, formater=repr, pivot_list=None, comment_list=None, variable_summary=None, tmp_dir=None, keep_file=None):
    '''basic GE layout:
    matrices:         [ [None, A0], [E1, A1], [E2, A2], ... ]
    Nrhs:             number of right hand side columns determines the placement of a partition line, if any
    pivot_list:       [ pivot_spec, pivot_spec, ... ] where pivot_spec = [grid_pos, [pivot_pos, pivot_pos, ...]]
    comment_list:     [ txt, txt, ... ] must have a txt entry for each layer. Multiline comments are separated by \\
    variable_summary: [ basic, ... ]  a list of true/false values specifying whether a column has a pivto or not
    '''
    extra_cols = None if comment_list     is None else 1
    extra_rows = None if variable_summary is None else 2

    m = MatrixGridLayout(matrices, extra_rows=extra_rows, extra_cols = extra_cols )

    # compute the format spec for the arrays and set up the entries (defaults to a single partition line)
    partitions = {} if Nrhs == 0 else { 1: [m.mat_col_width[-1]-Nrhs]}
    m.array_format_string_list( partitions=partitions )
    m.array_of_tex_entries(formater=formater)   # could overwride the entry to TeX string conversion here

    if pivot_list is not None:
        red_box = make_decorator( text_color='red', boxed=True, bf=True )
        for spec in pivot_list:
            m.decorate_tex_entries( *spec[0], red_box, entries=spec[1] )

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
                var.append( red( f'x_{i+1}'))
            else:
                typ.append(blue(r'\uparrow'))
                var.append(blue( f'x_{i+1}'))
        m.add_row_below(m.nGridRows-1,1,typ,           formater=lambda a: a )
        m.add_row_below(m.nGridRows-1,1,var, offset=1, formater=lambda a: a )

    m.nm_submatrix_locs()     # this defines the submatrices (the matrix delimiters)
    m.tex_repr()              # converts the array of TeX entries into strings with separators and spacers

    m_code = m.nm_latexdoc(template = GE_TEMPLATE, preamble = preamble, extension = extension )

    h = itikz.fetch_or_compile_svg(
        m_code, prefix='ge_', working_dir=tmp_dir, debug=False,
        **itikz.build_commands_dict(use_xetex=True,use_dvi=False,crop=True),
        nexec=1, keep_file=keep_file )

    return h, m

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
# m3_code = m3.nm_latexdoc(template = nM.GE_TEMPLATE, preamble = nM.preamble, extension = nM.extension )
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
