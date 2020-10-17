import numpy as np

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
def loc_format(x,parens=('(', ')') ):
    '''loc_format(x,parens=('(', ')') )
    location format of a node (i-j) or {i-j}
    '''
    return '{s}{i}-{j}{e}'.format(s=parens[0],i=x[0],j=x[1], e=parens[1])

# -----------------------------------------------------------------
def submatrix_locations(n_layers, shape, row_offset=1, col_offset=1, start_at_layer=0, rep=lambda x,y:loc_format((x,y),('{','}'))):
    '''submatrix_locations(n_layers, shape, offset=1, start_at_layer=0, rep=lambda x,y:loc_format((x,y),('{','}')))
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
def pivot_locations( locs, n_layers=1, M=0, row_offset=0, col_offset=0, c=("blue", "red"), rep=lambda x,y:loc_format((x,y)) ):
    '''construct all actual pivot locations in a matrix stack'''
    p = []
    for i,loc in enumerate(locs):
        if loc is not None:
            p =p + ( one_pivot_locations((loc[0]+i*M,loc[1]), n_layers-i, M, row_offset, col_offset, c, rep))
    return p
# -----------------------------------------------------------------
def ge_int_layout( layer_defs, Nrhs=0, pivots=None, txt=None, decorate=False ):
    n_layers = len(layer_defs)
    Ma,Na    = layer_defs[0][1].shape   # matrix sizes
    Me,Ne    = Ma,Ma                    #

    sep            = '% --------------------------------------------\n'
    mat_rep        = sep + tex_from_mat( layer_defs[0][1], front=Me, back=1, formater=(lambda x: f'{x:.0f}'))
    submatrix_locs = submatrix_locations( n_layers, (Me,Ne), row_offset=1, col_offset=1, start_at_layer=1)

    for layer in layer_defs[1:]:
        mat_rep        += ' \\\\ \\noalign{\\vskip2mm} \n % ---------------------------------------------\n' \
                 + tex_from_mat( np.hstack(layer), back=1, formater=lambda x: f'{x:.0f}')
        submatrix_locs += submatrix_locations(n_layers, (Ma,Na), row_offset=1, col_offset=1+Ne, start_at_layer=0)

    if pivots is not None:
        pivot_locs   = pivot_locations(pivots, n_layers, M=Ma, row_offset=0, col_offset=Ne)
    else:
        pivot_locs = []

    if txt is not None:
        txt_with_locs = [ (f'({1+i*Ma}-{Ne+Na}.east)','\\quad '+t) for i,t in enumerate(txt) ]
    else:
        txt_with_locs = []

    if Nrhs > 0:
        Na1 = Na - Nrhs
        mat_format = f'{{*{Ne}r@{{\qquad\ }}*{Na1}rI*{Nrhs}r@{{\qquad\;\;}}r}}'
    else:
        mat_format = f'{{*{Ne}r@{{\qquad\ }}*{Na}rr@{{\qquad\;\;}}r}}'

    return mat_rep, submatrix_locs, pivot_locs, txt_with_locs,mat_format

# -----------------------------------------------------------------
def ge_layout( layer_defs, Nrhs=0, pivots=None, txt=None, decorate=False, formater=lambda x: x ):
    n_layers = len(layer_defs)
    Ma,Na    = layer_defs[0][1].shape   # matrix sizes
    Me,Ne    = Ma,Ma                    #

    sep            = '% --------------------------------------------\n'
    mat_rep        = sep + tex_from_mat( layer_defs[0][1], front=Me, back=1, formater=formater)
    submatrix_locs = submatrix_locations( n_layers, (Me,Ne), row_offset=1, col_offset=1, start_at_layer=1)

    for layer in layer_defs[1:]:
        mat_rep        += ' \\\\ \\noalign{\\vskip2mm} \n % ---------------------------------------------\n' \
                 + tex_from_mat( np.hstack(layer), back=1, formater=formater)
        submatrix_locs += submatrix_locations(n_layers, (Ma,Na), row_offset=1, col_offset=1+Ne, start_at_layer=0)

    if pivots is not None:
        pivot_locs   = pivot_locations(pivots, n_layers, M=Ma, row_offset=0, col_offset=Ne)
    else:
        pivot_locs = []

    if txt is not None:
        txt_with_locs = [ (f'({1+i*Ma}-{Ne+Na}.east)','\\quad '+t) for i,t in enumerate(txt) ]
    else:
        txt_with_locs = []

    if Nrhs > 0:
        Na1 = Na - Nrhs
        mat_format = f'{{*{Ne}r@{{\qquad\ }}*{Na1}rI*{Nrhs}r@{{\qquad\;\;}}r}}'
    else:
        mat_format = f'{{*{Ne}r@{{\qquad\ }}*{Na}r@{{\qquad\;\;}}r}}'

    return mat_rep, submatrix_locs, pivot_locs, txt_with_locs,mat_format

# =================================================================
GE_TEMPLATE = r'''
\documentclass[notitlepage]{article}
%\pagenumbering{gobble}
\pagestyle{empty}

%\documentclass{standalone}
%\usepackage{standalone}

%\usepackage[french]{babel}
\usepackage{xltxtra}
%\usepackage{xcolor}
\usepackage{nicematrix,tikz}
\usetikzlibrary{calc,fit}

{{extension}}

\begin{document}
{{preamble}}

\bigskip

$\begin{NiceArray}[create-medium-nodes]{{mat_format}}
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
        \node [right,align=left] at {{loc}}  {\quad {{txt}} } ;
    {% endfor -%}

%\node [right,align=left] at (2-8.east)  {\quad Augment $A$ with both $b_1$ and $b_2$.\\
%                                         \quad Choose pivot 2.} ;
%\node [right] at (5-8.east)             {\quad Choose the second pivot 1} ;
%\node [right] at (8-8.east)             {\quad Choose the third pivot 3.} ;
%\node [right] at (11-8.east)            {\quad Finally, scale each pivot to 1.} ;
%\node [right,align=left] at (14-8.east) {\quad There are no free variables.\\
%                                        \quad We have obtained a unique solution.} ;

% ----------------------------------------- row echelon form path

\end{tikzpicture}
\end{NiceArray}$

\end{document}
'''
# -----------------------------------------------------------------
#loc_format( (1,2), parens=('{','}') )
#pivot_locations([(1,3),(2,4)], n_layers=4, M=4, row_offset=10, col_offset=0) 
#
#submatrix_locations( 2, (3,5), row_offset=1, col_offset=1+3, start_at_layer=0)
###########################################################
#pivots =[]; n_layers=0
#A  = np.array([[1.,2,1,  9,9],[3,4,5, 9,9], [5,6,1, 9,9]]); pivots.append((1,1)); n_layers=1
#E1 = np.array([[1,0,0],[-3,1,0],[-5, 0,1]]); A1 = E1 @ A;   pivots.append((2,2)); n_layers=2
#E2 = np.array([[1,0,0],[ 0,1,0],[ 0,-2,1]]); A2 = E2 @ A1;  pivots.append((3,3)); n_layers=3
#
#mat_rep, submatrix_locs, pivot_locs = nM.ge_int_layout( [[None, A], [E1, A1], [E2,A2]], pivots)
#
#print("mat_rep ="); print(mat_rep)
#print("pivot_locs =");pivot_locs
#print("submatrix_locs =");submatrix_locs
