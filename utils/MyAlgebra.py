"""
    Utility file for the algebraic equation needed and not natively supported in ufl API
"""

import ufl


perm = ufl.PermutationSymbol(3)

def tcurl(A):
    """
        Compute the curl of 3x3 tensor field A as:

        :math:`(\\nabla \\times A)_{ij} = A{il,k}\epsilon_{jkl}`
    """
    i, j, l = ufl.indices(3)
    return ufl.as_tensor(ufl.Dx(A[i,l],0)*perm[j,0,l]+ufl.Dx(A[i,l],1)*perm[j,1,l],
    (i, j)
)

def tcurl2d(A,shape):
    """
    for a 2d tensor field A(x,y), its curls is full of 0 expect the components 13 and 23
    returns a vector containing these 2 elements 
    
    Then tensor A can be fed as a vector (when using mpc) in this case shape=v or as a tensor when shape=t 
    """
    if shape=="t":
        _13 =ufl.Dx(A[0,1],0)-ufl.Dx(A[0,0],1)
        _23=ufl.Dx(A[1,1],0)-ufl.Dx(A[1,0],1)
    elif shape=="v":
        _13 =ufl.Dx(A[1],0)-ufl.Dx(A[0],1)
        _23=ufl.Dx(A[3],0)-ufl.Dx(A[2],1)
    return ufl.as_vector([_13,_23])


def tcrossv(A,v):
    """
        Compute the cross product of a 3x3 tensor field A with a 3d vector v

        :math:`(\A \\times v)_{ij} = A{ik}v_l\epsilon_{jkl}`
    """
    i, l,j = ufl.indices(3)
    return ufl.as_tensor(A[i,l]*v[0]*perm[l,0,j]+A[i,l]*v[1]*perm[l,1,j]+A[i,l]*v[2]*perm[l,2,j],
    (i, j)
)

def tdiv(A):
    """
        Given a 2x2 tensor A, compute its divergence as a 3d vector by appending a 0 to it
        Returns a 3d vector
    """
    return ufl.as_vector([
        A[0,0].dx(0)+A[0,1].dx(1),
        A[1,0].dx(0)+A[1,1].dx(1),
        0
    ]
)
def vdot(A,b):
    """
        Computes the dot product of a 2d tensor  with a 2d  vector 
        Returns a 3d vector

    """

    return ufl.as_vector([
        A[0,0]*b[0]+A[0,1]*b[1],
        A[1,0]*b[0]+A[1,1]*b[1],
        0.0
    ])

def vgrad(u):
    """
        COmpute the gradient of a 2x2 tensor field as a 3x3 tensor by appending 0 to it
    """

    return ufl.as_tensor([
         [u[0].dx(0),u[0].dx(1),0],
         [u[1].dx(0),u[1].dx(1),0],
         [0,0,0]
    ])



#TODO Add Condition on UFL shape before extending and restricting

def restrictT(A):
    """
        Given a 3x3 tensor, extracts the upper left 2x2 block to build a 2d tensor
    """
    return ufl.as_tensor([
        [A[0,0],A[0,1]],
        [A[1,0],A[1,1]]
    ])

def extendT(A):
    """
        Given a 2x2 tensor, appends a 0 row and colum to build a 3x3 tensor
    """
    return ufl.as_tensor([
        [A[0,0],A[0,1],0],
        [A[1,0],A[1,1],0],
        [0,0,0]
    ])



def extendV(A):
    """
        given a 2d vector, extend it to 3d by adding a 0 entry
    """
    return ufl.as_vector([A[0],A[1],0])


def retV(A):
    """
        given a 3d vector, restrit it to 2d by dropping the last entry
    """
    return ufl.as_vector([A[0],A[1]])

def T2v(A,d):
    """
        Flatten a tensor into a vector 

        ..math:

        \begin{bmatrix} a_{11} & a_{12} & a_{13} \\
        a_{21} & a_{22} & a_{23} \\
        a_{31} & a_{32} & a_{33} \end{bmatrix} \quad \Righarrow \quad \left[  a_{11} \ a_{12} \ a_{13} \ a_{21} & a_{22} & a_{23} \ a_{31} & a_{32} & a_{33} \right]
    """
    return ufl.as_vector([A[i, j] for i in range(d) for j in range(d)])

def v2T(v,d):
    """ 
        DOes the invert operation of T2v.

        Take as a flat vector and build a tensor of the right shape of it
    """
    return ufl.as_tensor([[v[d*i + j] for j in range(d)] for i in range(d)])

def restrictV(a):
    return ufl.as_tensor([
        a[0],
        a[1]
    ])

def epsilon(u):
   """
    strain tensor of a given vector field
   """
   return ufl.sym(ufl.grad(u))

def sigma(eps,lambda_,mu_):
    """
        stress tensor from a given strain tensor using lamé coefficients
    """
    return  lambda_ * ufl.tr(eps) * ufl.Identity(2)+ 2 * mu_ *eps  #TODO the hardcoded 3 should be dynamic # 

def strain(sig,lambda_,mu_):
    """
        Strain tensor from a given stress tensor using lamé coefficients
    """
    return (1/(2*mu_))*(sig - (lambda_/(3*lambda_+2*mu_))*ufl.tr(sig)*ufl.Identity(2))