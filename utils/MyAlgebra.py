import ufl


perm = ufl.PermutationSymbol(3)

def tcurl(A):
    i, j, l = ufl.indices(3)
    return ufl.as_tensor(ufl.Dx(A[i,l],0)*perm[j,0,l]+ufl.Dx(A[i,l],1)*perm[j,1,l],
    (i, j)
)

def tcurl2d(A,s):
    """
    for a 2d tensor field A(x,y), its curls is full of 0 expect the components 13 and 23
    returns a vector containing these 2 elements 
    
    Then tensor A can be fed as a vector (when using mpc) in this case s=v or as a tensor when s=t 
    """
    if s=="t":
        _13 =ufl.Dx(A[0,1],0)-ufl.Dx(A[0,0],1)
        _23=ufl.Dx(A[1,1],0)-ufl.Dx(A[1,0],1)
    elif s=="v":
        _13 =ufl.Dx(A[1],0)-ufl.Dx(A[0],1)
        _23=ufl.Dx(A[3],0)-ufl.Dx(A[2],1)
    return ufl.as_vector([_13,_23])


def tcrossv(A,v):
    i, l,j = ufl.indices(3)
    return ufl.as_tensor(A[i,l]*v[0]*perm[l,0,j]+A[i,l]*v[1]*perm[l,1,j]+A[i,l]*v[2]*perm[l,2,j],
    (i, j)
)

def tdiv(A):
    # i, j= ufl.indices(2)
    # return ufl.as_tensor(ufl.Dx(A[i,k],l) * perm[l,k,j],(i,j))
    return ufl.as_vector([
        A[0,0].dx(0)+A[0,1].dx(1),
        A[1,0].dx(0)+A[1,1].dx(1),
        0
    ]
)
def vdot(A,b):
    return ufl.as_vector([
        A[0,0]*b[0]+A[0,1]*b[1],
        A[1,0]*b[0]+A[1,1]*b[1],
        0.0
    ])

def vgrad(u):
    return ufl.as_tensor([
         [u[0].dx(0),u[0].dx(1),0],
         [u[1].dx(0),u[1].dx(1),0],
         [0,0,0]
    ])



#TODO Add Condition on UFL shape before extending and restricting

def restrictT(A):
    return ufl.as_tensor([
        [A[0,0],A[0,1]],
        [A[1,0],A[1,1]]
    ])

def extendT(A):
    return ufl.as_tensor([
        [A[0,0],A[0,1],0],
        [A[1,0],A[1,1],0],
        [0,0,0]
    ])



def extendV(A):
    return ufl.as_vector([A[0],A[1],0])


def retV(A):
    return ufl.as_vector([A[0],A[1]])

def T2v(A,d):
    return ufl.as_vector([A[i, j] for i in range(d) for j in range(d)])

def v2T(v,d):
    return ufl.as_tensor([[v[d*i + j] for j in range(d)] for i in range(d)])

def restrictV(a):
    return ufl.as_tensor([
        a[0],
        a[1]
    ])

def epsilon(u):
   return ufl.sym(ufl.grad(u))

def sigma(eps,lambda_,mu_):
    return  lambda_ * ufl.tr(eps) * ufl.Identity(2)+ 2 * mu_ *eps  #TODO the hardcoded 3 should be dynamic # 

def strain(sig,lambda_,mu_):
    return (1/(2*mu_))*(sig - (lambda_/(3*lambda_+2*mu_))*ufl.tr(sig)*ufl.Identity(2))