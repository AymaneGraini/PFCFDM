import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import tree_util
from functools import partial

def Jcompute_gradient(scalar_field, ax, sp):
    return jnp.gradient(scalar_field, sp, axis=ax)

def Jgradient_periodic(array, spacing, axis):
    return jnp.gradient(array, spacing, axis=axis)

# def Jspectral_filter(shape,dx,dy):
#         x = jnp.fft.fftfreq(shape[1],dx)  
#         y = jnp.fft.fftfreq(shape[0],dy) 
#         KX,KY=jnp.meshgrid(x,y)
#         radius_squared = KX**2 + KY**2

#         # g = jnp.exp(-40*(
#         #         (KX*dx)**2+
#         #         (KY*dy)**2
#         #         ))
#         g = jnp.exp(-2 * (jnp.pi**2) * sigma**2 * radius_squared)
#         return g

def JGkernel_jax(shape, dx, dy, sigma):
    wx = jnp.fft.fftfreq(shape[1], dx)  
    wy = jnp.fft.fftfreq(shape[0], dy)  #
    KX, KY = jnp.meshgrid(wx, wy)
    radius_squared = KX**2 + KY**2
    kernel = jnp.exp(-2 * (jnp.pi**2) * sigma**2 * radius_squared)
    return kernel

def levi_civita(i, j, k):
    """Returns the Levi-Civita symbol e_{ijk}."""
    return np.sign(np.linalg.det([[i == 0, j == 0, k == 0],
                                  [i == 1, j == 1, k == 1],
                                  [i == 2, j == 2, k == 2]]))
Levi= np.zeros((3,3,3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            Levi[i,j,k] = levi_civita(i, j, k)




#TODO The class is not clean and it repeats many functions
filterQ = True

class PFCProcessor():
    def __init__(
        self,
        space_coords,
        DofMap,
        qs ,
        a0,
        target_shape : tuple,
        pads : tuple,
    ) -> None:
        
        self.space_coords= space_coords
        self.a0=a0
        self.qs= jnp.array(qs)
        self.DofMap = DofMap
        self.target_shape=target_shape
        self.coords =space_coords[:,:2][DofMap]

        #Build the reverse map
        self.rev_DofMap = jnp.empty_like(self.DofMap)
        self.rev_DofMap = self.rev_DofMap.at[DofMap].set(jnp.arange(len(DofMap)))

        x_coords =jnp.unique(self.coords[:, 0],size= target_shape[1])
        y_coords =jnp.unique(self.coords[:, 1],size= target_shape[0])
        self.X, self.Y = jnp.meshgrid(x_coords, y_coords)
        self.L = self.X.max() - self.X.min()
        self.H = self.Y.max() - self.Y.min()
        self.nx = self.X.shape[1]
        self.ny = self.X.shape[0]
        self.dx = self.L / (self.target_shape[1] - 1) 
        self.dy = self.H / (self.target_shape[0]- 1)  

        self.padx, self.pady = pads

        self.AmpKernel = JGkernel_jax(self.target_shape,self.dx,self.dy,self.a0)
        self.filterQ = True
        kx = jnp.fft.fftfreq(self.nx,self.dx)*2*jnp.pi
        ky = jnp.fft.fftfreq(self.ny,self.dy) *2*jnp.pi
        KX,KY=jnp.meshgrid(kx,ky)
        self.K_vec = jnp.stack((KX, KY), axis=-1)
        self.FF = []  
        for i in range(len(qs)):
            q = self.qs[i]
            f = lambda x, y: jnp.exp(-1j * (q[0] * x + q[1] * y))
            self.FF.append(f(self.X, self.Y))  
        self.FF=jnp.array(self.FF) 
        self.filterQ = True
        
    # @partial(jax.jit, static_argnames=['self'])
    def C_Amp(self,op,i):
        #op is called as a flattened out ndarray coming from dolfinx
        psi = jnp.array(op[self.DofMap].reshape(self.target_shape))
        # padded = np.pad(
        #     psi,
        #     pad_width=((self.padx, self.padx), (self.pady, self.pady)),
        #     mode='wrap'
        # )
        image_fft = jnp.fft.fft2(psi*self.FF[i])
        convolved_fft = image_fft * self.AmpKernel
        convolved = jnp.fft.ifft2(convolved_fft)
        return convolved
        
    # @partial(jax.jit, static_argnames=['self'])
    def Compute_Q(self,amps):
        def compute_single_q(n):
            An = amps[n]
            grad_x = Jcompute_gradient(An, 1, self.dx)
            grad_y = Jcompute_gradient(An, 0, self.dy)
            nabla_An = jnp.zeros((self.ny, self.nx, 2), dtype=jnp.complex64).at[:, :, 0].set(grad_x / An).at[:, :, 1].set(grad_y / An)
            holder = jnp.imag(nabla_An)
            q_field = jnp.broadcast_to(self.qs[n], (self.ny, self.nx, 2))
            return -(2 / 6) * jnp.einsum('ijk,ijl->ijkl', q_field, holder, optimize=True)
        vecs = np.arange(0,len(self.qs),1)
        Qcomp = jnp.sum(jax.vmap(compute_single_q)(vecs), axis=0)
        if not self.filterQ:
            return Qcomp.reshape(-1,4)[self.rev_DofMap].ravel()
        else:
            sp = JGkernel_jax(self.target_shape,self.dx,self.dy,self.a0/4)
            fft_Q = jnp.fft.fft2(Qcomp, axes=(0, 1))
            filtered_fft_Q = jnp.einsum('xyij,xy->xyij', fft_Q, sp, optimize=True)
            convQ = jnp.fft.ifft2(filtered_fft_Q, axes=(0, 1)).real
            return convQ.reshape(-1,4)[self.rev_DofMap].ravel()


    # @partial(jax.jit, static_argnames=['self'])
    def Compute_alpha(self,Qt):
        #Qt is called as a flattened out ndarray coming from dolfinx
        Qcomp= Qt.reshape(-1,4)[self.DofMap].reshape(self.ny,self.nx,2,2)
        Nx, Ny = Qcomp.shape[1], Qcomp.shape[0]
        curl = jnp.zeros((Nx, Ny,3,3))
        grad_x = Jgradient_periodic(Qcomp, self.dy, 1)
        grad_y = Jgradient_periodic(Qcomp, self.dx , 0)
        derivative_array = jnp.zeros((self.ny, self.nx,2,2,2)).at[:,:,0,:,:].set(grad_x).at[:,:,1,:,:].set(grad_y)
        padded_array = jnp.pad(derivative_array, ((0, 0), (0, 0), (0, 1), (0, 1), (0, 1)), mode='constant', constant_values=0)
        curl = jnp.einsum('jkl,abkil->abij', Levi,padded_array,optimize=True)
        return -curl.reshape(-1,9)[self.rev_DofMap].ravel()

    # @partial(jax.jit, static_argnames=['self'])
    def Analytical_gradFuq(self,Qt,Ut,amps):
        Q= Qt.reshape(-1,4)[self.DofMap].reshape(self.ny,self.nx,2,2)
        U= Ut.reshape(-1,4)[self.DofMap].reshape(self.ny,self.nx,2,2)
        Diff = Q-U
        def Single_vec_con(n):
            q=self.qs[n]
            An = amps[n]
            grad_x = Jcompute_gradient(An, 1, self.dx)
            grad_y = Jcompute_gradient(An, 0, self.dy)
            A_term = jnp.zeros((self.ny, self.nx, 2), dtype=jnp.complex64).at[:, :, 0].set((grad_x / (An)**2)).at[:, :, 1].set((grad_y / (An)**2))
            gradF_dAm=jnp.zeros((self.ny, self.nx))
            for i in range(2):
                for j in range(2):
                    gradF_dAm += q[i]*self.FF[n]*jnp.fft.ifft2((jnp.fft.fft2(Diff[:,:,i,j]*A_term[:,:,j])+1j*self.K_vec[:,:,j]*jnp.fft.fft2(Diff[:,:,i,j]*1/An))*self.AmpKernel)
            return  gradF_dAm
        vecs = np.arange(0,len(self.qs),1)
        TA= jnp.sum(jax.vmap(Single_vec_con)(vecs), axis=0)
        return -(2/6)*TA.imag.ravel()[self.rev_DofMap]


    def Coarse_grain(self,R):
        """
        Retreives a coarsegrained version or the tensor R
        """
        sp = JGkernel_jax(self.target_shape,self.dx,self.dy,self.a0*0.75)
        Rcomp= R.reshape(-1,4)[self.DofMap].reshape(self.ny,self.nx,2,2)
        fft_R = jnp.fft.fft2(Rcomp, axes=(0, 1))
        filtered_fft_R = jnp.einsum('xyij,xy->xyij', fft_R, sp, optimize=True)
        convR = jnp.fft.ifft2(filtered_fft_R, axes=(0, 1)).real
        return convR.reshape(-1,4)[self.rev_DofMap].ravel()
    
def Compute_Q_jax(FE_psi,proc):
    """
        Computes Q based on psi using Jax:
        FE_psi is un unordered array coming for dolfinx. It's reordred into psi
        Returns a flattened out array in the same order as that of FE.
    """
    psi = jnp.array(FE_psi[proc.DofMap].reshape(proc.target_shape))
    # Convert qs to a JAX array outside the inner function
    qs_array = jnp.array(proc.qs)
    
    def compute_single_q(n):
        q = qs_array[n]  

        # Exponential modulation
        f = lambda x, y: jnp.exp(-1j * (q[0] * x + q[1] * y))
        FF = f(proc.X, proc.Y)

        # Forward FFT
        image_fft = jnp.fft.fft2(psi * FF)

        # Convolve in Fourier domain
        convolved_fft = image_fft * proc.AmpKernel

        # Inverse FFT to get convolved image
        An = jnp.fft.ifft2(convolved_fft)

        # Approximate gradient using finite differences manually, since jnp.gradient doesn't work with tracers
        grad_x = jnp.gradient(An, proc.dx, axis=1)
        grad_y = jnp.gradient(An, proc.dy, axis=0)

        # Build gradient field normalized by An
        nabla_An = jnp.zeros((proc.ny, proc.nx, 2), dtype=jnp.complex64)
        nabla_An = nabla_An.at[:, :, 0].set(grad_x / An)
        nabla_An = nabla_An.at[:, :, 1].set(grad_y / An)

        # Imaginary part of normalized gradient field
        holder = jnp.imag(nabla_An)

        # Broadcast q vector across field
        q_field = jnp.broadcast_to(q, (proc.ny, proc.nx, 2))

        # Tensor product and scale
        return -(2 / 6) * jnp.einsum('ijk,ijl->ijkl', q_field, holder, optimize=True)

    vecs = jnp.arange(len(proc.qs))

    Q = jnp.sum(jax.vmap(compute_single_q)(vecs), axis=0)
    
    sp = JGkernel_jax(proc.target_shape,proc.dx,proc.dy,proc.a0/4)
    fft_Q = jnp.fft.fft2(Q, axes=(0, 1))
    filtered_fft_Q = jnp.einsum('xyij,xy->xyij', fft_Q, sp, optimize=True)
    convQ = jnp.fft.ifft2(filtered_fft_Q, axes=(0, 1)).real

    # return convQ
    return convQ.reshape(-1,4)[proc.rev_DofMap].ravel()

def Compute_Q_sym_jax(FE_psi,proc):
    """
        Computes Q based on psi using Jax:
        FE_psi is un unordered array coming for dolfinx. It's reordred into psi
        Returns a flattened out array in the same order as that of FE.
    """
    psi = jnp.array(FE_psi[proc.DofMap].reshape(proc.target_shape))
    # Convert qs to a JAX array outside the inner function
    qs_array = jnp.array(proc.qs)
    
    def compute_single_q(n):
        q = qs_array[n]  

        # Exponential modulation
        f = lambda x, y: jnp.exp(-1j * (q[0] * x + q[1] * y))
        FF = f(proc.X, proc.Y)

        # Forward FFT
        image_fft = jnp.fft.fft2(psi * FF)

        # Convolve in Fourier domain
        convolved_fft = image_fft * proc.AmpKernel

        # Inverse FFT to get convolved image
        An = jnp.fft.ifft2(convolved_fft)

        # Approximate gradient using finite differences manually, since jnp.gradient doesn't work with tracers
        grad_x = jnp.gradient(An, proc.dx, axis=1)
        grad_y = jnp.gradient(An, proc.dy, axis=0)

        # Build gradient field normalized by An
        nabla_An = jnp.zeros((proc.ny, proc.nx, 2), dtype=jnp.complex64)
        nabla_An = nabla_An.at[:, :, 0].set(grad_x / An)
        nabla_An = nabla_An.at[:, :, 1].set(grad_y / An)

        # Imaginary part of normalized gradient field
        holder = jnp.imag(nabla_An)

        # Broadcast q vector across field
        q_field = jnp.broadcast_to(q, (proc.ny, proc.nx, 2))

        # Tensor product and scale
        return -(2 / 6) * jnp.einsum('ijk,ijl->ijkl', q_field, holder, optimize=True)

    vecs = jnp.arange(len(proc.qs))

    Q = jnp.sum(jax.vmap(compute_single_q)(vecs), axis=0)
    
    sp = JGkernel_jax(proc.target_shape,proc.dx,proc.dy,proc.a0/4)
    fft_Q = jnp.fft.fft2(Q, axes=(0, 1))
    filtered_fft_Q = jnp.einsum('xyij,xy->xyij', fft_Q, sp, optimize=True)
    convQ = jnp.fft.ifft2(filtered_fft_Q, axes=(0, 1)).real
    sym = (1/2)*(convQ+np.transpose(convQ,axes=(0, 1, 3, 2)))
    # return convQ
    return sym.reshape(-1,4)[proc.rev_DofMap].ravel()


def fuq_loss(FE_psi, U, proc):
    Q = Compute_Q_jax(FE_psi, proc)
    return jnp.sum((Q - U)**2)

def fuq_loss_sym(FE_psi, U, proc):
    Q = Compute_Q_sym_jax(FE_psi, proc)
    return jnp.sum((Q - U)**2)

def jax_computegradFuq(FE_psi,U,proc):
    k= jax.grad(fuq_loss,argnums=0)(FE_psi,U,proc)
    return k



def jax_computegradFuq_sym(FE_psi,U,proc):
    k= jax.grad(fuq_loss_sym,argnums=0)(FE_psi,U,proc)
    return k