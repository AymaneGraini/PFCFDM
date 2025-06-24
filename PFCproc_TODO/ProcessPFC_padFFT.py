"""
    A file containing the PFCProcessor class which processes PFC data external to dolfinx, namely FFT and jax operations.
""" 
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import tree_util
from functools import partial

def Jcompute_gradient(scalar_field, ax, sp) -> jnp.ndarray:
    """
    Computes the gradient of a scalar field along a specified axis using finite differences.

    Args:
        scalar_field (jnp.ndarray): The scalar field to compute the gradient of.
        ax (int): The axis along which to compute the gradient.
        sp (float): The spacing between points in the specified axis.       

    Returns:
        jnp.ndarray: The gradient of the scalar field along the specified axis.
    """
    return jnp.gradient(scalar_field, sp, axis=ax)

def Jgradient_periodic(array, spacing, axis):
    """Computes the gradient of a periodic array along a specified axis using finite differences.
    Args:
        array (jnp.ndarray): The input array for which the gradient is computed.
        spacing (float): The spacing between points in the specified axis.
        axis (int): The axis along which to compute the gradient.   

    Returns:
        jnp.ndarray: The gradient of the input array along the specified axis.
    """
    return jnp.gradient(array, spacing, axis=axis)


def JGkernel_jax(shape, dx, dy, sigma):
    """Generates a Gaussian kernel in Fourier space for a given shape and spacing.

    Args:
        shape (tuple): The shape of the kernel to be generated.
        dx (float): The spacing in the x-direction.
        dy (float): The spacing in the y-direction.
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        jnp.ndarray: The generated Gaussian kernel in Fourier space.
    """

    wx = jnp.fft.fftfreq(shape[1], dx)  
    wy = jnp.fft.fftfreq(shape[0], dy)  
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





class PFCProcessorEXT():
    def __init__(
        self,
        space_coords: np.ndarray,
        DofMap:np.array,
        qs : np.ndarray,
        a0:float,
        target_shape : tuple,
        pads : tuple,
    ) -> None:
        """
        Initializes the PFCProcessor with the necessary parameters for processing PFC data external to dolfinx.
        it is mainly used to store the target shape for meshgrid and how to go and from the dolfinx DofMap to  a meshgrid.

        Args:
            space_coords (_type_): ndarray of space coordinates of the domain
            DofMap (_type_): order of the dof to map them from FE to meshgrid
            qs (_type_): array of 1st mode vectors
            a0 (_type_): lattice spacing
            target_shape (tuple): Target shape
            pads (tuple): pads #TODO
        """
        
        self.space_coords= space_coords
        self.a0=a0
        self.qs= jnp.array(qs)

        # the order to go from the dolfinx DofMap to a meshgrid
        self.DofMap = DofMap
        # the target shape for the meshgrid
        self.target_shape=target_shape


        #Build the reverse map from the meshgrid to the dolfinx DofMap
        self.rev_DofMap = jnp.empty_like(self.DofMap)
        self.rev_DofMap = self.rev_DofMap.at[DofMap].set(jnp.arange(len(DofMap)))

        #Extract the x and y coordinates from the space_coords
        # and create a meshgrid for the target shape
        self.coords =space_coords[:,:2][DofMap]
        x_coords =jnp.unique(self.coords[:, 0],size= target_shape[1])
        y_coords =jnp.unique(self.coords[:, 1],size= target_shape[0])
        self.X, self.Y = jnp.meshgrid(x_coords, y_coords)
        self.L = self.X.max() - self.X.min()
        self.H = self.Y.max() - self.Y.min()
        self.nx = self.X.shape[1]
        self.ny = self.X.shape[0]
        self.dx = self.L / (self.target_shape[1] - 1) 
        self.dy = self.H / (self.target_shape[0]- 1)  


        #Build the padded shape and the kernel for the convolution
        self.padx, self.pady = pads
        self.pad_shape = (2*self.target_shape[0], 2*self.target_shape[1])
        kernel =  np.exp(-((self.X-self.L/2)**2 + (self.Y-self.H/2)**2) / (2 * self.a0**2))
        kernel /= np.sum(kernel)

        self.AmpKernel = np.zeros(self.pad_shape)

        start_x = (self.pad_shape[0] - self.target_shape[0]) // 2
        start_y = (self.pad_shape[1] - self.target_shape[1]) // 2

        self.AmpKernel[start_x:start_x + self.target_shape[0],
                    start_y:start_y + self.target_shape[1]] = kernel


        self.AmpKernel=jnp.fft.fft2(self.AmpKernel)
        self.AmpKernel = JGkernel_jax(self.pad_shape,self.dx,self.dy,self.a0) #TODO remove this line maybe the kernel is fft made not defined in fourier space over the paded domain  or remove all the previous definition

        #Wether to filter Q or not
        self.filterQ = True

        #build fourier space vectors
        kx = jnp.fft.fftfreq(self.nx,self.dx)*2*jnp.pi
        ky = jnp.fft.fftfreq(self.ny,self.dy) *2*jnp.pi
        KX,KY=jnp.meshgrid(kx,ky)
        self.K_vec = jnp.stack((KX, KY), axis=-1)

        #Build an array of Fourier modes for each q vector
        self.FF = []  
        for i in range(len(qs)):
            q = self.qs[i]
            f = lambda x, y: jnp.exp(-1j * (q[0] * x + q[1] * y))
            self.FF.append(f(self.X, self.Y))  
        self.FF=jnp.array(self.FF) 
        
    # @partial(jax.jit, static_argnames=['self'])
    def C_Amp(self,op,i) -> jnp.ndarray:
        """
        Computes the amplitude of the Fourier mode i from the flattened out ndarray op.
        The input op is expected to be a flattened out ndarray coming from dolfinx in the order of dolfinx and needs to be reorder into a meshgrid of the target shape. (Ny,Nx)

        Args:
            op (np.ndarray): psi field to demodulate
            i (int): Fourier mode index to compute the amplitude for.

        Returns:
            jnp.ndarray: The complex amplitude of the Fourier mode i. in  the shape of the target shape. and ordered as a meshgrid.
            it needs to be flattened out an reordered to be used in dolfinx.
        """
        #op is called as a flattened out ndarray coming from dolfinx
        psi = jnp.array(op[self.DofMap].reshape(self.target_shape)) # Reorder the input and reshape to the target shape

        # pad the field to the padded shape
        start_x = (self.pad_shape[0] - self.target_shape[0]) // 2
        start_y = (self.pad_shape[1] - self.target_shape[1]) // 2

        field_padded = np.zeros(self.pad_shape,dtype=np.complex64)

        field_padded[start_x:start_x + self.target_shape[0],
                    start_y:start_y + self.target_shape[1]] = psi*self.FF[i]
        
     
        # Compute the Fourier transform of the padded field
        # and convolve it with the amplitude kernel in Fourier space
        image_fft = jnp.fft.fft2(field_padded)
        convolved_fft = image_fft * self.AmpKernel
        convolved = jnp.fft.ifft2(convolved_fft)

        return convolved[start_x:start_x + self.target_shape[0],
                    start_y:start_y + self.target_shape[1]]
        
    # @partial(jax.jit, static_argnames=['self'])
    def Compute_Q(self,amps) -> jnp.ndarray:
        """Compute the Q tensor from the amplitude fields. 
        There are four steps:
        1. Compute the contribution of each mode, using eisnsum to compute the outer product
        2. Sum the contributions across all modes using jax.vmap
        3. If filterQ is True, convolve the result with a Gaussian kernel in Fourier space to smooth the result
        4. Reshape the result to match the original DofMap order. (the order of dolfinx)

        Args:
            amps (jnp.ndarray): The amplitude fields for each q vector, shape (N, ny, nx) in meshgrid order.
        Returns:
            jnp.ndarray: The computed Q tensor, flattened out and reordered to match the DofMap in dolfinx. 
        """
        def compute_single_q(n):
            An = amps[n]
            grad_x = Jcompute_gradient(An, 1, self.dx)
            grad_y = Jcompute_gradient(An, 0, self.dy)
            nabla_An = jnp.zeros((self.ny, self.nx, 2), dtype=jnp.complex64).at[:, :, 0].set(grad_x / An).at[:, :, 1].set(grad_y / An)
            holder = jnp.imag(nabla_An)
            q_field = jnp.broadcast_to(self.qs[n], (self.ny, self.nx, 2))
            return -(2 / 6) * jnp.einsum('ijk,ijl->ijkl', q_field, holder, optimize=True)
        vecs = np.arange(0,len(self.qs),1) 
        Qcomp = jnp.sum(jax.vmap(compute_single_q)(vecs), axis=0) #This computes the Q tensor by summing contributions from all modes shape is (ny, nx, 2, 2)

        # If not filtering Q, return the computed Q
        if not self.filterQ:
            return Qcomp.reshape(-1,4)[self.rev_DofMap].ravel()
        else:
            # If filtering Q, convolve it with a Gaussian kernel in Fourier space
            # and return the filtered result
            start_x = (self.pad_shape[0] - self.target_shape[0]) // 2
            start_y = (self.pad_shape[1] - self.target_shape[1]) // 2
            sp = JGkernel_jax(self.pad_shape,self.dx,self.dy,self.a0/4) #is this okay ? TODO
            convQ = np.zeros_like(Qcomp)
            for i in range(2):
                for j in range(2):
                    field_padded = np.zeros(self.pad_shape,dtype=np.complex64)
                    field_padded[start_x:start_x + self.target_shape[0],
                                        start_y:start_y + self.target_shape[1]] = Qcomp[:,:,i,j]
                    fft_Q = jnp.fft.fft2(field_padded)
                    convQ[:,:,i,j] = jnp.fft.ifft2(fft_Q*sp, axes=(0, 1)).real[start_x:start_x + self.target_shape[0],
                                        start_y:start_y + self.target_shape[1]]
            return convQ.reshape(-1,4)[self.rev_DofMap].ravel()


    # @partial(jax.jit, static_argnames=['self'])
    def Compute_alpha(self,Qt)-> jnp.ndarray:
        """ Computes the alpha tensor from the Q tensor.
        The alpha tensor is computed as the curl of the
        Q tensor using the Levi-Civita symbol and the gradients of Q.
        The input Qt is expected to be a flattened out ndarray coming from dolfinx in the order of dolfinx and needs to be reordered into a meshgrid of the target shape. (Ny,Nx,2,2)
        
        Args:
            Qt (jnp.ndarray): The Q tensor, flattened out and reordered to match the DofMap in dolfinx.
        Returns:   
            jnp.ndarray: The computed alpha tensor, flattened out and reordered to match the DofMap in dolfinx.
        """
        Qcomp= Qt.reshape(-1,4)[self.DofMap].reshape(self.ny,self.nx,2,2)
        Nx, Ny = Qcomp.shape[1], Qcomp.shape[0]
        curl = jnp.zeros((Nx, Ny,3,3))
        grad_x = Jcompute_gradient(Qcomp, 1, self.dx)
        grad_y = Jcompute_gradient(Qcomp, 0, self.dy)
        derivative_array = jnp.zeros((self.ny, self.nx,2,2,2)).at[:,:,0,:,:].set(grad_x).at[:,:,1,:,:].set(grad_y)
        padded_array = jnp.pad(derivative_array, ((0, 0), (0, 0), (0, 1), (0, 1), (0, 1)), mode='constant', constant_values=0)
        curl = jnp.einsum('jkl,abkil->abij', Levi,padded_array,optimize=True)
        return curl.reshape(-1,9)[self.rev_DofMap].ravel()

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
    
def Compute_Q_jax(FE_psi: jnp.ndarray,proc : PFCProcessorEXT) -> jnp.ndarray:
    """
        Computes Q based on psi using Jax:
        FE_psi is un unordered array coming for dolfinx. It needs to be reordered and reshaped into meshgrid of the target shape.

        Returns a flattened out array in the same order as that of FE .
        Args:
            FE_psi (jnp.ndarray): Unordered flat array of psi values from dolfinx
            proc (PFCProcessor): Instance of PFCProcessor containing processing parameters and order from FE to FFT.
    """

    # Reorder and reshape FE_psi into the target shape
    psi = jnp.array(FE_psi[proc.DofMap].reshape(proc.target_shape))

    # Convert qs to a JAX array outside the inner function
    qs_array = jnp.array(proc.qs)
    
    # Define a function to compute the contribution of a single q vector
    def compute_single_q(n):
        q = qs_array[n]  

        # Exponential modulation
        f = lambda x, y: jnp.exp(-1j * (q[0] * x + q[1] * y))
        FF = f(proc.X, proc.Y)

        start_x = (proc.pad_shape[0] - proc.target_shape[0]) // 2
        start_y = (proc.pad_shape[1] - proc.target_shape[1]) // 2

        field_padded = jnp.zeros(proc.pad_shape,dtype=np.complex64)

        field_padded= field_padded.at[start_x:start_x + proc.target_shape[0],
                    start_y:start_y + proc.target_shape[1]].set(psi*FF)
        
     

        image_fft = jnp.fft.fft2(field_padded)
        convolved_fft = image_fft * proc.AmpKernel
        convolved = jnp.fft.ifft2(convolved_fft)

        # Extract the amplitude field An from the convolved result
        An = convolved[start_x:start_x + proc.target_shape[0],
                    start_y:start_y + proc.target_shape[1]]
        

        # Approximate the gradient using jnp.gradient 
        grad_x = jnp.gradient(An, proc.dx, axis=1)
        grad_y = jnp.gradient(An, proc.dy, axis=0)

        # Build gradient field divided by An
        nabla_An = jnp.zeros((proc.ny, proc.nx, 2), dtype=jnp.complex64)
        nabla_An = nabla_An.at[:, :, 0].set(grad_x / An)
        nabla_An = nabla_An.at[:, :, 1].set(grad_y / An)

        # Imaginary part of gradient field / amplitude
        holder = jnp.imag(nabla_An)

        # Broadcast q vector across field
        q_field = jnp.broadcast_to(q, (proc.ny, proc.nx, 2))

        # Tensor product and scale
        return -(2 / 6) * jnp.einsum('ijk,ijl->ijkl', q_field, holder, optimize=True)

    vecs = jnp.arange(len(proc.qs))
    # Compute contributions from all modes using jax.vmap
    Qcomp = jnp.sum(jax.vmap(compute_single_q)(vecs), axis=0)


    #Filter Q with a Gaussian kernel in Fourier space
    start_x = (proc.pad_shape[0] - proc.target_shape[0]) // 2
    start_y = (proc.pad_shape[1] - proc.target_shape[1]) // 2
    sp = JGkernel_jax(proc.pad_shape,proc.dx,proc.dy,proc.a0/4)
    convQ = jnp.zeros_like(Qcomp)

    for i in range(2):
        for j in range(2):
                    field_padded = jnp.zeros(proc.pad_shape,dtype=np.complex64)
                    field_padded= field_padded.at[start_x:start_x + proc.target_shape[0],
                                        start_y:start_y + proc.target_shape[1]].set(Qcomp[:,:,i,j])
                    fft_Q = jnp.fft.fft2(field_padded)
                    convQ = convQ.at[:,:,i,j].set(jnp.fft.ifft2(fft_Q*sp, axes=(0, 1)).real[start_x:start_x + proc.target_shape[0],
                                        start_y:start_y + proc.target_shape[1]])
    # Return the computed Q tensor, flattened out and reordered to match the DofMap in dolfinx
    return convQ.reshape(-1,4)[proc.rev_DofMap].ravel()

def Compute_Q_sym_jax(FE_psi,proc):
    """
        Computes sym(Q) based on psi using Jax:
        FE_psi is un unordered array coming for dolfinx. It needs to be reordered and reshaped into meshgrid of the target shape.

        Returns a flattened out array in the same order as that of FE .
        Args:
            FE_psi (jnp.ndarray): Unordered flat array of psi values from dolfinx
            proc (PFCProcessor): Instance of PFCProcessor containing processing parameters and order from FE to FFT.
    """
    psi = jnp.array(FE_psi[proc.DofMap].reshape(proc.target_shape))
    # Convert qs to a JAX array outside the inner function
    qs_array = jnp.array(proc.qs)
    
    def compute_single_q(n):
        q = qs_array[n]  

        # Exponential modulation
        f = lambda x, y: jnp.exp(-1j * (q[0] * x + q[1] * y))
        FF = f(proc.X, proc.Y)

        start_x = (proc.pad_shape[0] - proc.target_shape[0]) // 2
        start_y = (proc.pad_shape[1] - proc.target_shape[1]) // 2

        field_padded = jnp.zeros(proc.pad_shape,dtype=np.complex64)
        field_padded= field_padded.at[start_x:start_x + proc.target_shape[0],
                    start_y:start_y + proc.target_shape[1]].set(psi*FF)
        
     
        # Compute the Fourier transform of the psi*e(-iqx)
        # and convolve it with the amplitude kernel in Fourier space
        image_fft = jnp.fft.fft2(field_padded)
        convolved_fft = image_fft * proc.AmpKernel
        convolved = jnp.fft.ifft2(convolved_fft)

        # Extract the amplitude field An from the convolved result
        An = convolved[start_x:start_x + proc.target_shape[0],
                    start_y:start_y + proc.target_shape[1]]

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
    # Compute contributions from all modes using jax.vmap
    # This computes the Q tensor by summing contributions from all modes shape is (ny, nx, 2, 2)
    # Note that this is not symmetrized yet
    Qcomp = jnp.sum(jax.vmap(compute_single_q)(vecs), axis=0)

    # Filter Q with a Gaussian kernel in Fourier space
    start_x = (proc.pad_shape[0] - proc.target_shape[0]) // 2
    start_y = (proc.pad_shape[1] - proc.target_shape[1]) // 2
    sp = JGkernel_jax(proc.pad_shape,proc.dx,proc.dy,proc.a0/4)
    convQ = jnp.zeros_like(Qcomp)
    for i in range(2):
        for j in range(2):
                    field_padded = jnp.zeros(proc.pad_shape,dtype=np.complex64)
                    field_padded= field_padded.at[start_x:start_x + proc.target_shape[0],
                                        start_y:start_y + proc.target_shape[1]].set(Qcomp[:,:,i,j])
                    fft_Q = jnp.fft.fft2(field_padded)
                    convQ = convQ.at[:,:,i,j].set(jnp.fft.ifft2(fft_Q*sp, axes=(0, 1)).real[start_x:start_x + proc.target_shape[0],
                                        start_y:start_y + proc.target_shape[1]])
                    
    # Compute the symmetric part of Q
    # sym(Q) = (Q + Q^T) / 2
    sym = (1/2)*(convQ+jnp.transpose(convQ,axes=(0, 1, 3, 2)))
    # Return the computed sym(Q) tensor, flattened out and reordered to match the DofMap in dolfinx
    return sym.reshape(-1,4)[proc.rev_DofMap].ravel()


def fuq_loss(FE_psi: jnp.ndarray, U: jnp.ndarray, proc: PFCProcessorEXT) -> float:
    """
        Compute the L2 between the given tensor U and the configurational distortion Q[psi] 

    Args:
        FE_psi (jnp.ndarray): Unordered flat array of psi values from dolfinx
        U (jnp.ndarray): Elastic distortion tensor to compare against
        proc (PFCProcessor): Instance of PFCProcessor containing processing parameters and order from FE to FFT.

    Returns:
        float: The computed L2 loss.
    """
    Q = Compute_Q_jax(FE_psi, proc) # Compute Q tensor for a given FE_psi
    # Compute the L2 loss between Q and U
    return jnp.sum((Q - U)**2)

def fuq_loss_sym(FE_psi: jnp.ndarray, U: jnp.ndarray, proc: PFCProcessorEXT) -> float:
    """
        Compute the L2 between the given tensor U and the symetric part ofconfigurational distortion Q[psi] 

    Args:
        FE_psi (jnp.ndarray): Unordered flat array of psi values from dolfinx
        U (jnp.ndarray): Elastic distortion tensor to compare against
        proc (PFCProcessor): Instance of PFCProcessor containing processing parameters and order from FE to FFT.

    Returns:
        float: The computed L2 loss.

    NOTE: This function requires that the input U is symetric and it is done in FE level, and passed as a symetric tensor to this function.
        It is not checked here. #TODO: add a check for symetry of U
    """
    Qsym = Compute_Q_sym_jax(FE_psi, proc) # Compute Q tensor for a given FE_psi
    # Compute the L2 loss between Q and U
    return jnp.sum((Qsym - U)**2)


def jax_computegradFuq(FE_psi: jnp.ndarray, U: jnp.ndarray, proc: PFCProcessorEXT) -> jnp.ndarray:
    """
        Computes the gradient of the L2 difference :math:s with respect to FE_psi using JAX.
    Args:
        FE_psi (jnp.ndarray): Unordered flat array of psi values from dolfinx
        U (jnp.ndarray): Elastic tensor to compare against
        proc (PFCProcessor): Instance of PFCProcessor containing processing parameters and order from FE to FFT.

    Returns:
        jnp.ndarray: The gradient of the loss function with respect to FE_psi in the same order as that of FE.
        The gradient is computed using JAX's automatic differentiation.
    NOTE:
        The output is a flattened out array in the same order as that of FE and can be used directly in dolfinx to replace the ndarray of df/du.

    """
    k = jax.grad(fuq_loss, argnums=0)(FE_psi, U, proc)
    return k

def jax_computegradFuq_sym(FE_psi: jnp.ndarray, U: jnp.ndarray, proc: PFCProcessorEXT) -> jnp.ndarray:
    """
        Computes the gradient of the L2 difference of symmetric parts function :math:s with respect to FE_psi using JAX.
    Args:
        FE_psi (jnp.ndarray): Unordered flat array of psi values from dolfinx
        U (jnp.ndarray): Elastic tensor to compare against
        proc (PFCProcessor): Instance of PFCProcessor containing processing parameters and order from FE to FFT.

    Returns:
        jnp.ndarray: The gradient of the loss function with respect to FE_psi in the same order as that of FE.
        The gradient is computed using JAX's automatic differentiation.
    NOTE:
        The output is a flattened out array in the same order as that of FE and can be used directly in dolfinx to replace the ndarray of df/du.
    """
    k = jax.grad(fuq_loss_sym, argnums=0)(FE_psi, U, proc)
    return k