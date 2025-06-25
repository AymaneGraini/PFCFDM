"""
A class to compute auxiliary fields that don't appear directly in the variational formulation but are used during the simulation or during the post-processing.
"""
from .PfFe import *



class PfComp:
    """
        A class to compute auxiliary fields doesn't appear directly in 
        the variational formulation but are used for either during the simulation or during the post-processing.
    """
    def __init__(self,
                 pfFe : PFFe):
        """
            Initializes the PfComp class with a PFFe instance.
            The parameter used is :math:`\\sigma = a_0/120` the standard deviation of a sharply peaked Gaussian used to approximate the delta function in the PFC post-processing.

            Args:
                pfFe (PFFe): An instance of the PFFe class which contains the phase field finite element setup.
        """
        self.pfFe = pfFe
        self.sig  = self.pfFe.pfc_params.a0/120
        self.set_funcs()


    def set_funcs(self):
        """
            Defines the main functions computable from :math:`\\psi` like:

            - :math:`\\alpha_T` the dislocation density computed from the complex amplitudes using eq. 12 from 2022 Jorge's paper
            - :math:`\mathbf{Q}` THe configurational distortion from the complex amplitudes
            - :math:`\mathbf{v}` Dislocation velocity field from the complex amplitudes using eq. 16 from 2022 Jorge's paper
            - :math:`\mathcal{J}` The current density field from the complex amplitudes using eq. 14 from 2022 Jorge's paper  
        """
        self.alphaT   = fem.Function(self.pfFe.tensor_sp3,name="alphaTild")
        self.Q        = fem.Function(self.pfFe.tensor_sp2,name="Q")
        self.velocity = fem.Function(self.pfFe.vector_sp2,name="Velocity")
        self.J        = fem.Function(self.pfFe.tensor_sp2,name="J_tens")

    def update_cAmps(self,amps,order):
        """
            Updates the complex amplitudes in the PFFe instance with the provided amplitudes.
            Stores the old amplitudes for interpolation purposes, and replace the current amplitudes with the new ones.
            
            Args:
                amps (list of numpy.ndarray): A list of complex amplitudes to be updated.
                order (numpy.ndarray): An array that defines the order in which the amplitudes should be reshaped and assigned.
        """
        for i in range(len(self.pfFe.pfc_params.qs)):
            self.pfFe.Re_amps_old[i].interpolate(self.pfFe.Re_amps[i])
            self.pfFe.Im_amps_old[i].interpolate(self.pfFe.Im_amps[i])

        for i, (re, im) in enumerate(zip(self.pfFe.Re_amps, self.pfFe.Im_amps)):
            re.x.array[:] = np.real(amps[i].reshape(-1)[order].ravel())
            im.x.array[:] = np.imag(amps[i].reshape(-1)[order].ravel())
            re.x.scatter_forward()
            im.x.scatter_forward()

    def compute_Q(self):
        """
            Computes the configurational distortion tensor :math:`\mathbf{Q}` from the complex amplitudes as:

            .. math::
            
                \\mathbf{Q} = - \\frac{n}{N} \sum_{i=1}^{N} \\vec{q_i} \otimes \\text{Im} \left( \\frac{\\vec{\\nabla} A_i}{A_i}\\right)
        """
        exp=0
        for i,q in enumerate(self.pfFe.pfc_params.qs):
            D=ufl.grad(self.pfFe.Im_amps[i])*self.pfFe.Re_amps[i]- ufl.grad(self.pfFe.Re_amps[i])*self.pfFe.Im_amps[i]
            q_field = ufl.as_vector(q) 
            pref=1/(self.pfFe.Im_amps[i]**2+self.pfFe.Re_amps[i]**2)
            exp+=pref*ufl.outer(q_field,D)
        exp*=-2.0/len(self.pfFe.pfc_params.qs)
        self.QT.interpolate(fem.Expression(exp,self.pfFe.tensor_sp2.element.interpolation_points()))
        self.alphapfc.interpolate(fem.Expression(tcurl(extendT(self.QT)),self.pfFe.tensor_sp3.element.interpolation_points()))

    def compute_velocityPFC(self):
        """
            Computes the dislocation velocity field :math:`\mathbf{v}` from the complex amplitudes using Jorge's 2022 paper, eq. 16:

            .. math::

                \\vec{v} = \\dots


            NOTE:
            This not used, not confirmed so far, but it is a good candidate to be used in the future maybee
        """
        self.indicator.interpolate(fem.Expression(ufl.sqrt(self.alphaT[0,2]**2+self.alphaT[1,2]**2),self.scalar_sp.element.interpolation_points()))
        v_exp=0
        for i,q in enumerate(self.pfFe.pfc_params.qs):
            D= ufl.cross(extendV(ufl.grad(self.pfFe.Re_amps[i])),extendV(ufl.grad(self.pfFe.Im_amps[i])))
            Re_dot = (self.pfFe.Re_amps[i]-self.pfFe.Re_amps_old[i])/self.pfFe.sim_params.dt
            Im_dot = (self.pfFe.Im_amps[i]-self.pfFe.Im_amps_old[i])/self.pfFe.sim_params.dt
            j= Im_dot*ufl.grad(self.pfFe.Re_amps[i])-Re_dot*ufl.grad(self.pfFe.Im_amps[i])
            q_field = ufl.as_vector([q[0],q[1],0])
            b= ufl.dot(self.alphaT,ufl.as_vector([0,0,1]))
            sn = ufl.dot(b/ufl.sqrt(ufl.dot(b,b)),q_field) # a scalar : WInding numer (but it's a density actually)
            v_exp += ((sn**2))*j/D[2]
        v_exp*=12*np.pi**2/len(self.pfFe.pfc_params.qs) #TODO 12 is maybe wrong
        self.velocity.interpolate(fem.Expression(ufl.conditional(ufl.ge(self.indicator,1e-2),v_exp,ufl.as_vector([0,0])),self.pfFe.vector_sp2.element.interpolation_points()))

    def compute_velocityPFC_bis(self):
        """
            Computes the dislocation velocity field :math:`\mathbf{v}` from the complex amplitudes using Jorge's 2022 paper, eq. 16:

            .. math::

                \\vec{v} = \\dots


            NOTE:
            This is different from the previous method in the term replacing :math:`\\vec{b}` because the Burger's vector field is not really known and must be replacement by something continuous using the dislocation density tensor :math:`\\alpha_T`
        """
        #TODO confirm
        self.indicator.interpolate(fem.Expression(ufl.sqrt(self.alphaT[0,2]**2+self.alphaT[1,2]**2),self.pfFe.scalar_sp.element.interpolation_points()))
        S=0
        D=0
        for i,q in enumerate(self.pfFe.pfc_params.qs):
            d= ufl.cross(extendV(ufl.grad(self.pfFe.Re_amps[i])),extendV(ufl.grad(self.pfFe.Im_amps[i])))
            Re_dot = (self.pfFe.Re_amps[i]-self.pfFe.Re_amps_old[i])/self.pfFe.sim_params.dt
            Im_dot = (self.pfFe.Im_amps[i]-self.pfFe.Im_amps_old[i])/self.pfFe.sim_params.dt
            j= Im_dot*ufl.grad(self.pfFe.Re_amps[i])-Re_dot*ufl.grad(self.pfFe.Im_amps[i])
            pref=(1/(2*np.pi*self.sig**2))*ufl.exp(-(self.pfFe.Re_amps[i]**2+self.pfFe.Im_amps[i]**2)/(2*self.sig**2))
            S+=3*pref*(q[0]**2+q[1]**2)*j #TODO 3 is maybe wrong
            D+=d
        v_exp=restrictV(ufl.cross(extendV(j),D)/ufl.dot(D,D))
        self.velocity.interpolate(fem.Expression(ufl.conditional(ufl.ge(self.indicator,1e-2),v_exp,ufl.as_vector([0,0])),self.pfFe.vector_sp2.element.interpolation_points()))

    def Compute_current(self):
        """
            Compute the current due to the conservation of topological charge. There are two ways :


            - If the desired method is "J", it computes the current density from the complex amplitudes using Jorge's 2022 paper, eq. 13 without computing the velocity field, but rather computing directly :math:`\\mathcal{J} \sim \\boldsymbol{\\alpha} \\times \\vec{v}` as:

            .. math::

                \\mathcal{J} = \\frac{2*d \\pi}{N} \sum_{i=1}^{N} \delta(A_i) \\vec{q_i} \otimes \\text{Im} \left( \dot{A_i} \\vec{\\nabla}A_i \\right)
                
            where :math:`\\dot{A_i} = \\frac{A_i ^{t+dt}-A_i ^{t}}{dt}` is computed using the stored amplitudes and the Dirac's delta function is approximated as :math:`\\delta(x) \\approx \\frac{1}{2 \\pi \\sigma^2} e^{-\\frac{x^2}{2 \\sigma^2}}` with :math:`\\sigma = \\frac{a_0}{122}`.

            - If the desired method is "v", then we start by computing the velcotiy field using self.compute_velocityPFC_bis() and then compute the current density as:

            .. math::
                \\mathcal{J} = \\alpha_T \\times \\vec{v}
            
            where :math:`\\alpha_T` is the dislocation density tensor computed from the complex amplitudes.

        """
        if self.pfFe.pfc_params.motion=="J":
            exp=0
            for i,q in enumerate(self.pfFe.pfc_params.qs):
                Re_dot = (self.pfFe.Re_amps[i]-self.pfFe.Re_amps_old[i])/self.pfFe.sim_params.dt
                Im_dot = (self.pfFe.Im_amps[i]-self.pfFe.Im_amps_old[i])/self.pfFe.sim_params.dt
                j= Im_dot*ufl.grad(self.pfFe.Re_amps[i])-Re_dot*ufl.grad(self.pfFe.Im_amps[i])
                q_field = ufl.as_vector([q[0],q[1]]) 
                pref=(1/(2*np.pi*self.sig**2))*ufl.exp(-(self.pfFe.Re_amps[i]**2+self.pfFe.Im_amps[i]**2)/(2*self.sig**2))
                exp+=pref*ufl.outer(q_field,j)
            exp*=(2*2*np.pi)/len(self.pfFe.pfc_params.qs) #TODO -1 ?
        elif self.pfFe.pfc_params.motion=="v":
            print("Updating with velocity")
            self.compute_velocityPFC_bis()
            exp = restrictT(tcrossv(self.alphaT,extendV(self.velocity)))

        else:
            raise ValueError("Not implemented way of updating UP")
        
        self.J.interpolate(fem.Expression(exp,self.pfFe.tensor_sp2.element.interpolation_points()))

    def Compute_alpha_tilde(self):
        """
        Computes the dislocation density tensor :math:`\\alpha_T` from the complex amplitudes using Jorge's 2022 paper, eq. 12:


        .. math::

            \\alpha_T = \\frac{2*d \\pi}{N} \sum_{i=1}^{N} \delta(A_i) \\vec{q_i} \otimes \\left(\\vec{\\nabla} \\text{Re}(A_i) \\times \\vec{\\nabla} \\text{Im}(A_i)\\right)
        
        where :math:`\\delta(x) \\approx \\frac{1}{2 \\pi \\sigma^2} e^{-\\frac{x^2}{2 \\sigma^2}}` is the Dirac's delta function approximated by a sharply peaked Gaussian with standard deviation :math:`\\sigma = \\frac{a_0}{122}`.
        """
        exp=0
        for i,q in enumerate(self.pfFe.pfc_params.qs):
            D= ufl.cross(extendV(ufl.grad(self.pfFe.Re_amps[i])),extendV(ufl.grad(self.pfFe.Im_amps[i])))
            q_field = ufl.as_vector([q[0],q[1],0]) 
            pref=(1/(2*np.pi*self.sig**2))*ufl.exp(-(self.pfFe.Re_amps[i]**2+self.pfFe.Im_amps[i]**2)/(2*self.sig**2))
            exp+=pref*ufl.outer(q_field,D)
        exp*=(2*2*np.pi)/len(self.pfFe.pfc_params.qs)
        self.alphaT.interpolate(fem.Expression(exp,self.pfFe.tensor_sp3.element.interpolation_points()))



    def Compute_microscopic_stress(self):
        """ TODO CHECK THIS FIRST 
        Due to the fact that sigmapsi depends on the gradient of gradient of psi, we use FE to define it weakly using integration by parts"""
        r=self.pfc_params['r']
        if self.MEl_space.num_sub_spaces ==3 :
            psi,mu,chi = ufl.split(self.zeta0)
        elif self.MEl_space.num_sub_spaces ==2 :
            psi,chi = ufl.split(self.zeta0)
        else: 
            raise ValueError("Mixed element space is not correctly defined, cannot extract chi and psi to compute microStress")
        
        self.a_sigma_psi = fem.form(ufl.inner(self.sigma_tri,self.tau)*ufl.dx)

        self.L_sigma_psi = fem.form(
                            ufl.inner(2*ufl.sym(ufl.outer(ufl.grad(psi+chi),ufl.grad(psi))),self.tau)*ufl.dx+
                            ufl.inner((psi+chi)*ufl.grad(psi),ufl.div(self.tau))*ufl.dx+
                            ufl.inner((1/2)*((psi+chi)**2-r*psi**2+(1/2)*psi**4),ufl.tr(self.tau))*ufl.dx
                            )
        if not self.SET_SIGMA_PROBLEM:
            self.A_sigma_psi = fem.petsc.assemble_matrix(self.a_sigma_psi, bcs=[])
            self.A_sigma_psi.assemble()
            self.problem_sigma_psi = PETSc.KSP().create(self.domain.comm)
            self.problem_sigma_psi.setOperators(self.A_sigma_psi)
            self.problem_sigma_psi.setType(PETSc.KSP.Type.PREONLY)
            pcs = self.problem_sigma_psi.getPC()
            pcs.setType(PETSc.PC.Type.LU)
            pcs.setReusePreconditioner(True)
            self.SET_SIGMA_PROBLEM= True

        with dolfinx.common.Timer() as t_cpu:
                b_sigma_psi = fem.petsc.create_vector(self.L_sigma_psi)
                with b_sigma_psi.localForm() as loc_b:
                    loc_b.set(0)
                fem.petsc.assemble_vector(b_sigma_psi, self.L_sigma_psi)
                self.problem_sigma_psi.solve(b_sigma_psi, self.micro_sigma.x.petsc_vec)


        #TODO REDO THE DERIVATION OF THE WEAK FORM ANC CHECK IF 1/2 is CORRECT
