from .PfFe import *



class PfComp:
    """
        A class to compute auxiliary fields doesn't appear directly in 
        the variational formulation but are used for post-processing
    """
    def __init__(self,
                 pfFe : PFFe):
        self.pfFe = pfFe
        self.sig  = self.pfFe.pfc_params.a0/122
        self.set_funcs()
    def set_funcs(self):
        """
            Defines the main functions computable from ψ like alpha, Q , V_d etc 
        """
        self.alphaT   = fem.Function(self.pfFe.tensor_sp3,name="alphaTild")
        self.Q        = fem.Function(self.pfFe.tensor_sp2,name="Q")
        self.velocity = fem.Function(self.pfFe.vector_sp2,name="Velocity")
        self.J        = fem.Function(self.pfFe.tensor_sp2,name="J_tens")
        self.alphaT   = fem.Function(self.pfFe.tensor_sp3,name="alphaTild")

    def update_cAmps(self,amps,order):
        for i in range(len(self.pfFe.pfc_params.qs)):
            self.pfFe.Re_amps_old[i].interpolate(self.pfFe.Re_amps[i])
            self.pfFe.Im_amps_old[i].interpolate(self.pfFe.Im_amps[i])

        for i, (re, im) in enumerate(zip(self.pfFe.Re_amps, self.pfFe.Im_amps)):
            re.x.array[:] = np.real(amps[i].reshape(-1)[order].ravel())
            im.x.array[:] = np.imag(amps[i].reshape(-1)[order].ravel())
            re.x.scatter_forward()
            im.x.scatter_forward()

    def compute_Q(self):
        exp=0
        for i,q in enumerate(self.pfFe.pfc_params.qs):
            D=ufl.grad(self.pfFe.Im_amps[i])*self.pfFe.Re_amps[i]- ufl.grad(self.pfFe.Re_amps[i])*self.pfFe.Im_amps[i]
            q_field = ufl.as_vector(q) 
            pref=1/(self.pfFe.Im_amps[i]**2+self.pfFe.Re_amps[i]**2)
            exp+=pref*ufl.outer(q_field,D)
        exp*=2/len(self.pfFe.pfc_params.qs)
        self.QT.interpolate(fem.Expression(exp,self.pfFe.tensor_sp2.element.interpolation_points()))
        self.alphapfc.interpolate(fem.Expression(tcurl(extendT(self.QT)),self.pfFe.tensor_sp3.element.interpolation_points()))

    def compute_velocityPFC(self):
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
        v_exp*=12*np.pi**2/len(self.pfFe.pfc_params.qs)
        self.velocity.interpolate(fem.Expression(ufl.conditional(ufl.ge(self.indicator,1e-2),v_exp,ufl.as_vector([0,0])),self.pfFe.vector_sp2.element.interpolation_points()))

    def compute_velocityPFC_bis(self):
        self.indicator.interpolate(fem.Expression(ufl.sqrt(self.alphaT[0,2]**2+self.alphaT[1,2]**2),self.pfFe.scalar_sp.element.interpolation_points()))
        S=0
        D=0
        for i,q in enumerate(self.pfFe.pfc_params.qs):
            d= ufl.cross(extendV(ufl.grad(self.pfFe.Re_amps[i])),extendV(ufl.grad(self.pfFe.Im_amps[i])))
            Re_dot = (self.pfFe.Re_amps[i]-self.pfFe.Re_amps_old[i])/self.pfFe.sim_params.dt
            Im_dot = (self.pfFe.Im_amps[i]-self.pfFe.Im_amps_old[i])/self.pfFe.sim_params.dt
            j= Im_dot*ufl.grad(self.pfFe.Re_amps[i])-Re_dot*ufl.grad(self.pfFe.Im_amps[i])
            pref=(1/(2*np.pi*self.sig**2))*ufl.exp(-(self.pfFe.Re_amps[i]**2+self.pfFe.Im_amps[i]**2)/(2*self.sig**2))
            S+=3*pref*(q[0]**2+q[1]**2)*j
            D+=d
        v_exp=restrictV(ufl.cross(extendV(j),D)/ufl.dot(D,D))
        self.velocity.interpolate(fem.Expression(ufl.conditional(ufl.ge(self.indicator,1e-2),v_exp,ufl.as_vector([0,0])),self.pfFe.vector_sp2.element.interpolation_points()))

    def Compute_current(self):
        if self.pfFe.pfc_params.motion=="up":
            exp=0
            for i,q in enumerate(self.pfFe.pfc_params.qs):
                Re_dot = (self.pfFe.Re_amps[i]-self.pfFe.Re_amps_old[i])/self.pfFe.sim_params.dt
                Im_dot = (self.pfFe.Im_amps[i]-self.pfFe.Im_amps_old[i])/self.pfFe.sim_params.dt
                j= Im_dot*ufl.grad(self.pfFe.Re_amps[i])-Re_dot*ufl.grad(self.pfFe.Im_amps[i])
                q_field = ufl.as_vector([q[0],q[1]]) 
                pref=(1/(2*np.pi*self.sig**2))*ufl.exp(-(self.pfFe.Re_amps[i]**2+self.pfFe.Im_amps[i]**2)/(2*self.sig**2))
                exp+=pref*ufl.outer(q_field,j)
            exp*=(2*3*np.pi)/len(self.pfFe.pfc_params.qs) #TODO -1 ?
        elif self.pfFe.pfc_params.motion=="v":
            print("Updating with velocity")
            self.compute_velocityPFC_bis()
            exp = restrictT(tcrossv(self.alphaT,extendV(self.velocity)))

        else:
            raise ValueError("Not implemented way of updating UP")
        
        self.J.interpolate(fem.Expression(exp,self.pfFe.tensor_sp2.element.interpolation_points()))

    def Compute_alpha_tilde(self):
        exp=0
        for i,q in enumerate(self.pfFe.pfc_params.qs):
            D= ufl.cross(extendV(ufl.grad(self.pfFe.Re_amps[i])),extendV(ufl.grad(self.pfFe.Im_amps[i])))
            q_field = ufl.as_vector([q[0],q[1],0]) 
            pref=(1/(2*np.pi*self.sig**2))*ufl.exp(-(self.pfFe.Re_amps[i]**2+self.pfFe.Im_amps[i]**2)/(2*self.sig**2))
            exp+=pref*ufl.outer(q_field,D)
        exp*=(2*3*np.pi)/len(self.pfFe.pfc_params.qs)
        self.alphaT.interpolate(fem.Expression(exp,self.pfFe.tensor_sp3.element.interpolation_points()))


    def Compute_microscopic_stress(self):
        """ Due to the fact that sigmapsi depends on the gradient of gradient of psi, we use FE to define it weakly using integration by parts"""
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
