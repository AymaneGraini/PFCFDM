import numpy as np
import dolfinx
import basix
import dolfinx.fem as fem
import ufl
import ufl.form
from utils.MPCLS import *
from utils.pbcs import *
from utils.MyAlgebra import *
from utils.monitor import *
from utils.utils import *
import scipy.ndimage as ndimage
import jax.numpy as jnp

class SolverPFC:
    def __init__(self,
                 domain:dolfinx.mesh.Mesh,
                 pfc_params: dict,
                 coupling_params: dict,
                 qs,
                 )->None:
        self.periodic = pfc_params['periodic']
        self.domain = domain
        self.pfc_params=pfc_params
        self.coupling_params=coupling_params
        self.qs=qs


        #We define a dictionary to keep track of positions of each defects, 
        # entries are in the pos[i]=[t,(x1,y1),(x2,y2)] position of each core at time t
        self.pos=[]
        self.last_pos=None

        self.sig = self.pfc_params['a0']/122    
        self.avg_list=[]

        self.SET_SIGMA_PROBLEM = False
        print("Called mother constructor ", self.domain)

    def set_spaces(self):
        self.scalar_sp = fem.functionspace(self.domain,self.elem)
        self.Ts_P3 = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1,shape=(3,3)) 
        self.tensor_sp3 = fem.functionspace(self.domain, self.Ts_P3)

        self.Ts_P2 = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1,shape=(2,2)) 
        self.tensor_sp2 = fem.functionspace(self.domain, self.Ts_P2)

        self.Vs_P2 = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1,shape=(2,)) 
        self.vector_sp2 = fem.functionspace(self.domain, self.Vs_P2)


    def set_funcs(self):
        """
            Defines the shared functions between all models, trial, split and tests functions remain
            specific to each model and their definition is implemented in each model, separetly
        """
        self.zeta0 = fem.Function(self.MEl_space) #function at previous step
        self.SH_sol = fem.Function(self.MEl_space) #Solved for (current sol)
        self.psiout=  fem.Function(self.scalar_sp,name="Psi") #function to store the output psi=mu.sub(0)
        self.dFQW=  fem.Function(self.scalar_sp,name="dFQW") 
        self.corr=  fem.Function(self.MEl_space,name="Correction")
        self.alphaT = fem.Function(self.tensor_sp3,name="alphaTild")
        self.alphapfc = fem.Function(self.tensor_sp3,name="alpha_pfc")
        self.velocity = fem.Function(self.vector_sp2,name="Velocity")
        self.indicator = fem.Function(self.scalar_sp,name="Ind") #Indicator, sacalar valued function to map the velocity!
        self.QT = fem.Function(self.tensor_sp2,name="QT")
        self.Q = fem.Function(self.tensor_sp2,name="Q")
        self.Qsym = fem.Function(self.tensor_sp2,name="Qsym")
        self.J = fem.Function(self.tensor_sp2,name="J_tens")
        self.micro_sigma = fem.Function(self.tensor_sp2,name="SigmaPsi")
        self.micro_sigma_avg = fem.Function(self.tensor_sp2,name="SigmaPsiAVG")
        self.Re_amps=[]
        self.Im_amps=[]

        for i,q in enumerate(self.qs):
            self.Re_amps.append(fem.Function(self.scalar_sp,name="RealAmp"+str(i),dtype=np.float64))
            self.Im_amps.append(fem.Function(self.scalar_sp,name="ImagAmp"+str(i),dtype=np.float64))

        self.Re_amps_old=[]
        self.Im_amps_old=[]
        for i,q in enumerate(self.qs):
            self.Re_amps_old.append(fem.Function(self.scalar_sp,name="RealAmp_old"+str(i),dtype=np.float64))
            self.Im_amps_old.append(fem.Function(self.scalar_sp,name="ImagAmp_old"+str(i),dtype=np.float64))

        self.sigma_tri = ufl.TrialFunction(self.tensor_sp2)
        self.tau = ufl.TestFunction(self.tensor_sp2)

    def set_projection_basis(self):
        self.b_basis = fem.Function(self.scalar_sp,name="basis")
        self.phi_i_intg = fem.assemble_vector(fem.form((1/(self.pfc_params['L']*self.pfc_params['H']))*ufl.TestFunction(self.scalar_sp) * ufl.dx))
        self.b_basis.x.array[:] = self.phi_i_intg.array.copy()
        self.b_basis_norm = np.dot(self.phi_i_intg.array, self.phi_i_intg.array)

    def create_forms(self):
        """ This is model dependent"""
        pass


    def initial_conditions(self,topo_info:str,qs,ps,list_defects)->None:
            Amp =  lambda avg,r : (1/5)*(np.absolute(avg)+(1/3)*np.sqrt(15*r-36*avg**2))
            A= Amp(self.pfc_params['avg'],self.pfc_params['r'])
            r = self.pfc_params["r"]
            if topo_info=="winding":
                 raise ValueError("Initializing directly with winding number is not implemented yet")
            elif topo_info=="burgers":
                self.zeta0.sub(0).interpolate(lambda x: initialize_from_burgers(qs,ps,list_defects,A,self.pfc_params['avg'])(x))

            else:
                 raise ValueError("Topological defects can either set by burgers vector or a winding number")
            self.psiout.interpolate(self.zeta0.sub(0))
            avg1= fem.assemble_scalar(fem.form(self.psiout*ufl.dx))/(self.pfc_params['L']*self.pfc_params['H'])
            print("After initing ", avg1)
            #Initialize the position dictionary
            # self.last_pos = [0,np.array(list_defects[0][:2]),np.array(list_defects[1][:2])]
            # self.pos.append(self.last_pos)

    def configure_solver(self)->None:
        if self.periodic:
            opts={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
            "ksp_reuse_preconditioner": True
                }
            self.problem_pfc = mpcLinearSolver(
                    self.a_pfc,
                    self.L_pfc,
                    self.pbcs,
                    bcs=[],
                    petsc_options=opts
                )
            self.problem_pfc.assembleBiLinear()
        else:
            self.A_pfc = fem.petsc.assemble_matrix(self.a_pfc, bcs=[])
            self.A_pfc.assemble()

            self.problem_pfc = PETSc.KSP().create(self.domain.comm)
            self.problem_pfc.setOperators(self.A_pfc)
            self.problem_pfc.setType(PETSc.KSP.Type.PREONLY)
            pc = self.problem_pfc.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setReusePreconditioner(True)

    def solve(self):
        if self.periodic:
            with dolfinx.common.Timer() as t_cpu:
                self.SH_sol = self.problem_pfc.solve()
                # print("PFC done in : %s" % t_cpu.elapsed()[0])
        else:
            with dolfinx.common.Timer() as t_cpu:
                b_pfc = fem.petsc.create_vector(self.L_pfc)
                with b_pfc.localForm() as loc_b:
                    loc_b.set(0)
                fem.petsc.assemble_vector(b_pfc, self.L_pfc)
                self.problem_pfc.solve(b_pfc, self.SH_sol.x.petsc_vec)
                print("PFC done in : %s" % t_cpu.elapsed()[0])

    

    def locate_defects(self,amp,proc,pad,t):
        ## Locate defects and append the position in the dictionnary 
        # ONLY if it's different from the previously saved position
        X_cropped = proc.X[pad:-1*pad, :]
        Y_cropped = proc.Y[pad:-1*pad, :]
        F = -1*np.abs(amp[pad:-1*pad, :])
        neighborhood_size = 10 
        filtered_F = ndimage.maximum_filter(F, size=neighborhood_size)
        peaks = np.where((F == filtered_F)) 
        peak_x = X_cropped[peaks]
        peak_y = Y_cropped[peaks]
        peak_values = F[peaks]
        same = True
        if len(peak_values) >= 2:
            if peak_x[0]<peak_x[1]:
                x1, y1 = peak_x[0], peak_y[0]
                x2, y2 = peak_x[1], peak_y[1]
            else:
                x1, y1 = peak_x[1], peak_y[1] 
                x2, y2 = peak_x[0], peak_y[0]
            current = np.array([np.array([x1,y1]),np.array([x2,y2])])
            same = np.allclose(current,self.last_pos[1:])
            if not same: 
                self.last_pos = [t,np.array([x1,y1]),np.array([x2,y2])]
                self.pos.append(self.last_pos)
        return same

    def correct(self):
        """ This is model dependent"""
        pass
    
    def relax(self,proc,Ue):
        n=0
        max_iter=100
        atol=1e-4
        rtol=1e-3
        res=np.inf
        alpha= self.coupling_params['Cw']*0.001
        while res > atol and res > rtol and n <= max_iter:
            avg1= fem.assemble_scalar(fem.form(self.psiout*ufl.dx))/(self.pfc_params['L']*self.pfc_params['H'])
            avg2= self.psiout.x.petsc_vec.dot(self.b_basis.x.petsc_vec)
            print("Currently", n," iters with ", res, "avg ", avg1,avg2)
            #################################
            ####Step 1 : Gradient descent####
            #################################
            amps= jnp.array([proc.C_Amp(jnp.array(self.psiout.x.array),i) for i in range(len(self.qs))])
            self.Q.x.array[:]= proc.Compute_Q(amps)
            self.Qsym.interpolate(fem.Expression(ufl.sym(self.Q),self.Q.function_space.element.interpolation_points()))
            self.dFQW.x.array[:] = proc.Analytical_gradFuq(self.Qsym.x.array,Ue.x.array,amps)
            self.psiout.x.petsc_vec.axpy(-1.0*alpha, self.dFQW.x.petsc_vec)
            self.psiout.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)  
            #################################
            #######Step 2 : Projection#######
            #################################
            beta = (self.psiout.x.petsc_vec.dot(self.b_basis.x.petsc_vec)-self.pfc_params['avg'])/self.b_basis_norm
            self.psiout.x.petsc_vec.axpy(-1.0*beta, self.b_basis.x.petsc_vec)
            self.psiout.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
            #################################
            ######Step 3 : Compute resi######
            #################################
            amps= jnp.array([proc.C_Amp(jnp.array(self.psiout.x.array),i) for i in range(len(self.qs))])
            self.Q.x.array[:]= proc.Compute_Q(amps)
            self.Qsym.interpolate(fem.Expression(ufl.sym(self.Q),self.Q.function_space.element.interpolation_points()))
            res= fem.assemble_scalar(fem.form(ufl.inner(self.Qsym-Ue,self.Qsym-Ue)*ufl.dx))
            n+=1
        print("Relaxed in ", n," iters with ", res)



    def update_cAmps(self,amps,order):
        for i in range(len(self.qs)):
            self.Re_amps_old[i].interpolate(self.Re_amps[i])
            self.Im_amps_old[i].interpolate(self.Im_amps[i])

        for i, (re, im) in enumerate(zip(self.Re_amps, self.Im_amps)):
            re.x.array[:] = np.real(amps[i].reshape(-1)[order].ravel())
            im.x.array[:] = np.imag(amps[i].reshape(-1)[order].ravel())
            re.x.scatter_forward()
            im.x.scatter_forward()

    def write_output(self,file,t,):
        file.write_function(self.psiout,t)
        file.write_function(self.dFQW,t)
        # file.write_function(self.alphaT,t)
        # file.write_function(self.J,t)
        # file.write_function(self.velocity,t)
        # file.write_function(self.micro_sigma,t)
        # file.write_function(self.micro_sigma_avg,t)
        if self.pfc_params['write_amps']:
            for i,q in enumerate(self.qs):
                file.write_function(self.Re_amps[i],t)
                file.write_function(self.Im_amps[i],t)
   
    def writepos(self):
        pos_array = np.array(self.pos, dtype=object)
        np.savetxt("positions.csv", pos_array, delimiter=",", fmt="%s")




    #########################################################################
    ###########################Field Computations############################
    #########################################################################