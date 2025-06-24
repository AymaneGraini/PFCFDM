"""
Mechanics IO class for writing output in XDMF format.
This class is used to write various mechanical indicators and fields to an XDMF file.
Can also save indicators to a CSV file, namely the different energies, divergence norms, and averages.
"""

import dolfinx.io

import numpy as np

class MecIO:
    """
        A Mechanics related class, used to write output in XDMF
        Also writes different relevant indicators, see below.
        The indicators include:
            - FSH: Swify-Hohenberg free energy
            - FUQ: l2 NORM of the difference between the symetric parts of the elastic distortion and configurational distortion
            - L2divs: L2 norm of the divergence of the elastic stress
            - L2divsQ: L2 norm of the divergence of the configurational stress given by :math:`\\sigma_Q = \\mathbb{C}:\\mathbf{Q}`.
            - avgs: Average value of the order parameter :math:`\\psi` over the domain.
        Args:
            file (dolfinx.io.XDMFFile): The XDMF file to write output to.
            mecFe (MecFE): The finite element part of the mechanics problem.
            mecComp (MecComp): The mechanics computation part, containing auxiliary fields and computations.
    """
    def __init__(self,file : dolfinx.io.XDMFFile,mecFe,mecComp):
        self.file = file
        self.mecComp=mecComp
        self.mecFe=mecFe

    def write_output(self ,t:float):
        """
            Function to write output to a file in XDMF format.
            

            Args:
                t (float): The current time step, used for naming the output.
        """
        self.file.write_function(self.mecFe.u_out,t)
        self.file.write_function(self.mecFe.UE,t)
        self.file.write_function(self.mecComp.sigmaUe,t)
        # self.file.write_function(self.mecComp.divsUe,t)

        self.file.write_function(self.mecFe.Q,t)
        # self.file.write_function(self.Qsym,t)
        self.file.write_function(self.mecFe.alpha,t)
        # self.file.write_function(self.mecFe.UPperp,t)
        # self.file.write_function(self.UPpara,t)
        self.file.write_function(self.mecFe.UP,t)
        # self.file.write_function(self.U,t)
        # self.file.write_function(self.UEsym,t)
        # self.file.write_function(self.epsilon_psi,t)
        # self.file.write_function(self.sigmaQ,t)
        # self.file.write_function(self.divsQ,t)
        # self.file.write_function(self.divUP,t)
        # self.file.write_function(self.sigmaUp,t)
        # self.file.write_function(self.div_C_UP,t)
        self.file.write_function(self.mecComp.curlUP,t)
        self.file.write_function(self.mecComp.curlUE,t)
        self.file.write_function(self.mecComp.curlQ,t)
        # # self.file.write_function(self.out,t)
        # self.file.write_function(self.mecComp.V_pk,t)

    def save_indicators(self,prefix: str):
        """
        Args:
            prefix (str): the path prefix to save the indicators to.

        Saves the following indicators to a CSV file:
        """
        np.savetxt(prefix+self.suffix+".csv",np.column_stack((self.FSH,self.FUQ,self.L2divs,self.L2divsQ,self.avgs)),delimiter="\t")