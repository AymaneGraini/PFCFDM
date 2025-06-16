import dolfinx.io

import numpy as np

class MecIO:
    """
        A Mechanics related class, used to write output in XDMF
        Also writes different relevant indicators, see below.
    """
    def __init__(self,file : dolfinx.io.XDMFFile,mecFe,mecComp):
        self.file = file
        self.mecComp=mecComp
        self.mecFe=mecFe

    def write_output(self ,t:float):
        """
            Function to write output to a file 
        """
        self.file.write_function(self.mecFe.u_out,t)
        self.file.write_function(self.mecFe.UE,t)
        self.file.write_function(self.mecComp.sigmaUe,t)
        self.file.write_function(self.mecComp.divsUe,t)

        self.file.write_function(self.mecFe.Q,t)
        # self.file.write_function(self.Qsym,t)
        self.file.write_function(self.mecFe.alpha,t)
        self.file.write_function(self.mecFe.UPperp,t)
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
        # self.file.write_function(self.V_pk,t)

    def save_indicators(self,prefix: str):
        np.savetxt(prefix+self.suffix+".csv",np.column_stack((self.FSH,self.FUQ,self.L2divs,self.L2divsQ,self.avgs)),delimiter="\t")