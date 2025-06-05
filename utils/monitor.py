import matplotlib.pyplot as plt
import numpy as np


class CVmonitor :
    def __init__(self):
        self.residuals=[]
        self.grad_norms = []
        self.its=[]
        self.set=False
        self.title=""
        self.n = 0
        self.solve_num=1
    def monitor(self,ksp,n,rnorm):
        self.residuals.append(rnorm)
        self.its.append(n)
        if not self.set:
            self.title= "field U "+str(ksp.getType())+" - "+str(ksp.getPC().getType())
            self.set= True

    def tao_monitor(self, tao):
        """Monitor for TAO iterations."""
        step = tao.getIterationNumber()
        fval = tao.getObjectiveValue()
        # gnorm = tao.getGradientNorm()
        # self.grad_norms.append(gnorm)
        print(f"[TAO] Iter {step}: Obj = {fval:.6e}")
        self.residuals.append(fval)
        self.its.append(step)
        if not self.set:
            self.title = f"TAO Monitor: {tao.getType()}"
            self.set = True

    def plot(self):
        plt.plot(self.its,self.residuals,marker="x",label="feval")
        # plt.plot(self.its,self.grad_norms,marker="x",label="gnorm")
        # plt.yscale("log")
        plt.title(self.title+" Solve num " + str(self.solve_num))
        plt.show()
        self.clear()

    def clear(self):
        self.residuals.clear()
        self.grad_norms.clear()
        self.its.clear()
        self.solve_num+=1

    def export(self):
        np.savetxt(str(self.n)+"_"+self.title+".csv",np.column_stack((self.its,self.residuals)),delimiter="\t")
        self.clear()