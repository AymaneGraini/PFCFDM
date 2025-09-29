'''
    Plots the curves of the swift Hohenberg energy evolution and the difference of U-Q 
    Used for the meeting with JOrge and AMit in 10 JUne 2025
'''


import matplotlib.pyplot as plt
import numpy as np
# import h5py
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

class SfFormatter(ScalarFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_format(self):
        self.format = '%.2f' if self._useMathText else '%1.2f'

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.loc'] = 'upper right'

blue_red = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])

cws=[0,1,2,5]
print("yo")

figuq, ax2 = plt.subplots(figsize=(4, 4))

# figuq11, ax11 = plt.subplots(figsize=(4, 4))
# figuq12, ax12 = plt.subplots(figsize=(4, 4))
# figuq21, ax21 = plt.subplots(figsize=(4, 4))
# figuq22, ax22 = plt.subplots(figsize=(4, 4))

xzoom1 =200
xzoom2 =600
root = "./out/Static/"
filename="staticPer_noCrop0.1"
bases=["LM","Field"]
colors = blue_red(np.linspace(0, 1, len(cws))) 
for i, cw in enumerate(cws):
    datauq = np.loadtxt(root+"errors_"+filename+"_"+str(cw)+".csv",delimiter="\t")
    l2 = np.sqrt(datauq[:,1]**2 +datauq[:,2]**2 +datauq[:,3]**2 +datauq[:,4]**2)
    ax2.plot(datauq[:,0],l2,lw=0.9,label=r"$C_w={{{}}}$".format(cw),color=colors[i])



# plt.show()


# ax_sh.set_ylim(-300, -250)

ax2.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(0.1),fontsize=14)    
ax2.set_ylabel(r"$||\mathbf{U^e_{sym}}-\mathbf{Q_{sym}}||_{L_2}$",fontsize=14)    
# ax2.set_ylim(None,7.5)    


# ax11.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(dt),fontsize=14)    
# ax11.set_ylabel(r"$||\mathbf{U^e_{11}}-\mathbf{Q_{11}}||_{L_2}$",fontsize=14)   

# ax12.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(dt),fontsize=14)    
# ax12.set_ylabel(r"$||\mathbf{U^e_{12,s}}-\mathbf{Q_{12,s}}||_{L_2}$",fontsize=14)   

# ax21.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(dt),fontsize=14)    
# ax21.set_ylabel(r"$||\mathbf{U^e_{21,s}}-\mathbf{Q_{21,s}}||_{L_2}$",fontsize=14) 

# ax22.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(dt),fontsize=14)    
# ax22.set_ylabel(r"$||\mathbf{U^e_{22}}-\mathbf{Q_{22}}||_{L_2}$",fontsize=14)    



ax2.set_xlim(None,500   )


# ax11.set_xlim(None,ylm)
# ax12.set_xlim(None,ylm)
# ax21.set_xlim(None,ylm)
# ax22.set_xlim(None,ylm)

# ax_sh.axvline(x=250,lw=0.5,color="black",ls="dashed")
# axins.axvline(x=250,lw=0.5,color="black",ls="dashed")
# ax_sh.text(200,-20,r"Relaxtion of $\psi$ with $\mathcal{F}_{sh}$ only",rotation=90, va='center', ha='center')
ax2.legend(frameon=False) 



# ax11.legend(frameon=False) 
# ax12.legend(frameon=False) 
# ax21.legend(frameon=False) 
# ax22.legend(frameon=False) 


# figsh.savefig(root+"Swifth-Hohenberg energy.png",dpi=300,bbox_inches='tight')
# figD.savefig(root+"Total Dissipation.png",dpi=300,bbox_inches='tight')
# figuq.savefig(root+"UQ difference.png",dpi=300,bbox_inches='tight')
# figmec.savefig(root+"Plastic difference.png",dpi=300,bbox_inches='tight')
# figuq11.savefig(root+"UQ11 difference.png",dpi=300,bbox_inches='tight')
# figuq12.savefig(root+"UQ12 difference.png",dpi=300,bbox_inches='tight')
# # figuq21.savefig("UQ21 difference.png",dpi=300,bbox_inches='tight')
# figuq22.savefig(root+"UQ22 difference.png",dpi=300,bbox_inches='tight')
plt.show()


