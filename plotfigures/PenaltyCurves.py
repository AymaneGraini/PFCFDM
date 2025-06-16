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

cws=[1]
colors = blue_red(np.linspace(0, 1, len(cws))) 
figsh, ax_sh = plt.subplots(figsize=(4.5, 4))

figD, ax_totalD = plt.subplots(figsize=(4.5, 4))
axins = inset_axes(ax_sh, width="60%", height="30%", 
                   bbox_transform=ax_sh.transAxes, 
                   bbox_to_anchor=(0.3, 0.3, 1, 1), 
                   loc='lower left')
figuq, ax2 = plt.subplots(figsize=(4, 4))
figuq, axmec = plt.subplots(figsize=(4, 4))
# figuq11, ax11 = plt.subplots(figsize=(4, 4))
# figuq12, ax12 = plt.subplots(figsize=(4, 4))
# figuq21, ax21 = plt.subplots(figsize=(4, 4))
# figuq22, ax22 = plt.subplots(figsize=(4, 4))

xzoom1 =230
xzoom2 =600
root = "./out/indentation/"
for i, cw in enumerate(cws):
    dataSH = np.loadtxt(root+"Energy_indentation0.1_"+str(cw)+".csv",delimiter="\t")
    dt = dataSH[1,0] - dataSH[0,0]
    ax_sh.plot(dataSH[:,0]/dt,dataSH[:,1]-dataSH[0,1],lw=0.9,label=r"$C_w={{{}}}$".format(cw),color=colors[i],marker="x")

    axmec.plot(dataSH[:,0]/dt,dataSH[:,2],lw=0.9,label=r"$C_w={{{}}}$".format(cw),color=colors[i],marker="x")
    # axD.plot(dataSH[:,0]/dt,dataSH[:,2],lw=0.9,label=r"$C_w={{{}}}$".format(cw),color=colors[i])

    datauq = np.loadtxt(root+"errors_indentation0.1_"+str(cw)+".csv",delimiter="\t")
    l2 = np.sqrt(datauq[:,1]**2 +datauq[:,2]**2 +datauq[:,3]**2 +datauq[:,4]**2)
    ax2.plot(datauq[:,0]/dt,l2,lw=0.9,label=r"$C_w={{{}}}$".format(cw),color=colors[i],marker="x")


    D = -np.gradient(dataSH[:,1],dataSH[:,0]) -0.5*cw *np.pad(np.gradient(l2,datauq[:,0]),(len(dataSH[:,1])-len(l2),0),mode='constant')+dataSH[:,2]
    ax_totalD.plot(dataSH[:,0]/dt,D,lw=0.9,label=r"$C_w={{{}}}$".format(cw),color=colors[i])
    # ax11.plot(datauq[:,0]/dt,datauq[:,1],lw=0.9,label=r"$C_w={{{}}}$".format(cw),color=colors[i])
    # ax12.plot(datauq[:,0]/dt,datauq[:,2],lw=0.9,label=r"$C_w={{{}}}$".format(cw),color=colors[i])
    # ax21.plot(datauq[:,0]/dt,datauq[:,3],lw=0.9,label=r"$C_w={{{}}}$".format(cw),color=colors[i])
    # ax22.plot(datauq[:,0]/dt,datauq[:,4],lw=0.9,label=r"$C_w={{{}}}$".format(cw),color=colors[i])


    axins.plot(dataSH[:,0]/dt,dataSH[:,1]-dataSH[0,1],lw=0.9,color=colors[i])
    axins.set_xlim(xzoom1,xzoom2)


# plt.show()
ylm = dataSH[-1,0]/dt

# mark_inset(ax, axins, loc1=2, loc2=4, fc="white", ec="0.4",
        #    linestyle='--',linewidth=0.7)
mask = (dataSH[:,0]/dt >= xzoom1) & (dataSH[:,0]/dt <= xzoom2)
y_zoom = dataSH[mask,1]

axins.set_ylim(-76, -75.2)
# axins.set_ylim(-182, -180.5)
ax_sh.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(dt),fontsize=14)    
ax_sh.set_ylabel(r"Swift Hohenberg Energy $\mathcal{F}_{sh} -\mathcal{F}_0$",fontsize=14)

ax2.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(dt),fontsize=14)    
ax2.set_ylabel(r"$||\mathbf{U^e_{sym}}-\mathbf{Q_{sym}}||_{L_2}$",fontsize=14)    

ax_totalD.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(dt),fontsize=14)    
ax_totalD.set_ylabel(r"Total Dissipation $\mathcal{D}$",fontsize=14)    

axmec.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(dt),fontsize=14)    
axmec.set_ylabel(r"Plastic Dissipation $\mathcal{D}$",fontsize=14) 

# ax11.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(dt),fontsize=14)    
# ax11.set_ylabel(r"$||\mathbf{U^e_{11}}-\mathbf{Q_{11}}||_{L_2}$",fontsize=14)   

# ax12.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(dt),fontsize=14)    
# ax12.set_ylabel(r"$||\mathbf{U^e_{12,s}}-\mathbf{Q_{12,s}}||_{L_2}$",fontsize=14)   

# ax21.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(dt),fontsize=14)    
# ax21.set_ylabel(r"$||\mathbf{U^e_{21,s}}-\mathbf{Q_{21,s}}||_{L_2}$",fontsize=14) 

# ax22.set_xlabel(r"Iterations, $dt={{{}}} \, [-]$".format(dt),fontsize=14)    
# ax22.set_ylabel(r"$||\mathbf{U^e_{22}}-\mathbf{Q_{22}}||_{L_2}$",fontsize=14)    

formatter2 = SfFormatter(useMathText=True)
formatter2.set_powerlimits((1, 0))
axins.yaxis.set_major_formatter(formatter2)


formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((1, 0))
ax_sh.yaxis.set_major_formatter(formatter)
ax_sh.set_xlim(None,ylm)
ax_sh.axvspan(0, 250, color='royalblue', alpha=0.1) 
axins.axvspan(0, 250, color='royalblue', alpha=0.1) 
ax2.set_xlim(None,ylm)

# ax11.set_xlim(None,ylm)
# ax12.set_xlim(None,ylm)
# ax21.set_xlim(None,ylm)
# ax22.set_xlim(None,ylm)

ax_sh.legend(frameon=False)
ax_sh.axvline(x=250,lw=0.5,color="black",ls="dashed")
axins.axvline(x=250,lw=0.5,color="black",ls="dashed")
ax_sh.text(200,-20,r"Relaxtion of $\psi$ with $\mathcal{F}_{sh}$ only",rotation=90, va='center', ha='center')
ax2.legend(frameon=False) 

# ax11.legend(frameon=False) 
# ax12.legend(frameon=False) 
# ax21.legend(frameon=False) 
# ax22.legend(frameon=False) 


# figsh.savefig(root+"Swifth-Hohenberg energy.png",dpi=300,bbox_inches='tight')
# figD.savefig(root+"Total Dissipation.png",dpi=300,bbox_inches='tight')
# figuq.savefig(root+"UQ difference.png",dpi=300,bbox_inches='tight')
# figuq11.savefig(root+"UQ11 difference.png",dpi=300,bbox_inches='tight')
# figuq12.savefig(root+"UQ12 difference.png",dpi=300,bbox_inches='tight')
# # figuq21.savefig("UQ21 difference.png",dpi=300,bbox_inches='tight')
# figuq22.savefig(root+"UQ22 difference.png",dpi=300,bbox_inches='tight')
plt.show()


