import numpy as np

from PostProcess.plotxdmf import *
import matplotlib.pyplot as plt

r       = 0.8            # cooling parameter
avg     = -0.43
Amp     = lambda avg,      r : (1/5)*(np.absolute(avg)+(1/3)*np.sqrt(15*r-36*avg**2)) #ground state amplitudes of a hexagonal lattice
lambda_ = 3*Amp(avg,r)**2, #lamé 1st coeff
mu      = 3*Amp(avg,r)**2,
nu      = 1/4
a0      = 4*np.pi/np.sqrt(3)

cws =[5]
path = "out/ESMC/Single0.1_5.h5"
X,Y,UE11,t = get_im_data(path,"UE",0,0)
print(X.shape)
# exit()
L=X.max()
H=Y.max()
print(H/2)
fig2, ax2 = plt.subplots(figsize=(3,3))
center_x_index = X.shape[1] // 2
# center_y_index = Y.shape[0] // 2-1
epsxx = (a0/(2*np.pi))*((3-2*nu)*(X-L/2)**2*(Y-H/2)+(1-2*nu)*(Y-H/2)**3)/(2*(nu-1)*((X-L/2)**2+(Y-H/2)**2)**2)
t=0
for cw in cws :
    path = "out/ESMC/Single0.1_"+str(cw)+".h5"
    X,Y,UE11,tt = get_im_data(path,"UE",t,0)
    X,Y,Q11,tt = get_im_data(path,"Q",t,0)
    xvals = Y[:,center_x_index]-H/2
    val_UE = UE11[:,center_x_index]
    val_Q = Q11[:,center_x_index]
    val_theory = epsxx[:,center_x_index]

    log_x = np.log10(xvals[:len(xvals)//2]*(-1.0))
    log_y = np.log10(val_UE[:len(xvals)//2])
    ax2.plot(log_x,log_y,lw=0.9,label=r"$\mathbf{U_e}^{{11}}$",color="red")

    log_yQ = np.log10(val_Q[:len(xvals)//2])
    ax2.plot(log_x,log_yQ,lw=0.9,label=r"$\mathbf{Q}^{{11}}$",color="blue")

    log_y_eps = np.log10(val_theory[:len(xvals)//2])
    # ax2.scatter(log_x,log_y_eps,marker="s",edgecolors="black",facecolor="none",label=r"Litt.",alpha=0.1)

    ax2.axvline(np.log10(a0/2),lw=0.9,color='black',ls="dashed")
    ax2.axvspan(0, np.log10(a0/2), color='royalblue', alpha=0.1)

    ax2.plot(log_x,-log_x+0.1,lw=0.9,ls="--",c="black",alpha=0.5)
    # ax2.plot(xvals[:len(xvals)//2],val_UE[:len(xvals)//2],lw=0.9,label=r"UE11",color="red")
    # ax2.plot(xvals[:len(xvals)//2],val_Q[:len(xvals)//2],lw=0.9,label=r"UE11",color="blue")

    # ax2.plot(xvals/a0,val_UE,lw=0.9,label=r"$\mathbf{U_e}^{{11}}$",color="red")
    # ax2.plot(xvals/a0,val_Q,lw=0.9,label=r"$\mathbf{Q}^{{11}}$",color="blue")
    # # ax2.scatter(xvals/a0,val_theory,marker="s",edgecolors="black",facecolor="none",label=r"Hirth",alpha=0.2)
    # # ax2.axvline(0)
    # ax2.set_xlim(-10,10)
    # ax2.set_ylim(-0.4,0.4)


path = "out/ESMC/Single0.1_"+str(0)+".h5"
X,Y,UE11,tt = get_im_data(path,"UE",t,0)
val_UE = UE11[:,center_x_index]
log_y = np.log10(val_UE[:len(xvals)//2])
ax2.plot(log_x,log_y,lw=0.9,label=r"$\mathbf{U_e}^{{11}}$",color="red",ls="--")

ax2.legend(frameon=False)
ax2.tick_params(
    axis='both',        # Apply to both x and y axes
    which='both',       # Both major and minor ticks
    direction='in',     # Ticks pointing inward
    top=True,           # Enable ticks on the top
    bottom=True,       # Disable ticks on the bottom
    left=True,          # Enable ticks on the left
    right=True,        # Disable ticks on the right
    labelsize=12         # Smaller font size for tick labels
)


ax2.set_xlabel(r"$\log d$",fontsize=14,fontweight='bold')
ax2.set_ylabel(r"$\log\, \mathbf{U_e}$, $\log\, \mathbf{Q}$",fontsize=14,fontweight='bold')
ax2.set_xlim(0.1,2.7)

ax2.legend(frameon=False)
ax2.text(1.4,      # mid‑ish x, nudged right
        -1.35,             # geometric‑mean y
        r'$\propto\;1/r$',
        rotation=-45, ha='left', va='center')

# ax2.text(1.4,      # mid‑ish x, nudged right
#         -1.35,             # geometric‑mean y
#         r'$a_0/2$',
#          ha='left', va='center')

# # --- pick two horizontal decades to span the triangle -----------------------
# x1, x2 = 1.3, 1.75          # r = 0.1 … 1  (one decade)
# y1, y2 = -1.22, -1.65       # f(r) at those points (also one decade)

# ax2.plot([x1, x2], [y1, y1], color='k', lw=0.9)
# ax2.plot([x2, x2], [y1, y2], color='k', lw=0.9)

# fig2.savefig("./figs/esmc/U11_log2.png",dpi=300,bbox_inches="tight")

plt.show()