import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



def get_unique(x, y, ymin, ymax):
    df = pd.DataFrame({'x': x, 'y': y})
    df_unique = df.drop_duplicates(subset='y', keep='first')

    # Filter by range
    df_filtered = df_unique[(df_unique['y'] >= ymin) & (df_unique['y'] <= ymax)]

    x_unique = df_filtered['x'].values
    y_unique = df_filtered['y'].values
    return x_unique, y_unique


s=(3.5,3.5)
fig0, ax0 = plt.subplots(figsize=s)
fig0y, ax0y = plt.subplots(figsize=s)
fig1, ax1 = plt.subplots(figsize=s)
fig2, ax2 = plt.subplots(figsize=s)
fig3, ax3 = plt.subplots(figsize=s)

cws=[0]
order=[4,6]
blue_red = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])

colors = blue_red(np.linspace(0, 1, len(order))) 

root="./out/Annihilation/4vs6/r1.4/"
for i,o in enumerate(order):

    data = np.loadtxt(root+"Annihilation"+str(o)+"th1.40.1_0.csv",dtype=float,delimiter="\t")
    x1s=data[:,1]
    y1s=data[:,2]
    x2s=data[:,3]
    y2s=data[:,4]
    dt = 20*0.1
    vx = np.diff(x1s)  /dt
    vy = np.diff(y1s) /dt
    d = data[:,5]
    tt = data[:,0]*20*0.1
    #Plot position vs Time
    ax0.scatter(tt,x1s,s=25,marker="o",linewidths=0.5,color="none",edgecolors=colors[i],alpha=0.05)
    ax0y.scatter(tt,y1s,s=25,marker="o",linewidths=0.5,color="none",edgecolors=colors[i],alpha=0.05)
    ax0y.scatter(tt,y2s,s=25,marker="o",linewidths=0.5,color="none",edgecolors=colors[i],alpha=0.05)
    xu2,yu2= get_unique(tt,x1s,100,250)
#     ax0.scatter(xu2,yu2,s=25,marker="x",linewidths=0.5,color=colors[i],alpha=1)
#     ax0.plot(xu,yu,color=colors[i],label=r"FEM position $C_w={{{}}}$".format(cw))

    ax0.scatter(tt,x2s,s=25,marker="o",linewidths=0.5,color="none",edgecolors=colors[i],alpha=0.05)
    xu1,yu1= get_unique(tt,x2s,100,250)

    x_combined = np.concatenate((xu1, np.flip(xu2)))
    y_combined = np.concatenate((yu1, np.flip(yu2)))
    np.savetxt('tst.csv', np.column_stack((x_combined,y_combined)),delimiter="\t")
    ax0.plot(xu1,yu1,linewidth=1,color=colors[i],alpha=1,label=r"$C_w={{{}}}$".format(o))

    ax0.plot(xu2,yu2,linewidth=1,color=colors[i],alpha=1)

#     ax0.plot(xu,yu,color=colors[i])

    ts,xs2= get_unique(tt,x2s,100,250)
    vx1 = np.gradient(xs2, ts)

    ind = np.where(np.isin(tt, ts))[0]

    xs1 = x1s[ind]
    d= np.abs(xs2-xs1)

#     ax1.scatter(ts,d,color=colors[i],marker="x", s=25,  linewidths=0.8)
    d0 = d[0]
#     def model(t, n, A):
#         term = (n + 1) * (-2 * A * t + (d0**(
#             n + 1)) / (n + 1))
#         return np.where(term > 0, term ** (1 / (n + 1)), np.nan)
    
#     p0 = [1.0, 1.0]
#     bounds = ([0.0, 0.000], [10, 1000])
#     params, covariance = curve_fit(model, ts, d, p0=p0,maxfev=10000,bounds=bounds)
#     t_fit = np.linspace(min(ts), max(ts), 500)
#     d_fit = model(t_fit, *params)
#     print( cw, params[0])
    ax1.plot(ts, d,color=colors[i],lw=0.9,label=r"$C_w={{{}}}$".format(o))

    ax2.plot(ts,vx1,color=colors[i],lw=0.9,label=r"Velocity $C_w={{{}}}$".format(o))
    d=d[:-1]
    vx1=vx1[:-1]
    ax3.plot(d,vx1,color=colors[i],lw=0.9,label=r"Velocity $C_w={{{}}}$".format(o))
    # ax3.scatter(d,vx1,s=25,marker="x",linewidths=0.5,color=colors[i],edgecolors=colors[i],label=r"Velocity vs dist")


# # A_fit, n_fit, B_fit, C_fit, phi_fit
# p0 = (1.5,1,0.1,0.3,4)
# popt = fit_vd(d[:-1],vx1[:-1],p0)
# print("fitted are ", popt)

# dfit=np.linspace(d[-1],d[0],200)
# vfit = PK_PN(dfit,*popt)
# # exit()
# ax3.plot(dfit,vfit, color="blue",lw=0.9,label=r"Model")



ax3.set_yscale("log")
ax3.set_xscale("log")

ax3.legend(frameon=False,prop={'size':10})
ax3.tick_params(
        axis='both',        # Apply to both x and y axes
        which='both',       # Both major and minor ticks
        direction='in',     # Ticks pointing inward
        top=True,           # Enable ticks on the top
        bottom=True,       # Disable ticks on the bottom
        left=True,          # Enable ticks on the left
        right=True,        # Disable ticks on the right
        labelsize=12)
ax3.margins(0,0)
ax3.set_xlabel(r"Distance",fontsize=14)
ax3.set_ylabel(r"Velocity",fontsize=14)


ax2.legend(frameon=False,prop={'size':10})
ax2.tick_params(
        axis='both',        # Apply to both x and y axes
        which='both',       # Both major and minor ticks
        direction='in',     # Ticks pointing inward
        top=True,           # Enable ticks on the top
        bottom=True,       # Disable ticks on the bottom
        left=True,          # Enable ticks on the left
        right=True,        # Disable ticks on the right
        labelsize=12)
ax2.margins(0,0)
ax2.set_xlabel(r"Time",fontsize=14)
ax2.set_ylabel(r"Velocity",fontsize=14)
ax2.set_yscale("log")




ax1.legend(frameon=False,prop={'size':10})
ax1.tick_params(
        axis='both',        # Apply to both x and y axes
        which='both',       # Both major and minor ticks
        direction='in',     # Ticks pointing inward
        top=True,           # Enable ticks on the top
        bottom=True,       # Disable ticks on the bottom
        left=True,          # Enable ticks on the left
        right=True,        # Disable ticks on the right
        labelsize=12)
ax1.margins(0,0)
ax1.set_xlabel(r"Time",fontsize=14)
ax1.set_ylabel(r"Distance",fontsize=14)
# ax1.set_yscale("log")
# ax1.set_xscale("log")


ax0.legend(frameon=False,prop={'size':10})
ax0.tick_params(
        axis='both',        # Apply to both x and y axes
        which='both',       # Both major and minor ticks
        direction='in',     # Ticks pointing inward
        top=True,           # Enable ticks on the top
        bottom=True,       # Disable ticks on the bottom
        left=True,          # Enable ticks on the left
        right=True,        # Disable ticks on the right
        labelsize=12)
ax0.margins(0,0)
ax0.set_xlabel(r"Time",fontsize=14)
ax0.set_ylabel(r"Dislocation $x$ Coordinate",fontsize=14)
ax0.set_ylim(130,230)
ax0.set_xlim(0,2000)


ax0y.legend(frameon=False,prop={'size':10})
ax0y.tick_params(
        axis='both',        # Apply to both x and y axes
        which='both',       # Both major and minor ticks
        direction='in',     # Ticks pointing inward
        top=True,           # Enable ticks on the top
        bottom=True,       # Disable ticks on the bottom
        left=True,          # Enable ticks on the left
        right=True,        # Disable ticks on the right
        labelsize=12)
ax0y.margins(0,0)
ax0y.set_xlabel(r"Time",fontsize=14)
ax0y.set_ylabel(r"Dislocations $x$ Coordinate",fontsize=14)
# ax0y.set_ylim(130,230)
# ax0y.set_xlim(0,1200)


fig0.savefig(root+"fig0.png", bbox_inches='tight',dpi=300)
# fig1.savefig("./motionpng/fig1.png", bbox_inches='tight',dpi=300)
# fig2.savefig("./motionpng/fig2.png", bbox_inches='tight',dpi=300)
# fig3.savefig("./motionpng/fig3.png", bbox_inches='tight',dpi=300)
plt.show()