from PostProcess.plotxdmf import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter


t=2
for c in [5]:
    cw=str(c)
    for t in [0,1]:
        print(cw,t)
        root="./out/Static/"
        file = root+"static0.1_"+cw+".h5"

        X,Y,UE11, timest = get_im_data(file,"UE",t,0)
        X,Y,UE12, timest = get_im_data(file,"UE",t,1)
        X,Y,UE21, timest = get_im_data(file,"UE",t,2)
        X,Y,UE22, timest = get_im_data(file,"UE",t,3)

        X,Y,Q11, timest = get_im_data(file,"Q",t,0)
        X,Y,Q12, timest = get_im_data(file,"Q",t,1)
        X,Y,Q21, timest = get_im_data(file,"Q",t,2)
        X,Y,Q22, timest = get_im_data(file,"Q",t,3)


        fig , ax = plt.subplots(1,3,figsize=(8,3))

        # Plot 1
        masked_data = np.ma.masked_where(np.abs(UE11) <= 1e-3*np.max(UE11), np.abs((UE11 - Q11))/ np.abs(UE11))
        # cf1 = ax[0].contourf(X, Y,Q11, levels=100, cmap="seismic")
        cf1 = ax[0].imshow(masked_data , origin="lower", cmap="seismic",vmin=0 , vmax=1)
        divider1 = make_axes_locatable(ax[0])
        cax1 = divider1.append_axes("right", size="3%", pad=0.05)
        cb1 = plt.colorbar(cf1, cax=cax1)

        # Apply scientific formatter with LaTeX
        fmt1 = ScalarFormatter(useMathText=True)
        fmt1.set_powerlimits((0,0))  # adjust to force sci notation if needed
        cb1.ax.yaxis.set_major_formatter(fmt1)
        ax[0].set_title(r"$Q_{11}$")
        # ax[0].set_title(r"$||(U^e_{11}-Q_{11})/U^e_{11}||$")
        # cb1.update_ticks()
        # cb1.ax.figure.canvas.draw()

        # Plot 2
        masked_data = np.ma.masked_where(np.abs(UE12 + UE21) <= 1e-5*np.max((UE12 + UE21)), np.abs((0.5 * (UE12 + UE21) -0.5 * (Q12 + Q21)))/np.abs((0.5 * (UE12 + UE21))))
        # cf2 = ax[1].contourf(X, Y,0.5 * (Q12 + Q21) , levels=100, cmap="seismic")
        cf2 = ax[1].imshow(masked_data , origin="lower", cmap="seismic",vmin=0 , vmax=1)
        divider2 = make_axes_locatable(ax[1])
        cax2 = divider2.append_axes("right", size="3%", pad=0.05)
        cb2 = plt.colorbar(cf2, cax=cax2)
        ax[1].set_title(r"$Q_{12,s}$")
        # ax[1].set_title(r"$||(U^e_{12,s}-Q_{12,s})/U^e_{12,s}||$")
        fmt2 = ScalarFormatter(useMathText=True)
        fmt2.set_powerlimits((0,0))
        cb2.ax.yaxis.set_major_formatter(fmt2)
        # cb2.update_ticks()
        # cb2.ax.figure.canvas.draw()

        # Plot 3
        masked_data = np.ma.masked_where(np.abs(UE22)<= 1e-5*np.max(UE22), np.abs((UE22-Q22))/np.abs(UE22))
        # cf3 = ax[2].contourf(X, Y,Q22, levels=100, cmap="seismic")
        cf3 = ax[2].imshow(masked_data , origin="lower", cmap="seismic",vmin=0 , vmax=1)
        divider3 = make_axes_locatable(ax[2])
        cax3 = divider3.append_axes("right", size="3%", pad=0.05)
        cb3 = plt.colorbar(cf3, cax=cax3)
        fmt3 = ScalarFormatter(useMathText=True)
        fmt3.set_powerlimits((0,0))
        cb3.ax.yaxis.set_major_formatter(fmt3)
        # ax[2].set_title(r"$||(U^e_{22}-Q_{22})/U^e_{22}||$")
        ax[2].set_title(r"$Q_{22}$")

        # cb3.update_ticks()
        # cb3.ax.figure.canvas.draw()

        # fig.tight_layout()
        # if t==1:
        #     fig.savefig(root+"ContourQ_"+cw+"inital.png",dpi=300,bbox_inches='tight')
        # elif t==2:
        #     fig.savefig(root+"ContourQ_"+cw+"final.png",dpi=300,bbox_inches='tight')
        # plt.close()
    plt.show()