from PostProcess.plotxdmf import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
from scipy.signal import find_peaks
import scipy.ndimage as ndimage
from mpi4py import MPI
from dolfinx.io import XDMFFile
from PFCproc_TODO.ProcessPFC_padFFT import *

a0=4*np.pi/np.sqrt(3)
qs=[
    [0,1],
    [np.sqrt(3)/2,-1/2],
    [-np.sqrt(3)/2,-1/2],
    [0,-1],
    [-np.sqrt(3)/2,1/2],
    [np.sqrt(3)/2,1/2]
]



root='./out/Annihilation/4vs6/'
cws=[0]
for cw in cws:
    base = "Annihilationth1.40.1_"
    filenameh5=root+base+str(cw)+".h5"
    filenamexdmf=root+base+str(cw)+".xdmf"
    print("file name", filenameh5)
    with XDMFFile(MPI.COMM_WORLD, filenamexdmf, "r") as xdmf:
        domain      = xdmf.read_mesh()
    print('MEsh read')
    X,Y,data1,t,tmax= get_im_data(filenameh5,"Psi",0)
    mesh_coords = domain.geometry.x
    mapping     = np.lexsort((mesh_coords[:, 0], mesh_coords[:, 1]))
    rev_DofMap = np.empty_like(mapping)

    rev_DofMap[mapping] = np.arange(len(mapping))

    Processor   = PFCProcessor(domain.geometry.x,DofMap=mapping,qs=qs,a0=a0,target_shape=X.shape,pads=(0,0))
    
    print("ciao")
    comm = MPI.COMM_WORLD
    N=comm.size
    ts= np.arange(0,tmax,1)
    file_sublists= np.array_split(ts, N)
    ind = 0
    k=comm.rank
    dists=[]
    times=[]
    x1s=[]
    x2s=[]
    y1s=[]
    y2s=[]
    d_prev = np.inf
    for tt in file_sublists[k]:
        print(tt)
        comm.Barrier()
        X,Y,data1,t,tmax= get_im_data(filenameh5,"Psi",tt)
        comm.Barrier()
        L = X.max() - X.min()
        H = Y.max() - Y.min()
        amp= np.abs(Processor.C_Amp(data1.ravel()[rev_DofMap],2).reshape(X.shape))
        X_cropped = X[10:-10, :]
        Y_cropped = Y[10:-10, :]
        F = -1*amp[10:-10, :]
        neighborhood_size = 5  
        filtered_F = ndimage.maximum_filter(F, size=neighborhood_size)
        peaks = np.where((F == filtered_F))  
        peak_x = X_cropped[peaks]
        peak_y = Y_cropped[peaks]
        peak_values = F[peaks]
        peak_coords = np.argwhere(peaks)  
        sorted_indices = np.argsort(peak_values)
        peak_x = peak_x[sorted_indices]
        peak_y = peak_y[sorted_indices] 

        if len(peak_values) >= 2:
            List_x = np.array([peak_x[-1],peak_x[-2]])
            List_y = np.array([peak_y[-1],peak_y[-2]])
            sorted_x = np.argsort(List_x)
            print("sorted x = ", sorted_x)
            List_x= List_x[sorted_x]
            List_y= List_y[sorted_x]
            x1, y1 = List_x[-1], List_y[-1]
            x2, y2 = List_x[-2], List_y[-2]
            dist=np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # m = np.absolute((y2-y1)/(x2-x1))
            # angle = np.arctan(m)*180/(np.pi)
            # padding = 0.1 *dist   # 10% of line length
            # mid_x = (x2 + x1) / 2
            # mid_y = (y2 + y1) / 2
            # dx = padding * np.cos(np.arctan(m))
            # dy = padding * np.sin(np.arctan(m))

            # text_x = mid_x + dx
            # text_y = mid_y + dy

            if False: #dist < d_prev : 
                fig,axs = plt.subplots(2,1,figsize=(12,8))
                axs[0].contourf(X,Y,data1,levels=200,cmap="seismic",vmin=-1.4,vmax=1.4)
                axs[1].contourf(X,Y,data2,levels=200,cmap="seismic",vmin=-0.5,vmax=0.5)
                # axs[1].plot([x1,x2], [y1,y2], color='black', lw=0.9)  # Top two peaks
                # axs[1].text(text_x,text_y,s=r"$d={{{:.5f}}}$".format(dist),rotation=angle, va='bottom', ha='center')
                # plt.show()
                plt.tight_layout()
                fig.savefig("./motionpng/NL_dip_"+str(ind)+".png", bbox_inches='tight',dpi=300)
                plt.clf()
                plt.close()
                # d_prev = dist
                ind+=1
        else:
            peak1 = peak_coords[0]
            if len(x1s)>0:
                x1, y1 = x1s[-1], y1s[-1]
            else:
                x1, y1 = 0,0

            x2, y2  =  x1, y1
            m = 0
            dist=0

        dists.append(dist)
        times.append(tt)
        x1s.append(x1)
        x2s.append(x2)
        y1s.append(y1)
        y2s.append(y2)
        # plt.show()
        # exit()
        # exit()

    np.savetxt(root+base+str(cw)+".csv", np.column_stack((times,x1s,y1s,x2s,y2s,dists,)),delimiter="\t")


