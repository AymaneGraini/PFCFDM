import matplotlib.pyplot as plt
import numpy as np
import h5py
# from PostProcess.plotxdmf import get_im_data
import matplotlib.ticker as ticker
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def get_im_data(filename,fieldname,ti,comp=None):
    data1=None
    X,Y= None,None
    with h5py.File(filename, "r") as f:
        timekeys= list(f.keys())[1:]
        print(len(timekeys))
        sorted_timesteps = sorted(timekeys, key=lambda s: float(s.replace("Step_", "")))
        print("exporting data from ", sorted_timesteps[ti])
        print("ti = ", ti)
        if ti >= len(sorted_timesteps):
            raise ValueError("Requested timestep is not available, maximum snapshot is", len(sorted_timesteps)-1)
        print("tis is", ti)
        print(f[sorted_timesteps[ti]])
        Full_field = f[sorted_timesteps[ti]+"/"+fieldname+"/"]
        mesh= f["Geometry"]
        print("mesh ", mesh)
        x = f["Geometry/X"][:]
        y = f["Geometry/Y"][:]
        nx = len(np.unique(x))
        ny = len(np.unique(y))
        newx = np.linspace(x.min(),x.max(),nx)
        newy = np.linspace(y.min(),y.max(),ny)
        X,Y = np.meshgrid(newx,newy) 
        # mapping = np.lexsort((mesh[:, 0], mesh[:, 1]))

        if comp==None:
            field=Full_field[:]
        else:
            field=Full_field[:,comp]
            
        data1= field.reshape(ny, nx)
    timest = float(sorted_timesteps[ti].replace("Step_",""))
    return X,Y,data1, timest,len(sorted_timesteps)


class OneDecimalFormatter(ticker.ScalarFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(useMathText=True)
        self.set_scientific(True)
        self.set_powerlimits((0, 0))

    def _set_format(self):
        self.format = "%1.1f"

levels = MaxNLocator(nbins=100).tick_values(-0.2, 0.2)
cmap = plt.get_cmap('seismic')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

file = "/home/graini/Desktop/Codes/Package/FFT/out/coupled/debug/AmitJDipole_0.1_0_-0.8_0.h5"
for t in range(300):
    fig, ax = plt.subplots(2, 3, figsize=(17, 7))
    X,Y,Q00, timest,d= get_im_data(file,"Q_00",t)
    X,Y,UE00, timest,d= get_im_data(file,"Ue_00",t)
    X,Y,UP12, timest,d= get_im_data(file,"UP_01",t)

    X,Y,curlQ, timest,d= get_im_data(file,"curlQ",t)
    X,Y,curlUE, timest,d= get_im_data(file,"curlUe",t)
    X,Y,curlUP, timest,d= get_im_data(file,"curlUP",t)

    cf00 = ax[0,1].imshow(UE00,extent=[0,X.max(),0,Y.max()], cmap='seismic',vmin=-0.2,vmax=0.2)
    cb00 = fig.colorbar(cf00, ax=ax[0,1] ,ticks=np.linspace(-0.2, 0.2, 10),fraction=0.046, pad=0.04)
    cb00.formatter = OneDecimalFormatter()
    cb00.update_ticks()
    ax[0,1].set_title(r"${{U^e}}_{{00}}$",fontsize=14)


    # cf01 = ax[0,0].contourf(X, Y, Q00, levels=100, cmap='seismic')
    cf01 = ax[0,0].imshow(Q00,extent=[0,X.max(),0,Y.max()], cmap='seismic',vmin=-0.2,vmax=0.2)
    cb01 = fig.colorbar(cf01, ax=ax[0,0] ,ticks=np.linspace(-0.2, 0.2, 10),fraction=0.046, pad=0.04)
    cb01.formatter = OneDecimalFormatter()
    cb01.update_ticks()
    ax[0,0].set_title(r"$Q_{{00}}$",fontsize=14)


    cf02 = ax[0,2].imshow(UP12,extent=[0,X.max(),0,Y.max()], cmap='seismic',vmin=-0.6,vmax=0.3)
    cb02 = fig.colorbar(cf02, ax=ax[0,2] ,ticks=np.linspace(-0.6, 0.3, 10),fraction=0.046, pad=0.04)
    cb02.formatter = OneDecimalFormatter()
    cb02.update_ticks()
    ax[0,2].set_title(r"${{U^p}}_{{12}}$",fontsize=14)

    cf10 = ax[1,0].imshow(curlQ,extent=[0,X.max(),0,Y.max()], cmap='seismic',vmin=-0.4,vmax=0.4)
    cb10 = fig.colorbar(cf10, ax=ax[1,0],fraction=0.046, pad=0.04)
    cb10.formatter = OneDecimalFormatter()
    cb10.update_ticks()
    ax[1,0].set_title(r"$(\nabla\times Q)_{{13}}$",fontsize=14)

    cf11 = ax[1,1].imshow(curlUE,extent=[0,X.max(),0,Y.max()], cmap='seismic',vmin=-0.4,vmax=0.4)
    cb11 = fig.colorbar(cf11, ax=ax[1,1],fraction=0.046, pad=0.04)
    cb11.formatter = OneDecimalFormatter()
    cb11.update_ticks()
    ax[1,1].set_title(r"$(\nabla\times U^e)_{{13}}$",fontsize=14)

    cf12 = ax[1,2].imshow(curlUP,extent=[0,X.max(),0,Y.max()], cmap='seismic',vmin=-0.4,vmax=0.4)
    cb12 = fig.colorbar(cf12, ax=ax[1,2],fraction=0.046, pad=0.04)
    cb12.formatter = OneDecimalFormatter()
    cb12.update_ticks()
    ax[1,2].set_title(r"$(\nabla\times U^p)_{{13}}$",fontsize=14)

    fig.suptitle(r"Mechanical fields at $t={{{}}}$, pure SH run".format(int(timest)), fontsize=16)
    plt.tight_layout()
    fig.savefig("/home/graini/Desktop/Codes/Package/FFT/out/coupled/debug/MechanicalField_pureSH/MechanicalFields_at_t_"+str(t)+".png", dpi=600, bbox_inches='tight')
    plt.close()