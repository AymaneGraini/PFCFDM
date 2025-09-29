import matplotlib.pyplot as plt

from PostProcess.plotxdmf import get_im_data
import matplotlib.ticker as ticker


class OneDecimalFormatter(ticker.ScalarFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(useMathText=True)
        self.set_scientific(True)
        self.set_powerlimits((0, 0))

    def _set_format(self):
        self.format = "%1.1f"


file = "out/trash/ShearCompFFT0.1_0_.h5"

fig, ax = plt.subplots(2, 3, figsize=(17, 7))

t=100
X,Y,Q00, timest,d= get_im_data(file,"Q",t,comp=0)
X,Y,UE00, timest,d= get_im_data(file,"UE",t,comp=0)
X,Y,UP12, timest,d= get_im_data(file,"UP",t,comp=1)

X,Y,curlQ, timest,d= get_im_data(file,"alpha",t,comp=2)
X,Y,curlUE, timest,d= get_im_data(file,"curlUe",t,comp=2)
X,Y,curlUP, timest,d= get_im_data(file,"curlUp",t,comp=2)

cf00 = ax[0,1].contourf(X, Y, UE00, levels=100, cmap='seismic')
cb00 = fig.colorbar(cf00, ax=ax[0,1])
cb00.formatter = OneDecimalFormatter()
cb00.update_ticks()
ax[0,1].set_title(r"${{U^e}}_{{xx}}$",fontsize=14)

cf01 = ax[0,0].contourf(X, Y, Q00, levels=100, cmap='seismic')
cb01 = fig.colorbar(cf01, ax=ax[0,0])
cb01.formatter = OneDecimalFormatter()
cb01.update_ticks()
ax[0,0].set_title(r"$Q_{{xx}}$",fontsize=14)

cf02 = ax[0,2].contourf(X, Y, UP12, levels=100, cmap='seismic')
cb02 = fig.colorbar(cf02, ax=ax[0,2])
cb02.formatter = OneDecimalFormatter()
cb02.update_ticks()
ax[0,2].set_title(r"${{U^p}}_{{xy}}$",fontsize=14)

cf10 = ax[1,0].contourf(X, Y, curlQ, levels=100, cmap='seismic')
cb10 = fig.colorbar(cf10, ax=ax[1,0])
cb10.formatter = OneDecimalFormatter()
cb10.update_ticks()
ax[1,0].set_title(r"$(\nabla\times Q)_{{xz}}$",fontsize=14)

cf11 = ax[1,1].contourf(X, Y, curlUE, levels=100, cmap='seismic')
cb11 = fig.colorbar(cf11, ax=ax[1,1])
cb11.formatter = OneDecimalFormatter()
cb11.update_ticks()
ax[1,1].set_title(r"$(\nabla\times U^e)_{{xz}}$",fontsize=14)

cf12 = ax[1,2].contourf(X, Y, curlUP, levels=100, cmap='seismic')
cb12 = fig.colorbar(cf12, ax=ax[1,2])
cb12.formatter = OneDecimalFormatter()
cb12.update_ticks()
ax[1,2].set_title(r"$(\nabla\times U^p)_{{xz}}$",fontsize=14)

fig.suptitle(r"Mechanical fields at $t={{{}}}$, pure SH run".format(timest), fontsize=16)
plt.tight_layout()
fig.savefig("out/trash/SH_annihilation_"+str(t)+".png", dpi=300, bbox_inches='tight')
plt.show()