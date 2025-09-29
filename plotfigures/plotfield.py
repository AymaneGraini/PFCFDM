from PostProcess.plotxdmf import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter



file = "out/shearDis/ShearDis0.1_1.h5"
t=0
X,Y,F, timest = get_im_data(file,"alphaTild",t,2)



fig , ax = plt.subplots(1,1,figsize=(8,3))

# Plot 1
# cf1 = ax.contourf(X, Y,F, levels=100, cmap="seismic")
cf1 = ax.imshow(F, origin="lower", cmap="seismic")
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="3%", pad=0.05)
cb1 = plt.colorbar(cf1, cax=cax1)

# Apply scientific formatter with LaTeX
fmt1 = ScalarFormatter(useMathText=True)
fmt1.set_powerlimits((0,0))  # adjust to force sci notation if needed
cb1.ax.yaxis.set_major_formatter(fmt1)
ax.set_title(r"field")

plt.show()