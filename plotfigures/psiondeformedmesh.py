import numpy as np
import meshio
import pyvista as pv
from dolfinx.io import XDMFFile
from mpi4py import MPI
import h5py
import dolfinx.plot as plot
import dolfinx
import basix
import dolfinx.fem as fem
if pv.OFF_SCREEN:
    pv.start_xvfb(wait=0.1)




def get_im_data(filename,fieldname,ti,comp=None):
    data1=None
    X,Y= None,None
    with h5py.File(filename, "r") as f:
        try :
            full_field_timeseries = f["Function/"+fieldname+"/"]
        except:
            raise ValueError("Requested field is not available. Available are", list(f["Function"]))

        timekeys= full_field_timeseries.keys()
        sorted_timesteps = sorted(timekeys, key=lambda s: float(s.replace("_", ".")))
        # print("exporting data from ", sorted_timesteps[ti])
        # print("ti = ", ti)
        if ti >= len(sorted_timesteps):
            raise ValueError("Requested timestep is not available, maximum snapshot is", len(sorted_timesteps)-1)
        Full_field = f["Function/"+fieldname+"/"+sorted_timesteps[ti]]
        mesh= f["Mesh/mesh/geometry"]
        x = mesh[:, 0]
        y = mesh[:, 1]
        nx = len(np.unique(x))
        ny = len(np.unique(y))
        newx = np.linspace(x.min(),x.max(),nx)
        newy = np.linspace(y.min(),y.max(),ny)
        X,Y = np.meshgrid(newx,newy) 
        mapping = np.lexsort((mesh[:, 0], mesh[:, 1]))
        if comp==None:
            field=Full_field[:]
        else:
            field=Full_field[:,comp]
            
        data1= field[mapping].reshape(ny, nx)
    timest = float(sorted_timesteps[ti].replace("_","."))
    Map = np.copy(mapping)
    return X,Y,field, timest,Map


# ---- Step 1: Read the mesh and field data from the XDMF file ----
file   = "./out/shear_nodislocation/Shear_dispadfft0.2_1.xdmf"
fileh5 = "./out/shear_nodislocation/Shear_dispadfft0.2_1.h5"
with XDMFFile(MPI.COMM_WORLD, file, "r") as xdmf:
    domain =  xdmf.read_mesh()





X,Y,Psi, timest,Map = get_im_data(fileh5,"Psi",2,comp=None)
X,Y,ux, timest,Map = get_im_data(fileh5,"u",2,comp=0)
X,Y,uy, timest,Map = get_im_data(fileh5,"u",2,comp=1)

mesh_coords = domain.geometry.x
Map     = np.lexsort((mesh_coords[:, 0], mesh_coords[:, 1]))


rev_DofMap = np.empty_like(Map)

rev_DofMap = rev_DofMap[Map] = np.arange(len(Map))

elem       = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
Vs_P2      = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(2,))
main_space = fem.functionspace(domain, elem)
vectorspace = fem.functionspace(domain, Vs_P2)

psi_func =  fem.Function(main_space,name="Psi")
u =  fem.Function(vectorspace,name="u")



u_flat = np.empty((ux.size + uy.size,), dtype=ux.dtype)
u_flat[0::2] = ux
u_flat[1::2] = uy
u.x.array[:] = u_flat

psi_func.x.array[:] = Psi[:,0]

print("dim ",domain.geometry.dim)
domain.geometry.x[:, :domain.geometry.dim] += u.x.array.reshape((-1, domain.geometry.dim))*5

cells, types, x = plot.vtk_mesh(domain)
grid = pv.UnstructuredGrid(cells, types, x)

grid.point_data["psi"] = psi_func.x.array
grid.set_active_scalars("psi")
plotter = pv.Plotter(window_size=(750, 400))
plotter.add_mesh(grid, scalars="psi", cmap="seismic", show_edges=False)
# plotter.add_scalar_bar("psi")
plotter.view_xy()
plotter.show()
