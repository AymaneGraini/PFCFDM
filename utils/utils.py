import numpy as np
import matplotlib.pyplot as plt
import dolfinx.geometry as gm
from dolfinx import fem, mesh
import dolfinx.mesh
import pandas as pd
from dolfinx.fem import (Expression, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological)
import ufl
from mpi4py import MPI
import petsc4py
def get2dplot(psi,x_vals, y_vals,domain,tree,filename,show):
    X_parametric, Y_parametric = np.meshgrid(x_vals, y_vals)
    grid_points = np.vstack([X_parametric.ravel(), Y_parametric.ravel(), np.zeros_like(X_parametric.ravel())]).T
    tree = gm.bb_tree(domain, domain.topology.dim)
    Z= np.full_like(X_parametric, np.nan)

    for idx, point in enumerate(grid_points):
        cell_candidates = gm.compute_collisions_points(tree, point)
        colliding_cells = gm.compute_colliding_cells(domain, cell_candidates, point)
        
        if len(colliding_cells) > 0:
            cell_index = colliding_cells.array[0]
            value = psi.eval(point, cell_index)
            i, j = divmod(idx, X_parametric.shape[1])
            Z[i, j] = value
        else:
            print(f"Point {point[:2]} is outside the mesh.")
    x_flat = X_parametric.ravel()
    y_flat = Y_parametric.ravel()
    z_flat = Z.ravel()

    data = {
        "x": x_flat,
        "y": y_flat,
        "value": z_flat
    }
    df = pd.DataFrame(data)

    csv_filename = filename+".csv"
    df.to_csv(csv_filename, index=False)
    if show:
        plt.imshow(Z,extent=[0, x_vals[-1], 0, y_vals[-1]], origin="lower", cmap="seismic")
        plt.colorbar()
        plt.show()


def error_L2(uh, u_ex, degree_raise=3):
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().family_name

    mesh = uh.function_space.mesh
    W = functionspace(mesh, (family, degree + degree_raise,uh.ufl_shape))
    u_W = Function(W)
    u_W.interpolate(uh)
    u_ex_W = Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    e_W = Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    error = form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)

def error_L2_rel(uh, u_ex, degree_raise=3):
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().family_name

    mesh = uh.function_space.mesh
    W = functionspace(mesh, (family, degree + degree_raise,uh.ufl_shape))
    u_W = Function(W)
    u_W.interpolate(uh)
    u_ex_W = Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    e_W = Function(W)
    e_W.x.array[:] = (u_W.x.array - u_ex_W.x.array)/u_W.x.array 

    error = form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)

def Tabulate_Vals(v,sp,filename):
    dof_coords = sp.tabulate_dof_coordinates()
    u_values = v.x.array[:]
    data = np.column_stack((dof_coords[:, 0],dof_coords[:, 1],u_values))
    np.savetxt("./"+filename+".csv",data,delimiter="\t",fmt='%1.6e')


def LoadPsi_csv(filename):
    file_path = filename  
    data = pd.read_csv(file_path, header=None, delimiter="\t", names=['x', 'y', 'f(x,y)'])

    data['x'] = data['x'].round(6)
    data['y'] = data['y'].round(6)

    x_order = sorted(data['x'].unique())
    y_order = sorted(data['y'].unique())

    X,Y = np.meshgrid(x_order,y_order)
    pivot_table = data.pivot_table(index='y', columns='x', values='f(x,y)')

    Psi = pivot_table.to_numpy()

    return X,Y,Psi


def mark_facets(domain,L,H):
    fdim = domain.topology.dim - 1
    marked_values = []
    marked_facets = []
    tag=1
    def location(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], L) | np.isclose(x[1], 0.0) | np.isclose(x[1], H)
    facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, location)
    marked_facets.append(facets)
    marked_values.append(np.full_like(facets, tag))
    marked_facets = np.hstack(marked_facets)
    marked_values = np.hstack(marked_values)
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(
        domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
    )
    return facet_tag


def save_simulation_details(pfcSolver,mecSolver,a0,qs,ps,domain,shape,comment,filename): 

    pass


def buildnullspace(domain ,V: fem.FunctionSpace):
        c1 = dolfinx.fem.Function(V)
        c1.interpolate(lambda x: np.array([[1],[0]]))
        C1 = c1.x.petsc_vec
        # C1.scale(1/C1.norm())
        c2 = dolfinx.fem.Function(V)
        c2.interpolate(lambda x: np.array([[0],[1]]))
        C2 = c2.x.petsc_vec
        # C2.scale(1/C2.norm())
        c3 = dolfinx.fem.Function(V)
        c3.interpolate(lambda x: np.array((-x[1],x[0])))
        C3 = c3.x.petsc_vec
        # C3.scale(1.0/C3.norm())


        # print(C1.array_r)
        # # dolfinx.la.orthonormalize([C1, C2, C3])
        nullspace = petsc4py.PETSc.NullSpace().create(vectors=[C1, C2, C3], comm=domain.comm)
        return nullspace