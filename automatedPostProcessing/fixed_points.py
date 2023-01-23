from pathlib import Path
from postprocessing_common import read_command_line
from dolfin import *

"""
Post-processing tool for finding fixed points from WSS.
"""

def compute_fixed_points(case_path, velocity_degree):
    case_path = Path(case_path)
    mesh_path = case_path / "mesh.h5"
    # Read mesh saved as HDF5 format
    mesh = Mesh()
    with HDF5File(MPI.comm_world, mesh_path.__str__(), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)
    # Generate boundary mesh from the volume
    bm = BoundaryMesh(mesh, 'exterior')
    V_b1 = VectorFunctionSpace(bm, "CG", 1)
    U_b1 = FunctionSpace(bm, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", velocity_degree)

    n = FacetNormal(mesh)
    
    # Define function spaces
    wss = Function(V_b1)
    wss_path = (case_path / "WSS.xdmf").__str__()
    with XDMFFile(MPI.comm_world, wss_path) as wss_file:
        wss_file.read_checkpoint(wss, "WSS")
        wss_file.close()

    from IPython import embed; embed(); exit(1)
    # NOTE: we want to take the divergence of WSS after this, but first check if WSS is read correctly

if __name__ == "__main__":
    folder, nu, rho, dt, velocity_degree, _, _, T, save_frequency, _, start_cycle, step, average_over_cycles \
        = read_command_line()

    compute_fixed_points(folder, velocity_degree)