from pathlib import Path
from postprocessing_common import read_command_line
from dolfin import *

"""
Post-processing tool for finding fixed points from WSS.
Currently, it does not work because WSS is not readalbe from XDMF file. 
If we use `write_checkpoint` instead of `write`, it should work.
"""

def compute_fixed_points(case_path):
    case_path = Path(case_path)
    mesh_path = case_path / "mesh.h5"
    # Read mesh saved as HDF5 format
    mesh = Mesh()
    with HDF5File(MPI.comm_world, mesh_path.__str__(), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)
    # Generate boundary mesh from the volume
    bm = BoundaryMesh(mesh, 'exterior')

    # Define function spaces
    tawss = VectorFunctionSpace(bm, "CG", 1)
    tawss_path = (case_path / "TAWSS.xdmf").__str__()
    tmp_file = XDMFFile(MPI.comm_world, tawss_path)
    tmp_file.read_checkpoint(tawss, "TAWSS", 0)
    tmp_file.close()
    from IPython import embed; embed(); exit(1)
    # NOTE: we want to take the divergence of WSS after this, but first check if WSS is read correctly

if __name__ == "__main__":
    folder, nu, rho, dt, velocity_degree, _, _, _, _, _, _ = read_command_line()
    compute_fixed_points(folder)