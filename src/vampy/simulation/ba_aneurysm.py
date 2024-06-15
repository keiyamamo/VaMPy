import json
import os
import pickle
from pprint import pprint
import numpy as np
from dolfin import set_log_level, MPI

from oasis.problems.NSfracStep import *

from vampy.simulation.simulation_common import store_u_mean, get_file_paths, print_mesh_information, \
    store_velocity_and_pressure_h5
from vampy.simulation.Womersley import make_womersley_bcs, compute_boundary_geometry_acrn

from vasp.simulations.simulation_common import calculate_and_print_flow_properties, print_probe_points

# FEniCS specific command to control the desired level of logging, here set to critical errors
set_log_level(50)
comm = MPI.comm_world


def problem_parameters(commandline_kwargs, NS_parameters, scalar_components, Schmidt, NS_expressions, **NS_namespace):
    """
    Problem file for running CFD simulation in left atrial models consisting of arbitrary number of pulmonary veins (PV)
    (normally 3 to 7), and one outlet being the mitral valve. A Womersley velocity profile is applied at the inlets,
    where the total flow rate is split between the area ratio of the PVs. The mitral valve is considered open with a
    constant pressure of p=0. Flow rate and flow split values for the inlet condition are computed from the
    pre-processing script automatedPreProcessing.py. The simulation is run for two cycles (adjustable), but only the
    results/solutions from the second cycle are stored to avoid non-physiological effects from the first cycle.
    One cardiac cycle is set to 0.951 s from [1], and scaled by a factor of 1000, hence all parameters are in
    [mm] or [ms].

    [1] Hoi, Yiemeng, et al. "Characterization of volumetric flow rate waveforms at the carotid bifurcations of older
        adults." Physiological measurement 31.3 (2010): 291.
    """
    if "restart_folder" in commandline_kwargs.keys():
        restart_folder = commandline_kwargs["restart_folder"]
        f = open(path.join(restart_folder, 'params.dat'), 'rb')
        NS_parameters.update(pickle.load(f))
        NS_parameters['restart_folder'] = restart_folder
    else:
        # Override some problem specific parameters
        # Parameters are in mm and ms
        cardiac_cycle = float(commandline_kwargs.get("cardiac_cycle", 951))
        number_of_cycles = float(commandline_kwargs.get("number_of_cycles", 2))

        NS_parameters.update(
            # Fluid parameters
            nu=3.3018868e-3,  # Viscosity [nu: 0.0035 Pa-s / 1060 kg/m^3 = 3.3018868E-6 m^2/s == 3.3018868E-3 mm^2/ms]
            # Geometry parameters
            id_in=[6, 2],  # Inlet boundary IDs
            id_out=[1, 3, 4, 5],  # Outlet boundary IDs
            vel_t_ramp=0.0,  # Time for velocity ramp [ms]
            # Simulation parameters
            cardiac_cycle=cardiac_cycle,  # Run simulation for 1 cardiac cycles [ms]
            T=number_of_cycles * cardiac_cycle,  # Number of cycles
            dt=0.0951,  # # Time step size [ms]
            # Frequencies to save data
            dump_probe_frequency=2,
            save_solution_frequency=5,  # Save frequency for velocity and pressure field
            save_solution_after_cycle=1,  # Store solution after 1 cardiac cycle
            # Oasis specific parameters
            checkpoint=100,  # Overwrite solution in Checkpoint folder each checkpoint
            print_intermediate_info=100,
            folder="results_ba",
            mesh_path=commandline_kwargs["mesh_path"],
            # Solver parameters
            velocity_degree=1,
            pressure_degree=1,
            use_krylov_solvers=True,
            krylov_solvers=dict(monitor_convergence=False)
        )

    mesh_file = NS_parameters["mesh_path"].split("/")[-1]
    case_name = mesh_file.split(".")[0]
    NS_parameters["folder"] = path.join(NS_parameters["folder"], case_name)

    if MPI.rank(MPI.comm_world) == 0:
        print("=== Starting simulation for Atrium.py ===")
        print("Running with the following parameters:")
        pprint(NS_parameters)


def mesh(mesh_path, **NS_namespace):
    # Read mesh and print mesh information
    atrium_mesh = Mesh(mesh_path)
    print_mesh_information(atrium_mesh)

    return atrium_mesh


# # Define velocity inlet parabolic profile
# class VelInPara(UserExpression):
#     def __init__(self, t, dt, vel_t_ramp, n, dsi, mesh, interp_velocity, **kwargs):
#         self.t = t
#         self.dt = dt
#         self.t_ramp = vel_t_ramp
#         self.interp_velocity = interp_velocity
#         self.number = int(self.t / self.dt)
#         self.n = n  # normal direction
#         self.dsi = dsi  # surface integral element
#         self.d = mesh.geometry().dim()
#         self.x = SpatialCoordinate(mesh)
#         # Compute area of boundary tesselation by integrating 1.0 over all facets
#         self.A = assemble(Constant(1.0, name="one") * self.dsi)
#         # Compute barycenter by integrating x components over all facets
#         self.c = [assemble(self.x[i] * self.dsi) / self.A for i in range(self.d)]
#         # Compute radius by taking max radius of boundary points
#         self.r = np.sqrt(self.A / np.pi)
#         super().__init__(**kwargs)

#     def update(self, t):
#         self.t = t
#         # if self.number + 1 < len(self.interp_velocity):
#         self.number = int(self.t / self.dt)
#         if MPI.rank(MPI.comm_world) == 0:
#             print("Number: ", self.number)
#             print("Velocity: ", self.interp_velocity[self.number])
#             print("Time: ", self.t)
#             print("n[0]: ", self.n[0])
#             print("n[1]: ", self.n[1])
#             print("n[2]: ", self.n[2])

#     def eval(self, value, x):
#         # Define the parabola
#         rv = x - self.c
#         rvn = rv.dot(self.n)
#         rp = rv - rvn * self.n
#         r2_new = rp.dot(rp)
#         y = np.sqrt(r2_new) / self.r
#         fact_r = 1 - y ** 2

#         # r2 = (x[0] - self.c[0]) ** 2 + (x[1] - self.c[1]) ** 2 + (x[2] - self.c[2]) ** 2  # radius**2
#         # fact_r = 1 - (r2 / self.r ** 2)

#         value[0] = -self.n[0] * (self.interp_velocity[self.number]) * fact_r
#         value[1] = -self.n[1] * (self.interp_velocity[self.number]) * fact_r
#         value[2] = -self.n[2] * (self.interp_velocity[self.number]) * fact_r
#         if fact_r > 0.9 and fact_r < 1.0:
#             if MPI.rank(MPI.comm_world) == 0:
#                 print("Value: ", value)

#     def value_shape(self):
#         return (3,)


def create_bcs(NS_expressions, mesh, T, dt, nu, V, Q, id_in, id_out, vel_t_ramp, t, **NS_namespace):
    # Variables needed during the simulation
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1, mesh.domains())
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    # ID for boundary conditions
    id_lva = id_in[0]
    id_rva = id_in[1]

    if MPI.rank(MPI.comm_world) == 0:
        print("LVA ID: ", id_lva)
        print("RVA ID: ", id_rva)
    
    if MPI.rank(MPI.comm_world) == 0:
        print("Create bcs")

    # Fluid velocity BCs
    dsi1 = ds(id_lva)
    dsi2 = ds(id_rva)

    # inlet area
    inlet_area1 = assemble(1 * dsi1)
    inlet_area2 = assemble(1 * dsi2)

    if MPI.rank(MPI.comm_world) == 0:
        print("Inlet area1: ", inlet_area1)
        print("Inlet area2: ", inlet_area2)

    n = FacetNormal(mesh)
    ndim = mesh.geometry().dim()
    ni1 = np.array([assemble(n[i] * dsi1) for i in range(ndim)])
    ni2 = np.array([assemble(n[i] * dsi2) for i in range(ndim)])
    n_len1 = np.sqrt(sum([ni1[i] ** 2 for i in range(ndim)]))
    n_len2 = np.sqrt(sum([ni2[i] ** 2 for i in range(ndim)]))
    normal1 = ni1 / n_len1
    normal2 = ni2 / n_len2
    
    if MPI.rank(MPI.comm_world) == 0:
        print("Normal1: ", normal1)
        print("Normal2: ", normal2)
    
    lva_velocity = [0.848637616,
                    0.80701909,
                    0.847125648,
                    0.938298849,
                    1.254212349,
                    1.891392824,
                    2.027271038,
                    1.77177903,
                    1.792383614,
                    1.913592303,
                    1.888049839,
                    1.740999381,
                    1.520779612,
                    1.344482404,
                    1.382113934,
                    1.499291641,
                    1.443681085,
                    1.329292421,
                    1.29572188,
                    1.23181514,
                    1.178122082,
                    1.174780102,
                    1.136369058,
                    1.022520303,
                    0.93012271,
                    0.912917651,
                    0.920154996,
                    0.895582053,
                    0.942527649,
                    0.936963696]
        
    rva_velocity = [0.247426581,
                    0.422073122,
                    0.810264593,
                    1.229616588,
                    2.163382059,
                    3.7332934,
                    4.550075292,
                    4.245502431,
                    4.118670492,
                    3.095260393,
                    2.180491478,
                    1.521510309,
                    0.758527135,
                    0.368080821,
                    0.481735522,
                    0.380346012,
                    0.185040665,
                    0.10628233,
                    0.119750561,
                    0.234517943,
                    0.396352626,
                    0.375314551,
                    0.332350363,
                    0.251735247,
                    0.234669003,
                    0.182901337,
                    0.299768979,
                    0.433334731,
                    0.381273636,
                    0.319951809]
    
    len_v = len(lva_velocity)
    t_v = np.arange(len(lva_velocity))
    num_t = 1000
    tnew = np.linspace(0, len_v, num=num_t)
    t_values = np.linspace(0, 951, num=num_t)
    lva = np.array(np.interp(tnew, t_v, lva_velocity))
    # Inflow at lva
    tmp_area, tmp_center, tmp_radius, tmp_normal = compute_boundary_geometry_acrn(mesh, id_in[0], boundaries)
    flow_rate = lva * tmp_area
    inlet_lva = make_womersley_bcs(t_values, flow_rate, nu, tmp_center, tmp_radius, tmp_normal, V.ufl_element())
    NS_expressions[f"inlet_{id_in[0]}"] = inlet_lva

    # Inflow at rva
    tmp_area, tmp_center, tmp_radius, tmp_normal = compute_boundary_geometry_acrn(mesh, id_in[1], boundaries)
    interp_rva = np.array(np.interp(t_values, t_v, rva_velocity))
    flow_rate = interp_rva * tmp_area
    inlet_rva = make_womersley_bcs(t_values, flow_rate, nu, tmp_center, tmp_radius, tmp_normal, V.ufl_element())
    NS_expressions[f"inlet_{id_in[1]}"] = inlet_rva

    # Initial condition
    for ID in id_in:
        for i in [0, 1, 2]:
            NS_expressions[f"inlet_{ID}"][i].set_t(t)

    # Create inlet boundary conditions
    bc_inlets = {}
    for ID in id_in:
        bc_inlet = [DirichletBC(V, NS_expressions[f"inlet_{ID}"][i], boundaries, ID) for i in range(3)]
        bc_inlets[ID] = bc_inlet

    # Set outlet boundary conditions, assuming one outlet (Mitral Valve)
    bc_p = [DirichletBC(Q, Constant(0), boundaries, ID) for ID in id_out]

    # No slip on walls
    id_wall = 0
    bc_wall = [DirichletBC(V, Constant(0.0), boundaries, id_wall)]

    #print wall area
    wall_area = assemble(1 * ds(id_wall))
    if MPI.rank(MPI.comm_world) == 0:
        print("Wall area: ", wall_area)

    # Create lists with all velocity boundary conditions
    bc_u0 = [bc_inlets[ID][0] for ID in id_in] + bc_wall
    bc_u1 = [bc_inlets[ID][1] for ID in id_in] + bc_wall
    bc_u2 = [bc_inlets[ID][2] for ID in id_in] + bc_wall

    return dict(u0=bc_u0, u1=bc_u1, u2=bc_u2, p=bc_p)


def pre_solve_hook(V, cardiac_cycle, dt, save_solution_after_cycle, mesh_path, mesh, newfolder, velocity_degree,
                   restart_folder, id_in, **NS_namespace):

    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1, mesh.domains())
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    # ID for boundary conditions
    id_lva = id_in[0]
    dsi1 = ds(id_lva)
    inlet_area1 = assemble(1 * dsi1)

    # Create point for evaluation
    n = FacetNormal(mesh)
    eval_dict = {}
    rel_path = mesh_path.split(".xml")[0] + "_probe_point.json"
    with open(rel_path, 'r') as infile:
        probe_points = np.array(json.load(infile))

    if restart_folder is None:
        # Get files to store results
        files = get_file_paths(newfolder)
        NS_parameters.update(dict(files=files))
    else:
        files = NS_namespace["files"]

    # Save mesh as HDF5 file for post-processing
    with HDF5File(MPI.comm_world, files["mesh"], "w") as mesh_file:
        mesh_file.write(mesh, "mesh")

    # Create vector function for storing velocity
    Vv = VectorFunctionSpace(mesh, "CG", velocity_degree)
    U = Function(Vv)
    u_mean = Function(Vv)
    u_mean0 = Function(V)
    u_mean1 = Function(V)
    u_mean2 = Function(V)

    # Time step when solutions for post-processing should start being saved
    save_solution_at_tstep = int(cardiac_cycle * save_solution_after_cycle / dt)

    return dict(n=n, eval_dict=eval_dict, U=U, u_mean=u_mean, u_mean0=u_mean0, u_mean1=u_mean1,
                u_mean2=u_mean2, save_solution_at_tstep=save_solution_at_tstep, probe_points=probe_points,
                inlet_area1=inlet_area1, dsi1=dsi1, **NS_namespace)


def temporal_hook(mesh, dt, t, save_solution_frequency, u_, NS_expressions, id_in, tstep, 
                p_, save_solution_at_tstep, nu, U, u_mean0, u_mean1, dump_probe_frequency,
                  u_mean2, n, inlet_area1, dsi1, probe_points, **NS_namespace):
    # Update inlet condition
    for ID in id_in:
        for i in [0, 1, 2]:
            NS_expressions["inlet_{}".format(ID)][i].set_t(t)

      # Assign velocity components to vector solution
    if tstep % dump_probe_frequency == 0:
        for i in range(3):
            assign(U.sub(i), u_[i])
        print_probe_points(U, p_,  probe_points)

        if MPI.rank(MPI.comm_world) == 0:
            txt = "Solved for timestep {:d}, t = {:2.04f} in {:3.1f} s"
            txt = txt.format(tstep, t, 0)
            print(txt)
    
    calculate_and_print_flow_properties(dt, mesh, u_, inlet_area1, nu, 1, n, dsi1, True)

    inlet_averaged_velocity = assemble(inner(u_, n) * dsi1) / inlet_area1

    if MPI.rank(MPI.comm_world) == 0:
        print("Inlet averaged velocity: ", abs(inlet_averaged_velocity))

    # Save velocity and pressure for post-processing
    if tstep % save_solution_frequency == 0 and tstep >= save_solution_at_tstep:
        store_velocity_and_pressure_h5(NS_parameters, U, p_, tstep, u_, u_mean0, u_mean1, u_mean2)


# Oasis hook called after the simulation has finished
def theend_hook(u_mean, u_mean0, u_mean1, u_mean2, T, dt, save_solution_at_tstep, save_solution_frequency,
                **NS_namespace):
    store_u_mean(T, dt, save_solution_at_tstep, save_solution_frequency, u_mean, u_mean0, u_mean1, u_mean2,
                 NS_parameters)
