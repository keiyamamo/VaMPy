from os import path

from dolfin import *

from postprocessing_common import STRESS, read_command_line, get_dataset_names

try:
    parameters["reorder_dofs_serial"] = False
except NameError:
    pass

"""
Post-processing tool for finding fixed points from WSS.
"""

def compute_fixed_points(case_path, velocity_degree):
 # File paths
    file_path_u = path.join(folder, "u.h5")
    mesh_path = path.join(folder, "mesh.h5")
    file_u = HDF5File(MPI.comm_world, file_path_u, "r")

    # Determine what time step to start post-processing from
    start = int(T / dt / save_frequency * (start_cycle - 1))

    # Get names of data to extract
    if MPI.rank(MPI.comm_world) == 0:
        print("Reading dataset names")

    dataset = get_dataset_names(file_u, start=start, step=step)

    # Read mesh saved as HDF5 format
    mesh = Mesh()
    with HDF5File(MPI.comm_world, mesh_path.__str__(), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)

    bm = BoundaryMesh(mesh, 'exterior')

    if MPI.rank(MPI.comm_world) == 0:
        print("Defining function spaces")

    V_b1 = VectorFunctionSpace(bm, "CG", 1)
    U_b1 = FunctionSpace(bm, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", velocity_degree)
    bmQdg0 = FunctionSpace(bm, 'DG', 0)

    if MPI.rank(MPI.comm_world) == 0:
        print("Defining functions")

    u = Function(V)

    # WSS_mean
    WSS_mean = Function(V_b1)
    WSS_mean_avg = Function(V_b1)

    # TAWSS
    TAWSS = Function(U_b1)
    TAWSS_avg = Function(U_b1)

    stress = STRESS(u, 0.0, nu, mesh)

    # Get number of saved steps and cycles
    saved_time_steps_per_cycle = int(T / dt / save_frequency / step)
    n_cycles = int(len(dataset) / saved_time_steps_per_cycle)

    # Set number of cycles to average over
    cycles_to_average = list(range(1, n_cycles + 1)) if average_over_cycles else []
    counters_to_save = [saved_time_steps_per_cycle * cycle for cycle in cycles_to_average]
    cycle_names = [""] + ["_cycle_{:02d}".format(cycle) for cycle in cycles_to_average]

    # Create XDMF files for saving indices
    fullname = file_path_u.replace("u.h5", "%s%s.xdmf")
    fullname = fullname.replace("Solutions", "Hemodynamics")
    
    index_names = ["TAWSS"]
    index_variables = [TAWSS]
    index_variables_avg = [TAWSS_avg]

    index_dict = dict(zip(index_names, index_variables))
    index_dict_cycle = dict(zip(index_names, index_variables_avg))

    indices = {}
    for cycle_name in cycle_names:
        for index in index_names + ["WSS"]:
            indices[index + cycle_name] = XDMFFile(MPI.comm_world, fullname % (index, cycle_name))
            indices[index + cycle_name].parameters["rewrite_function_mesh"] = False
            indices[index + cycle_name].parameters["flush_output"] = True
            indices[index + cycle_name].parameters["functions_share_mesh"] = True

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    counter = start
    for data in dataset:
        # Update file_counter
        counter += 1

        file_u.read(u, data)

        if MPI.rank(MPI.comm_world) == 0:
            timestamp = file_u.attributes(data)["timestamp"]
            print("=" * 10, "Timestep: {}".format(timestamp), "=" * 10)

        # Compute WSS
        if MPI.rank(MPI.comm_world) == 0:
            print("Compute WSS (mean)")
        tau = stress()
        tau.vector()[:] = tau.vector()[:] * rho
        WSS_mean_avg.vector().axpy(1, tau.vector())

        if MPI.rank(MPI.comm_world) == 0:
            print("Compute WSS (absolute value)")
        tawss = project(inner(tau, tau) ** (1 / 2), U_b1)
        TAWSS_avg.vector().axpy(1, tawss.vector())

        # Save instantaneous WSS
        tau.rename("WSS", "WSS")
        indices["WSS"].write(tau, dt * counter)

        # Compute divergence of WSS
        # FIXME: This will result in a constant over the cell but we want divergence of WSS to be continuous
        div_wss = Function(bmQdg0)
        div_wss = project(div(tau), bmQdg0)
        # Save the divergence of WSS
        
     
        if len(cycles_to_average) != 0 and counter == counters_to_save[0]:
            # Get cycle number
            cycle = int(counters_to_save[0] / saved_time_steps_per_cycle)
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Storing cardiac cycle {}".format(cycle), "=" * 10)

            # Get average over sampled time steps
            for index in [TAWSS_avg, WSS_mean_avg]:
                index.vector()[:] = index.vector()[:] / saved_time_steps_per_cycle

            # Compute OSI, RRT and ECAP
            wss_mean = project(inner(WSS_mean_avg, WSS_mean_avg) ** (1 / 2), U_b1)
            wss_mean_vec = wss_mean.vector().get_local()
            tawss_vec = TAWSS_avg.vector().get_local()

            # Rename displayed variable names
            for var, name in zip(index_variables_avg, index_names):
                var.rename(name, name)

            # Store solution
            for name, index in index_dict_cycle.items():
                indices[name + "_cycle_{:02d}".format(cycle)].write(index)

            # Append solution to total solution
            for index, index_avg in zip(index_dict.values(), index_dict_cycle.values()):
                index_avg.vector().apply("insert")
                index.vector().axpy(1, index_avg.vector())

            WSS_mean_avg.vector().apply("insert")
            WSS_mean.vector().axpy(1, WSS_mean_avg.vector())

            # Reset tmp solutions
            for index_avg in index_dict_cycle.values():
                index_avg.vector().zero()

            WSS_mean_avg.vector().zero()

            counters_to_save.pop(0)

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Saving hemodynamic indices", "=" * 10)

    # Time average computed indices
    n = n_cycles if average_over_cycles else (counter - start) // step
    index_dict = index_dict if len(cycles_to_average) != 0 else index_dict_cycle
    WSS_mean = WSS_mean if len(cycles_to_average) != 0 else WSS_mean_avg

    index_dict['TWSSG'].vector()[:] = index_dict['TWSSG'].vector()[:] / n
    index_dict['TAWSS'].vector()[:] = index_dict['TAWSS'].vector()[:] / n
    WSS_mean.vector()[:] = WSS_mean.vector()[:] / n
    wss_mean = project(inner(WSS_mean, WSS_mean) ** (1 / 2), U_b1)
   
    # Rename displayed variable names
    for name, var in index_dict.items():
        var.rename(name, name)

    # Write indices to file
    for name, index in index_dict.items():
        indices[name].write(index)

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Post processing finished", "=" * 10)
        print("Results saved to: {}".format(folder))


if __name__ == '__main__':
    folder, nu, rho, dt, velocity_degree, _, _, T, save_frequency, _, start_cycle, step, average_over_cycles \
        = read_command_line()

    compute_fixed_points(folder, nu, rho, dt, T, velocity_degree, save_frequency, start_cycle,
                                step, average_over_cycles)
