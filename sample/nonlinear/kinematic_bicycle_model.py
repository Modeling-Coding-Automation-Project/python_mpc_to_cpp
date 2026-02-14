"""
File: kinematic_bicycle_model.py

Description: Example of Nonlinear MPC for a kinematic bicycle model with nonlinear dynamics.
This script implements and simulates a Nonlinear Model Predictive Control (MPC)
system for a kinematic bicycle model. The vehicle dynamics are symbolically derived using SymPy,
including the state-space and measurement models and their Jacobians.
The simulation runs a closed-loop control scenario,
where the MPC tracks a reference trajectory for vehicle position and orientation.
The code also visualizes the results using a custom plotter,
allowing analysis of the controller's performance over time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_mpc'))

import numpy as np
import sympy as sp
from dataclasses import dataclass

from external_libraries.MCAP_python_mpc.python_mpc.nonlinear_mpc import NonlinearMPC_TwiceDifferentiable
from python_mpc.nonlinear_mpc_deploy import NonlinearMPC_Deploy

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter
from external_libraries.MCAP_python_mpc.sample.nonlinear.support.interpolate_path import interpolate_path_csv


def load_cpp_run_data(cpp_csv_relpath=None):
    """
    Load C++ run CSV produced by the C++ demo.

    Args:
        cpp_csv_relpath: relative path from cwd to csv (default: 'sample/nonlinear/cpp_run_data.csv')

    Returns:
        (exists, px, py, yaw, v, delta, iteration, absolute_path)
        where each time-series is a (N,1) numpy array or None if missing.
    """
    if cpp_csv_relpath is None:
        cpp_csv_relpath = os.path.join(
            'sample', 'nonlinear', 'cpp_run_data.csv')

    cpp_csv_path = os.path.join(os.getcwd(), cpp_csv_relpath)
    cpp_run_data_exists = False
    px_cpp = py_cpp = yaw_cpp = v_cpp = delta_cpp = iteration_cpp = None

    try:
        if os.path.exists(cpp_csv_path):
            # The CSV header is: px,py,yaw,v,delta,iteration
            # Use numpy.genfromtxt to robustly skip header and handle missing values.
            cpp_data = np.genfromtxt(
                cpp_csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8')

            # If the file was empty or only header, genfromtxt may return an empty array
            if cpp_data is None or getattr(cpp_data, 'size', 0) == 0:
                cpp_data = None

            if cpp_data is not None:
                def col(name):
                    if name in cpp_data.dtype.names:
                        arr = np.asarray(cpp_data[name], dtype=float)
                        return arr.reshape(-1, 1)
                    return None

                px_cpp = col('px')
                py_cpp = col('py')
                yaw_cpp = col('yaw')
                v_cpp = col('v')
                delta_cpp = col('delta')
                iteration_cpp = None
                if 'iteration' in cpp_data.dtype.names:
                    iteration_cpp = np.asarray(
                        cpp_data['iteration']).reshape(-1, 1)

                print(
                    f"Loaded C++ run data from: {cpp_csv_path} (rows={px_cpp.shape[0] if px_cpp is not None else 0})")

                cpp_run_data_exists = True
    except Exception as e:
        print(f"Failed to load C++ run data: {e}")

    return cpp_run_data_exists, px_cpp, py_cpp, yaw_cpp, v_cpp, delta_cpp, iteration_cpp, cpp_csv_path


def create_plant_model():
    """
    Creates the symbolic nonlinear kinematic bicycle plant model using SymPy.

    Returns:
        fxu (sympy.Matrix): The symbolic state transition function f(x, u),
          representing the next state as a function of current state and control input.
        hx (sympy.Matrix): The symbolic output function h(x),
          representing the measurement or output equation.
        X (sympy.Matrix): The symbolic state vector [px, py, q0, q3],
          where px and py are positions, and q0, q3 are orientation components.
        U (sympy.Matrix): The symbolic control input vector [v, delta],
          where v is velocity and delta is steering angle.

    Notes:
        - The model uses quaternion-like orientation representation (q0, q3).
        - The state transition is discretized using the given delta_time.
        - wheel_base and delta_time are symbolic parameters for
          vehicle geometry and integration step.
    """
    wheel_base = sp.Symbol('wheel_base', real=True,
                           positive=True, nonzero=True)
    delta_time = sp.Symbol('delta_time', real=True,
                           positive=True, nonzero=True)

    px, py, q0, q3 = sp.symbols('px py q0 q3', real=True)
    v, delta = sp.symbols('v delta', real=True)

    X = sp.Matrix([px, py, q0, q3])
    U = sp.Matrix([v, delta])

    dtheta = (v / wheel_base) * sp.tan(delta)
    half = sp.Rational(1, 2)
    dq0 = sp.cos(dtheta * delta_time * half)
    dq3 = sp.sin(dtheta * delta_time * half)

    q0_next = q0 * dq0 - q3 * dq3
    q3_next = q0 * dq3 + q3 * dq0

    f = sp.Matrix([
        px + delta_time * v * (2 * q0**2 - 1),
        py + delta_time * v * (2 * q3 * q0),
        q0_next,
        q3_next
    ])
    fxu = sp.simplify(f)

    Y = sp.Matrix([px, py, q0, q3])
    hx = Y

    return fxu, hx, X, U


@dataclass
class Parameters:
    wheel_base: float = 2.8  # [m]
    delta_time: float = 0.1  # [s]


def main():
    # simulation setup
    simulation_time = 60.0
    delta_time = 0.1
    Number_of_Delay = 0

    fxu, hx, x_syms, u_syms = create_plant_model()

    OUTPUT_SIZE = hx.shape[0]

    # Prediction horizon
    Np = 10

    # define parameters
    state_space_parameters = Parameters()

    # input bounds
    U_min = np.array([[-1.0], [-1.5]])
    U_max = np.array([[1.0], [1.5]])

    # weights
    Weight_U = np.array([0.05, 0.05])
    Weight_Y = np.array([1.0, 1.0, 1.0, 1.0])

    Q_ekf = np.diag([1.0, 1.0, 1.0, 1.0])
    R_ekf = np.diag([1.0, 1.0, 1.0, 1.0])

    # Reference
    # "office_area_RRT_path_data.csv" is generated by sample/office_area_path_plan_demo.py
    # in "https://github.com/Modeling-Coding-Automation-Project/MCAP_python_navigation"
    times, px_reference_path, py_reference_path, yaw_reference_path = interpolate_path_csv(
        input_path="./external_libraries/MCAP_python_mpc/sample/nonlinear/support/office_area_RRT_path_data.csv",
        delta_time=delta_time,
        total_time=simulation_time
    )

    q0_reference_path = np.cos(yaw_reference_path * 0.5)
    q3_reference_path = np.sin(yaw_reference_path * 0.5)

    # --- Write reference path to CSV for C++ consumption ---
    # Combine px, py, q0, q3 into a single (T x 4) array and save as CSV
    try:
        reference_array = np.hstack((
            px_reference_path.reshape(-1, 1),
            py_reference_path.reshape(-1, 1),
            q0_reference_path.reshape(-1, 1),
            q3_reference_path.reshape(-1, 1),
        ))

        reference_csv_path = os.path.join(os.getcwd(), 'reference_path.csv')
        # Save with header so C++ can ignore or parse it if needed
        header = 'px,py,q0,q3'
        np.savetxt(reference_csv_path, reference_array,
                   delimiter=',', header=header, comments='')
        print(f"Wrote reference CSV to: {reference_csv_path}")
    except Exception as e:
        print(f"Failed to write reference CSV: {e}")

    # Nonlinear MPC object
    X_initial = np.array([[px_reference_path[0, 0]],
                          [py_reference_path[0, 0]],
                          [q0_reference_path[0, 0]],
                          [q3_reference_path[0, 0]]])

    nonlinear_mpc = NonlinearMPC_TwiceDifferentiable(
        delta_time=state_space_parameters.delta_time,
        X=x_syms,
        U=u_syms,
        X_initial=X_initial,
        fxu=fxu,
        hx=hx,
        parameters_struct=state_space_parameters,
        Np=Np,
        Weight_U=Weight_U,
        Weight_Y=Weight_Y,
        U_min=U_min,
        U_max=U_max,
        Q_kf=Q_ekf,
        R_kf=R_ekf,
        Number_of_Delay=Number_of_Delay,
    )

    # You can create cpp header which can easily define MPC as C++ code
    deployed_file_names = NonlinearMPC_Deploy.generate_Nonlinear_MPC_cpp_code(
        nonlinear_mpc)
    print(deployed_file_names)

    x_true = X_initial
    u = np.array([[0.0], [0.0]])

    nonlinear_mpc.set_solver_max_iteration(5)

    plotter = SimulationPlotter()

    y_measured = np.array([[0.0], [0.0], [0.0], [0.0]])
    y_store = [y_measured] * (Number_of_Delay + 1)
    delay_index = 0

    # simulation
    for i in range(round(simulation_time / delta_time)):
        # system response
        if i > 0:
            u = np.copy(u_from_mpc)

        x_true = nonlinear_mpc.kalman_filter.state_function(
            x_true, u, state_space_parameters)

        q_norm = np.sqrt(x_true[2, 0]**2 + x_true[3, 0]**2)
        x_true[2, 0] = x_true[2, 0] / q_norm
        x_true[3, 0] = x_true[3, 0] / q_norm

        y_store[delay_index] = nonlinear_mpc.kalman_filter.measurement_function(
            x_true, state_space_parameters)

        # system delay
        delay_index += 1
        if delay_index > Number_of_Delay:
            delay_index = 0

        y_measured = y_store[delay_index]

        # Reference for NMPC
        reference_path = np.zeros((OUTPUT_SIZE, Np))
        for j in range(Np):
            index = i + j
            if index >= px_reference_path.shape[0]:
                index = px_reference_path.shape[0] - 1

            reference_path[0, j] = px_reference_path[index, 0]
            reference_path[1, j] = py_reference_path[index, 0]
            reference_path[2, j] = q0_reference_path[index, 0]
            reference_path[3, j] = q3_reference_path[index, 0]

        u_from_mpc = nonlinear_mpc.update_manipulation(
            reference_path, y_measured)

        # monitoring
        solver_iteration = nonlinear_mpc.get_solver_step_iterated_number()

        px_reference = reference_path[0, 0]
        py_reference = reference_path[1, 0]
        yaw_reference = 2.0 * \
            np.arctan2(reference_path[3, 0], reference_path[2, 0])
        px_measured = y_measured[0, 0]
        py_measured = y_measured[1, 0]
        yaw_measured = 2.0 * np.arctan2(y_measured[3, 0], y_measured[2, 0])

        v = u_from_mpc[0, 0]
        delta = u_from_mpc[1, 0]

        plotter.append_name(px_reference, "px_reference")
        plotter.append_name(py_reference, "py_reference")
        plotter.append_name(yaw_reference, "yaw_reference")
        plotter.append_name(px_measured, "px_measured")
        plotter.append_name(py_measured, "py_measured")
        plotter.append_name(yaw_measured, "yaw_measured")
        plotter.append_name(v, "v")
        plotter.append_name(delta, "delta")
        plotter.append_name(solver_iteration, "solver_iteration")

    # Read C++ run data.
    # Read C++ run data using helper function
    cpp_csv_relpath = os.path.join('sample', 'nonlinear', 'cpp_run_data.csv')
    cpp_run_data_exists, px_cpp, py_cpp, yaw_cpp, v_cpp, delta_cpp, iteration_cpp, cpp_csv_path = \
        load_cpp_run_data(cpp_csv_relpath)

    if cpp_run_data_exists:
        plotter.append_sequence_name(px_cpp, "px_cpp")
        plotter.append_sequence_name(py_cpp, "py_cpp")
        plotter.append_sequence_name(yaw_cpp, "yaw_cpp")
        plotter.append_sequence_name(v_cpp, "v_cpp")
        plotter.append_sequence_name(delta_cpp, "delta_cpp")
        plotter.append_sequence_name(iteration_cpp, "solver_iteration_cpp")

    plotter.assign("px_reference", column=0, row=0, position=(0, 0),
                   x_sequence=times, label="px_reference")
    plotter.assign("px_measured", column=0, row=0, position=(0, 0),
                   x_sequence=times, label="px_measured")
    if cpp_run_data_exists:
        plotter.assign("px_cpp", column=0, row=0, position=(0, 0),
                       x_sequence=times, label="px_cpp")

    plotter.assign("py_reference", column=0, row=0, position=(1, 0),
                   x_sequence=times, label="py_reference")
    plotter.assign("py_measured", column=0, row=0, position=(1, 0),
                   x_sequence=times, label="py_measured")
    if cpp_run_data_exists:
        plotter.assign("py_cpp", column=0, row=0, position=(1, 0),
                       x_sequence=times, label="py_cpp")

    plotter.assign("yaw_reference", column=0, row=0, position=(2, 0),
                   x_sequence=times, label="yaw_reference")
    plotter.assign("yaw_measured", column=0, row=0, position=(2, 0),
                   x_sequence=times, label="yaw_measured")
    if cpp_run_data_exists:
        plotter.assign("yaw_cpp", column=0, row=0, position=(2, 0),
                       x_sequence=times, label="yaw_cpp")

    plotter.assign("v", column=0, row=0, position=(0, 1),
                   x_sequence=times, label="v")
    if cpp_run_data_exists:
        plotter.assign("v_cpp", column=0, row=0, position=(0, 1),
                       x_sequence=times, label="v_cpp")

    plotter.assign("delta", column=0, row=0, position=(1, 1),
                   x_sequence=times, label="delta")
    if cpp_run_data_exists:
        plotter.assign("delta_cpp", column=0, row=0, position=(1, 1),
                       x_sequence=times, label="delta_cpp")

    plotter.assign("solver_iteration", column=0, row=0, position=(2, 1),
                   x_sequence=times, label="solver_iteration")
    if cpp_run_data_exists:
        plotter.assign("solver_iteration_cpp", column=0, row=0, position=(2, 1),
                       x_sequence=times, label="solver_iteration_cpp")

    plotter.plot()


if __name__ == "__main__":
    main()
