import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_mpc'))

import numpy as np
import sympy as sp
from dataclasses import dataclass

from external_libraries.MCAP_python_mpc.python_mpc.nonlinear_mpc import NonlinearMPC_TwiceDifferentiable
from python_mpc.nonlinear_mpc_deploy import NonlinearMPC_Deploy

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter


def create_plant_model():
    theta, omega, u0, dt, a, b, c, d = sp.symbols(
        'theta omega u0 dt a b c d', real=True)

    theta_next = theta + dt * omega
    omega_dot = -a * sp.sin(theta) - b * omega + c * \
        sp.cos(theta) * u0 + d * (u0 ** 2)
    omega_next = omega + dt * omega_dot

    f = sp.Matrix([theta_next, omega_next])
    h = sp.Matrix([[theta]])

    x_syms = sp.Matrix([[theta], [omega]])
    u_syms = sp.Matrix([[u0]])

    return f, h, x_syms, u_syms


@dataclass
class Parameters:
    a: float = 9.81     # gravity/l over I scaling
    b: float = 0.3      # damping
    c: float = 1.2      # state-dependent control effectiveness: cos(theta)*u
    d: float = 0.10     # actuator nonlinearity: u^2
    dt: float = 0.05    # sampling time step


def main():
    # simulation setup
    simulation_time = 10.0
    delta_time = 0.05
    Number_of_Delay = 0

    time = np.arange(0, simulation_time, delta_time)

    # Create symbolic plant model
    f, h, x_syms, u_syms = create_plant_model()

    # Prediction horizon
    Np = 10

    # define parameters
    state_space_parameters = Parameters()

    # input bounds
    U_min = np.array([[-2.0]])
    U_max = np.array([[2.0]])

    # weights
    Weight_U = np.array([0.05])
    Weight_X = np.array([2.5, 0.5])
    Weight_Y = np.array([2.5])

    Q_ekf = np.diag([1.0, 1.0])
    R_ekf = np.diag([1.0])

    # reference
    reference = np.array([[0.0]])

    # Nonlinear MPC object
    X_initial = np.array([[np.pi / 4.0], [0.0]])

    nonlinear_mpc = NonlinearMPC_TwiceDifferentiable(
        delta_time=state_space_parameters.dt,
        X=x_syms,
        U=u_syms,
        X_initial=X_initial,
        fxu=f,
        hx=h,
        parameters_struct=state_space_parameters,
        Np=Np,
        Weight_U=Weight_U,
        Weight_X=Weight_X,
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

    nonlinear_mpc.set_solver_max_iteration(10)

    x_true = X_initial
    u = np.array([[0.0]])

    plotter = SimulationPlotter()

    y_measured = np.array([[0.0]])
    y_store = [y_measured] * (Number_of_Delay + 1)
    delay_index = 0

    # simulation
    for i in range(round(simulation_time / delta_time)):
        # system response
        if i > 0:
            u = np.copy(u_from_mpc)

        x_true = nonlinear_mpc.kalman_filter.state_function(
            x_true, u, state_space_parameters)
        y_store[delay_index] = nonlinear_mpc.kalman_filter.measurement_function(
            x_true, state_space_parameters)

        # system delay
        delay_index += 1
        if delay_index > Number_of_Delay:
            delay_index = 0

        y_measured = y_store[delay_index]

        u_from_mpc = nonlinear_mpc.update_manipulation(reference, y_measured)

        solver_iteration = nonlinear_mpc.get_solver_step_iterated_number()

        plotter.append_name(x_true, "x_true")
        plotter.append_name(reference, "reference")
        plotter.append_name(y_measured, "y_measured")
        plotter.append_name(u_from_mpc, "u")
        plotter.append_name(solver_iteration, "solver_iteration")

    # plot
    plotter.assign("y_measured", column=0, row=0, position=(0, 0),
                   x_sequence=time, label="theta")
    plotter.assign("reference", column=0, row=0, position=(0, 0),
                   x_sequence=time, label="theta_ref")

    plotter.assign("x_true", column=0, row=0, position=(1, 0),
                   x_sequence=time, label="x_0")
    plotter.assign("x_true", column=1, row=0, position=(2, 0),
                   x_sequence=time, label="x_1")

    plotter.assign("u", column=0, row=0, position=(0, 1),
                   x_sequence=time, label="u")
    plotter.assign("solver_iteration", column=0, row=0, position=(1, 1),
                   x_sequence=time, label="solver_iteration")

    plotter.plot()


if __name__ == "__main__":
    main()
