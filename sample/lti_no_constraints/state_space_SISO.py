"""
File: state_space_SISO.py

This script demonstrates the simulation and deployment of a discrete-time state-space SISO (Single Input Single Output) system using Model Predictive Control (MPC) without constraints. The code defines a plant model, sets up an MPC controller, generates input and reference signals, simulates the closed-loop system with delay, and visualizes the results. It also provides functionality to export the MPC controller as C++ code for deployment.
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np

from mpc_utility.state_space_utility import SymbolicStateSpace
from python_mpc.linear_mpc import LTI_MPC_NoConstraints
from python_mpc.linear_mpc_deploy import LinearMPC_Deploy

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter
from sample.simulation_manager.signal_edit.sampler import PulseGenerator

PATH_FOLLOWING = False


def create_input_signal(dt, time, Np):
    if PATH_FOLLOWING:
        latest_time = time[-1]
        for i in range(Np):
            time = np.append(time, latest_time + dt * (i + 1))

        freq = 2.0
        amplitude = 1.0
        input_signal = (amplitude * np.sin(2 * np.pi *
                        freq * time)).reshape(-1, 1)
    else:
        _, input_signal = PulseGenerator.sample_pulse(
            sampling_interval=dt,
            start_time=time[0],
            period=1.0,
            pulse_width=50.0,
            pulse_amplitude=1.0,
            duration=time[-1],
        )

    return input_signal


def get_reference_signal(input_signal, index, Np):
    if PATH_FOLLOWING:
        ref = np.zeros((1, Np))
        for i in range(Np):
            ref[0, i] = input_signal[index + i, 0]
    else:
        ref = np.array([[input_signal[index, 0]]])

    return ref


def main():
    # %% define discrete state-space model
    A = np.array([[0.7, 0.2],
                  [-0.3, 0.8]])
    B = np.array([[0.1],
                  [0.2]])
    C = np.array([[1.0, 0.0]])
    # D = np.array([[0.0]])

    dt = 0.01
    Number_of_Delay = 5

    ideal_plant_model = SymbolicStateSpace(
        A, B, C, delta_time=dt, Number_of_Delay=Number_of_Delay)

    Weight_U = np.diag([1.0])
    Weight_Y = np.diag([1.0])

    Np = 10
    Nc = 2

    lti_mpc = LTI_MPC_NoConstraints(
        ideal_plant_model, Np=Np, Nc=Nc,
        Weight_U=Weight_U, Weight_Y=Weight_Y,
        is_ref_trajectory=PATH_FOLLOWING)

    # You can create cpp header which can easily define lti_mpc as C++ code
    deployed_file_names = LinearMPC_Deploy.generate_LTI_MPC_NC_cpp_code(
        lti_mpc, number_of_delay=Number_of_Delay)
    print(deployed_file_names)

    # %% simulation
    t_sim = 1.0
    time = np.arange(0, t_sim, dt)

    input_signal = create_input_signal(dt, time, Np)

    # real plant model
    # You can change the characteristic with changing the A, B, C matrices
    A = np.array([[0.7, 0.2],
                  [-0.3, 0.8]])
    B = np.array([[0.1],
                  [0.2]])
    C = np.array([[1.0, 0.0]])
    # D = np.array([[0.0]])

    X = np.array([[0.0],
                  [0.0]])
    Y = np.array([[0.0]])
    U = np.array([[0.0]])

    plotter = SimulationPlotter()

    y_measured = Y
    y_store = [Y] * (Number_of_Delay + 1)
    delay_index = 0

    for i in range(len(time)):
        # system response
        X = A @ X + B @ U
        y_store[delay_index] = C @ X

        # system delay
        delay_index += 1
        if delay_index > Number_of_Delay:
            delay_index = 0

        y_measured = y_store[delay_index]

        # controller
        ref = get_reference_signal(input_signal, i, Np)
        U = lti_mpc.update(ref, y_measured)

        plotter.append_name(ref, "ref")
        plotter.append_name(U, "U")
        plotter.append_name(y_measured, "y_measured")
        plotter.append_name(X, "X")

    plotter.assign("ref", position=(0, 0), column=0, row=0, x_sequence=time)
    plotter.assign_all("y_measured", position=(0, 0), x_sequence=time)
    plotter.assign_all("X", position=(1, 0), x_sequence=time)
    plotter.assign_all("U", position=(2, 0), x_sequence=time)

    plotter.plot("State-Space plant, MPC Response")


if __name__ == "__main__":
    main()
