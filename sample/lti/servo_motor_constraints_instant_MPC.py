"""
File: servo_motor_constraints_instant_MPC.py

This script demonstrates the setup,
simulation, and deployment of a constrained
Instant Model Predictive Controller (iMPC)
for a servo motor system.
The code constructs a discrete-time state-space model of the plant,
defines MPC weights and constraints,
and simulates the closed-loop response of the system under the designed iMPC.
The iMPC performs a single primal-dual update per control step
instead of solving a full QP, enabling real-time control.
The simulation results,
including reference tracking, control input,
and state trajectories, are visualized using a plotting utility.

References:
    The algorithm is based on the iMPC framework using Lyapunov-based
    stability guarantees with back-stepping line search.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

import math
import numpy as np
import control

from external_libraries.MCAP_python_mpc.mpc_utility.state_space_utility import SymbolicStateSpace
from external_libraries.MCAP_python_mpc.python_mpc.linear_mpc_instant import InstantMPC_LTI
from python_mpc.linear_mpc_instant_deploy import InstantMPC_LTI_Deploy

from sample.simulation_manager.visualize.simulation_plotter_dash import SimulationPlotterDash
from sample.simulation_manager.signal_edit.sampler import PulseGenerator

PATH_FOLLOWING = False


def create_input_signal(dt, time, Np):
    if PATH_FOLLOWING:
        latest_time = time[-1]
        for i in range(Np):
            time = np.append(time, latest_time + dt * (i + 1))

        freq = 0.1
        amplitude = 1.0
        input_signal = (amplitude * np.sin(2 * np.pi *
                        freq * time)).reshape(-1, 1)
    else:
        _, input_signal = PulseGenerator.sample_pulse(
            sampling_interval=dt,
            start_time=time[0],
            period=20.0,
            pulse_width=50.0,
            pulse_amplitude=1.0,
            duration=time[-1],
        )

    return input_signal


def get_reference_signal(input_signal, index, Np):
    if PATH_FOLLOWING:
        reference = np.zeros((2, Np))
        for i in range(Np):
            reference[0, i] = input_signal[index + i, 0]
    else:
        reference = np.array([[input_signal[index, 0]], [0.0]])

    return reference


def create_plant_model():
    Lshaft = 1.0
    dshaft = 0.02
    shaftrho = 7850.0
    G = 81500.0 * 1.0e6

    tauam = 50.0 * 1.0e6

    Mmotor = 100.0
    Rmotor = 0.1
    Jmotor = 0.5 * Mmotor * Rmotor ** 2
    Bmotor = 0.1
    R = 20.0
    Kt = 10.0

    gear = 20.0

    Jload = 50.0 * Jmotor
    Bload = 25.0

    Ip = math.pi / 32.0 * dshaft ** 4
    Kth = G * Ip / Lshaft
    Vshaft = math.pi * (dshaft ** 2) / 4.0 * Lshaft
    Mshaft = shaftrho * Vshaft
    Jshaft = Mshaft * 0.5 * (dshaft ** 2 / 4.0)

    JM = Jmotor
    JL = Jload + Jshaft

    A = np.array([[0.0, 1.0, 0.0, 0.0],
                  [-Kth / JL, -Bload / JL, Kth / (gear * JL), 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [Kth / (JM * gear), 0.0, -Kth / (JM * gear ** 2), -(Bmotor + Kt ** 2 / R) / JM]])

    B = np.array([[0.0],
                  [0.0],
                  [0.0],
                  [Kt / (R * JM)]])

    C = np.array([[1.0, 0.0, 0.0, 0.0],
                  [Kth, 0.0, -Kth / gear, 0.0]])

    D = np.array([[0.0],
                  [0.0]])

    dc_motor_plant_sys = control.StateSpace(A, B, C, D)

    return dc_motor_plant_sys


def main():
    # %% create state-space model
    sys = create_plant_model()

    dt = 0.05
    Number_of_Delay = 0

    sys_d = sys.sample(Ts=dt, method='euler')

    ideal_plant_model = SymbolicStateSpace(
        sys_d.A, sys_d.B, sys_d.C, delta_time=dt, Number_of_Delay=Number_of_Delay)

    Weight_U = np.diag([0.0005])
    Weight_Y = np.diag([1.0, 0.005])

    Np = 20
    Nc = 2

    U_min = np.array([[-180.0]])
    U_max = np.array([[180.0]])

    zeta = 50.0
    near_zero_relative_limit = 1.0e-5

    impc = InstantMPC_LTI(
        ideal_plant_model,
        Np=Np, Nc=Nc,
        Weight_U=Weight_U, Weight_Y=Weight_Y,
        U_min=U_min, U_max=U_max,
        zeta=zeta,
        near_zero_relative_limit=near_zero_relative_limit,
        is_reference_trajectory=PATH_FOLLOWING)

    # You can create cpp header which can easily define MPC as C++ code
    deployed_file_names = InstantMPC_LTI_Deploy.generate_InstantMPC_LTI_cpp_code(
        impc)
    print(deployed_file_names)

    # %% simulation
    t_sim = 20.0
    time = np.arange(0, t_sim, dt)

    # create input signal
    input_signal = create_input_signal(dt, time, Np)

    # real plant model
    # You can change the characteristic with changing the A, B, C matrices
    A = sys_d.A
    B = sys_d.B
    C = sys_d.C
    # D = sys_d.D

    X = np.array([[0.0],
                  [0.0],
                  [0.0],
                  [0.0]])
    Y = np.array([[0.0],
                  [0.0]])
    U = np.array([[0.0]])

    plotter = SimulationPlotterDash()

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
        reference = get_reference_signal(input_signal, i, Np)
        U = impc.update(reference, y_measured)

        plotter.append_name(reference, "reference")

        plotter.append_name(U, "U")
        plotter.append_name(U_min, "U_min")
        plotter.append_name(U_max, "U_max")

        plotter.append_name(y_measured, "y_measured")

        plotter.append_name(X, "X")

    plotter.assign("reference", position=(0, 0),
                   row=0, column=0, x_sequence=time)
    plotter.assign("y_measured", position=(0, 0),
                   row=0, column=0, x_sequence=time)
    plotter.assign("reference", position=(0, 1),
                   row=1, column=0, x_sequence=time)
    plotter.assign("y_measured", position=(0, 1),
                   row=1, column=0, x_sequence=time)

    plotter.assign("X", position=(1, 0), row=0, column=0, x_sequence=time)
    plotter.assign("X", position=(1, 0), row=1, column=0, x_sequence=time)
    plotter.assign("X", position=(1, 1), row=2, column=0, x_sequence=time)
    plotter.assign("X", position=(2, 1), row=3, column=0, x_sequence=time)

    plotter.assign("U", position=(2, 0), x_sequence=time)
    plotter.assign("U_min", position=(2, 0), x_sequence=time, line_style='--')
    plotter.assign("U_max", position=(2, 0), x_sequence=time, line_style='--')

    if not os.environ.get("CI"):
        plotter.plot("Servo Motor plant, iMPC Response")


if __name__ == "__main__":
    main()
