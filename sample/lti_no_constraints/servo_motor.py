"""
File: servo_motor.py

This script demonstrates Model Predictive Control (MPC) for a DC servo motor system without constraints.
It models the servo motor as a linear time-invariant (LTI) system,
 discretizes the plant, sets up an MPC controller,
and simulates the closed-loop response to a pulse reference input.
 The simulation results are visualized using a plotting utility.

References:
A. Bemporad and E. Mosca, "Fulfilling hard constraints in uncertain linear systems
 by referenceerence managing," Automatica, vol. 34, no. 4, pp. 451-461, 1998.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_mpc'))

import math
import numpy as np
import control

from external_libraries.MCAP_python_mpc.mpc_utility.state_space_utility import SymbolicStateSpace
from external_libraries.MCAP_python_mpc.python_mpc.linear_mpc import LTI_MPC_NoConstraints
from python_mpc.linear_mpc_deploy import LinearMPC_Deploy

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter
from sample.simulation_manager.signal_edit.sampler import PulseGenerator


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

    Weight_U = np.diag([0.001])
    Weight_Y = np.diag([1.0, 0.005])

    Np = 20
    Nc = 2

    lti_mpc = LTI_MPC_NoConstraints(
        ideal_plant_model, Np=Np, Nc=Nc,
        Weight_U=Weight_U, Weight_Y=Weight_Y)

    # You can create cpp header which can easily define MPC as C++ code
    deployed_file_names = LinearMPC_Deploy.generate_LTI_MPC_NC_cpp_code(
        lti_mpc)
    print(deployed_file_names)

    # %% simulation
    t_sim = 20.0
    time = np.arange(0, t_sim, dt)

    # create input signal
    _, input_signal = PulseGenerator.sample_pulse(
        sampling_interval=dt,
        start_time=time[0],
        period=20.0,
        pulse_width=50.0,
        pulse_amplitude=1.0,
        duration=time[-1],
    )

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
        reference = np.array([[input_signal[i, 0]], [0.0]])
        U = lti_mpc.update(reference, y_measured)

        plotter.append_name(reference, "reference")
        plotter.append_name(U, "U")
        plotter.append_name(y_measured, "y_measured")
        plotter.append_name(X, "X")

    plotter.assign("reference", position=(0, 0),
                   column=0, row=0, x_sequence=time)
    plotter.assign("y_measured", position=(0, 0),
                   column=0, row=0, x_sequence=time)
    plotter.assign("reference", position=(0, 1),
                   column=1, row=0, x_sequence=time)
    plotter.assign("y_measured", position=(0, 1),
                   column=1, row=0, x_sequence=time)

    plotter.assign("X", position=(1, 0), column=0, row=0, x_sequence=time)
    plotter.assign("X", position=(1, 0), column=1, row=0, x_sequence=time)
    plotter.assign("X", position=(1, 1), column=2, row=0, x_sequence=time)
    plotter.assign("X", position=(2, 1), column=3, row=0, x_sequence=time)

    plotter.assign_all("U", position=(2, 0), x_sequence=time)

    plotter.plot("Servo Motor plant, MPC Response")


if __name__ == "__main__":
    main()
