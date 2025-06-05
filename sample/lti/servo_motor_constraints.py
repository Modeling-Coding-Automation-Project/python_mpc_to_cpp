"""
DC Servo Motor control with MPC

References:
A. Bemporad and E. Mosca, "Fulfilling hard constraints in uncertain linear systems
 by reference managing," Automatica, vol. 34, no. 4, pp. 451-461, 1998.
"""

import os
import sys
sys.path.append(os.getcwd())

import math
import numpy as np
import control

from mpc_utility.state_space_utility import SymbolicStateSpace
from python_mpc.linear_mpc import LTI_MPC_NoConstraints, LTI_MPC
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

    delta_U_min = np.array([[-101.0]])
    delta_U_max = np.array([[102.0]])
    U_min = np.array([[-301.0]])
    U_max = np.array([[302.0]])
    Y_min = np.array([[-10.0], [-100.0]])
    Y_max = np.array([[10.0], [100.0]])

    mpc = LTI_MPC(ideal_plant_model, Np=Np, Nc=Nc,
                  Weight_U=Weight_U, Weight_Y=Weight_Y,
                  delta_U_min=delta_U_min, delta_U_max=delta_U_max,
                  U_min=U_min, U_max=U_max,
                  Y_min=Y_min, Y_max=Y_max)

    # You can create cpp header which can easily define lti_mpc as C++ code
    # deployed_file_names = LinearMPC_Deploy.generate_LTI_MPC_NC_cpp_code(
    #     lti_mpc, number_of_delay=Number_of_Delay)
    # print(deployed_file_names)

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
        ref = np.array([[input_signal[i, 0]], [0.0]])
        # U = lti_mpc.update(ref, y_measured)
        U = mpc.update(ref, y_measured)

        plotter.append_name(ref, "ref")
        plotter.append_name(U, "U")
        plotter.append_name(y_measured, "y_measured")
        plotter.append_name(X, "X")

    plotter.assign("ref", position=(0, 0), column=0, row=0, x_sequence=time)
    plotter.assign("y_measured", position=(0, 0),
                   column=0, row=0, x_sequence=time)
    plotter.assign("ref", position=(0, 1), column=1, row=0, x_sequence=time)
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
