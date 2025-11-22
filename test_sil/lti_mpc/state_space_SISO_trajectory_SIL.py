import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_mpc'))

import numpy as np

from external_libraries.MCAP_python_mpc.mpc_utility.state_space_utility import SymbolicStateSpace
from external_libraries.MCAP_python_mpc.python_mpc.linear_mpc import LTI_MPC_NoConstraints
from python_mpc.linear_mpc_deploy import LinearMPC_Deploy
from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester

from sample.simulation_manager.signal_edit.sampler import PulseGenerator

PATH_FOLLOWING = True


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
        is_reference_trajectory=PATH_FOLLOWING)

    deployed_file_names = LinearMPC_Deploy.generate_LTI_MPC_NC_cpp_code(
        lti_mpc)

    current_dir = os.path.dirname(__file__)
    generator = SIL_CodeGenerator(deployed_file_names, current_dir)
    generator.build_SIL_code()

    from test_sil.lti_mpc import LtiMpcSIL
    LtiMpcSIL.initialize()

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

    y_measured = Y
    y_store = [Y] * (Number_of_Delay + 1)
    delay_index = 0

    tester = MCAPTester()
    NEAR_LIMIT = 1e-5

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

        U_cpp = LtiMpcSIL.update(ref, y_measured)

        tester.expect_near(
            U, U_cpp, NEAR_LIMIT,
            "LTI MPC state space SISO SIL, check update.")

    tester.throw_error_if_test_failed()


if __name__ == "__main__":
    main()
