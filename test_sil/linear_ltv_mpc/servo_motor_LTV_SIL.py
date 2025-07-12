import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_mpc'))

import math
import numpy as np
import sympy as sp
from dataclasses import dataclass

from external_libraries.MCAP_python_mpc.mpc_utility.state_space_utility import SymbolicStateSpace
from external_libraries.MCAP_python_mpc.python_mpc.linear_mpc import LTV_MPC_NoConstraints
from python_mpc.linear_mpc_deploy import LinearMPC_Deploy
from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester

from sample.simulation_manager.signal_edit.sampler import PulseGenerator

from external_libraries.MCAP_python_mpc.mpc_utility.state_space_utility_deploy \
    import MPC_STATE_SPACE_UPDATER_FILE_NAME_NO_EXTENSION
from external_libraries.MCAP_python_mpc.mpc_utility.state_space_utility_deploy \
    import MPC_STATE_SPACE_UPDATER_CLASS_NAME
from external_libraries.MCAP_python_mpc.mpc_utility.state_space_utility_deploy \
    import MPC_STATE_SPACE_UPDATER_FUNCTION_NAME


class StateSpaceUpdater:
    @staticmethod
    def update(parameters):
        """
        Updates and returns the state-space matrices (A, B, C, D)
          for the servo motor model using the provided parameters.

        You need to run LTV_MPC_NoConstraints initialization
          before calling this function to ensure that the Python code file exists.

        This function dynamically imports a specified class and
          function to compute the state-space matrices based on the given parameters.
        It executes the import and function call at runtime,
          allowing for flexible updates of the model.

        Args:
            parameters (struct): parameters
              required by the state-space updater function.

        Returns:
            tuple: A tuple containing the updated state-space matrices (A, B, C, D).
        """

        local_vars = {"parameters": parameters}

        exe_code = (
            f"from {MPC_STATE_SPACE_UPDATER_FILE_NAME_NO_EXTENSION} import " +
            MPC_STATE_SPACE_UPDATER_CLASS_NAME + "\n"
            "A, B, C, D = " +
            MPC_STATE_SPACE_UPDATER_CLASS_NAME +
            f".{MPC_STATE_SPACE_UPDATER_FUNCTION_NAME}(parameters)\n"
        )

        exec(exe_code, globals(), local_vars)

        A = local_vars["A"]
        B = local_vars["B"]
        C = local_vars["C"]
        D = local_vars["D"]

        return A, B, C, D


@dataclass
class ServoMotorParameters:
    Lshaft: float = 1.0         # Length of the shaft
    dshaft: float = 0.02        # Diameter of the shaft
    # Density of the shaft material (carbon steel)
    shaftrho: float = 7850.0
    G: float = 81500.0 * 1.0e6  # Shear modulus

    Mmotor: float = 100.0       # Mass of the rotor
    Rmotor: float = 0.1         # Radius of the rotor

    # Viscous friction coefficient of the rotor (A CASO)
    Bmotor: float = 0.1
    R: float = 20.0             # Resistance of the contactor

    Kt: float = 10.0            # Motor constant
    Bload: float = 25.0         # Load viscous friction coefficient


def create_plant_model_ABCD():
    PI = math.pi

    Lshaft = sp.Symbol('Lshaft', real=True)
    dshaft = sp.Symbol('dshaft', real=True)
    shaftrho = sp.Symbol('shaftrho', real=True)
    G = sp.Symbol('G', real=True)

    Mmotor = sp.Symbol('Mmotor', real=True)
    Rmotor = sp.Symbol('Rmotor', real=True)
    # Moment of inertia of the rotor about its axis
    Jmotor = 0.5 * Mmotor * Rmotor ** 2

    Bmotor = sp.Symbol('Bmotor', real=True)
    R = sp.Symbol('R', real=True)
    Kt = sp.Symbol('Kt', real=True)

    gear = 20.0                         # Gear ratio

    Jload = 50.0 * Jmotor               # Nominal moment of inertia of the load
    # Nominal viscous friction coefficient of the load
    Bload = sp.Symbol('Bload', real=True)

    Ip = PI / 32.0 * dshaft ** 4         # Polar moment of inertia of the shaft
    Kth = G * Ip / Lshaft                # Torsional stiffness (Torque/Angle)
    Vshaft = PI * (dshaft ** 2) / 4.0 * Lshaft  # Volume of the shaft
    Mshaft = shaftrho * Vshaft           # Mass of the shaft
    # Moment of inertia of the shaft
    Jshaft = Mshaft * 0.5 * (dshaft ** 2 / 4.0)

    JM = Jmotor
    JL = Jload + Jshaft

    A = sp.Matrix([[0.0, 1.0, 0.0, 0.0],
                  [-Kth / JL, -Bload / JL, Kth / (gear * JL), 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [Kth / (JM * gear), 0.0, -Kth / (JM * gear ** 2), -(Bmotor + Kt ** 2 / R) / JM]])
    A = sp.simplify(A)

    B = sp.Matrix([[0.0],
                  [0.0],
                  [0.0],
                  [Kt / (R * JM)]])
    B = sp.simplify(B)

    C = sp.Matrix([[1.0, 0.0, 0.0, 0.0],
                  [Kth, 0.0, -Kth / gear, 0.0]])
    C = sp.simplify(C)

    D = sp.Matrix([[0.0],
                  [0.0]])
    D = sp.simplify(D)

    return A, B, C, D


def discretize_state_space_euler(
        A: sp.Matrix, B: sp.Matrix, C: sp.Matrix, D: sp.Matrix, dt):

    Ad = sp.eye(A.shape[0]) + dt * A
    Ad = sp.simplify(Ad)

    Bd = dt * B
    Bd = sp.simplify(Bd)

    Cd = C
    Cd = sp.simplify(Cd)

    Dd = D
    Dd = sp.simplify(Dd)

    return Ad, Bd, Cd, Dd


def main():
    # %% create state-space model
    sym_A, sym_B, sym_C, sym_D = create_plant_model_ABCD()

    dt = 0.05
    Number_of_Delay = 0

    sym_Ad, sym_Bd, sym_Cd, sym_Dd = discretize_state_space_euler(
        sym_A, sym_B, sym_C, sym_D, dt)

    ideal_plant_model = SymbolicStateSpace(
        sym_Ad, sym_Bd, sym_Cd, delta_time=dt, Number_of_Delay=Number_of_Delay)

    plant_parameters = ServoMotorParameters()
    controller_parameters = ServoMotorParameters()

    Weight_U = np.diag([0.001])
    Weight_Y = np.diag([1.0, 0.005])

    Np = 20
    Nc = 2

    ltv_mpc = LTV_MPC_NoConstraints(state_space=ideal_plant_model,
                                    parameters_struct=controller_parameters,
                                    Np=Np, Nc=Nc,
                                    Weight_U=Weight_U, Weight_Y=Weight_Y)

    # You can create cpp header which can easily define lti_mpc as C++ code
    deployed_file_names = LinearMPC_Deploy.generate_LTV_MPC_NC_cpp_code(
        ltv_mpc,
        parameters=controller_parameters,
        number_of_delay=Number_of_Delay)

    current_dir = os.path.dirname(__file__)
    generator = SIL_CodeGenerator(deployed_file_names, current_dir)
    generator.build_SIL_code()

    from test_sil.linear_mpc import LinearLtvMpcSIL
    LinearLtvMpcSIL.initialize()

    # %% simulation
    t_sim = 80.0
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
    A, B, C, _ = StateSpaceUpdater.update(plant_parameters)

    X = np.array([[0.0],
                  [0.0],
                  [0.0],
                  [0.0]])
    Y = np.array([[0.0],
                  [0.0]])
    U = np.array([[0.0]])

    y_measured = Y
    y_store = [Y] * (Number_of_Delay + 1)
    delay_index = 0

    PARAMETER_CHANGE_TIME = 20.0
    parameter_changed = False
    MPC_UPDATE_TIME = 40.0
    MPC_updated = False

    tester = MCAPTester()
    NEAR_LIMIT = 1e-5

    for i in range(len(time)):
        if not parameter_changed and time[i] > PARAMETER_CHANGE_TIME:
            plant_parameters.Mmotor = 250.0
            A, B, C, _ = StateSpaceUpdater.update(plant_parameters)
            parameter_changed = True

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

        if not MPC_updated and time[i] > MPC_UPDATE_TIME:
            controller_parameters.Mmotor = 250.0

            ltv_mpc.update_parameters(controller_parameters)
            LinearLtvMpcSIL.update_parameters(controller_parameters.Mmotor)

            MPC_updated = True

        U = ltv_mpc.update_manipulation(ref, y_measured)

        U_cpp = LinearLtvMpcSIL.update_manipulation(ref, y_measured)

        tester.expect_near(
            U, U_cpp, NEAR_LIMIT,
            "Linear MPC servo motor LTV SIL, check update.")

    tester.throw_error_if_test_failed()


if __name__ == "__main__":
    main()
