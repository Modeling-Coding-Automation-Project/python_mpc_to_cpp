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

from external_libraries.MCAP_python_mpc.python_mpc.adaptive_mpc import AdaptiveMPC
from python_mpc.adaptive_mpc_deploy import AdaptiveMPC_Deploy

from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester


def create_model(delta_time: float):
    # define state, input, and output symbols
    x1, x2 = sp.symbols('x1 x2', real=True)
    u1, u2 = sp.symbols('u1 u2', real=True)

    X = sp.Matrix([[x1], [x2]])
    U = sp.Matrix([[u1], [u2]])
    Y = sp.Matrix([[x1], [x2]])

    # continuous-time dynamics
    #   dx1/dt = x2 + u1
    #   dx2/dt = -x1 + u2 + 1/sqrt(x1)
    fxu_continuous = sp.Matrix([
        [x2 + u1],
        [-x1 + u2 + 1 / sp.sqrt(x1)],
    ])

    # forward-Euler discretisation: x(k+1) = x(k) + f(x,u) * dt
    fxu: sp.Matrix = X + fxu_continuous * delta_time

    # measurement function
    hx = sp.Matrix([[x1], [x2]])

    # derive Jacobians
    fxu_jacobian_X = fxu.jacobian(X)
    fxu_jacobian_U = fxu.jacobian(U)
    hx_jacobian = hx.jacobian(X)

    return X, U, Y, \
        fxu, fxu_jacobian_X, fxu_jacobian_U, \
        hx, hx_jacobian


def create_reference(time: np.ndarray):
    """Create a piecewise-constant step reference for y1 and y2.

    Phase 1 (0 – 3 s): y1_ref = 1.0, y2_ref = 0.0  (initial equilibrium)
    Phase 2 (3 – 7 s): y1_ref = 4.0, y2_ref = 0.0
    Phase 3 (7 – end): y1_ref = 2.0, y2_ref = 0.0
    """
    y1_ref = np.zeros((len(time), 1))
    y2_ref = np.zeros((len(time), 1))

    for i in range(len(time)):
        if time[i] < 3.0:
            y1_ref[i, 0] = 1.0
            y2_ref[i, 0] = 0.0
        elif time[i] < 7.0:
            y1_ref[i, 0] = 4.0
            y2_ref[i, 0] = 0.0
        else:
            y1_ref[i, 0] = 2.0
            y2_ref[i, 0] = 0.0

    return y1_ref, y2_ref


@dataclass
class Parameter:
    """Empty parameter dataclass (model has no external parameters)."""
    pass


def main():
    # simulation setup
    sim_delta_time = 0.01
    simulation_time = 10.0

    time = np.arange(0, simulation_time, sim_delta_time)

    X, U, Y, \
        fxu, fxu_jacobian_X, fxu_jacobian_U, \
        hx, hx_jacobian = create_model(sim_delta_time)

    plant_parameters = Parameter()
    controller_parameters = Parameter()

    # EKF covariance matrices  (state dim = 2, output dim = 2)
    Q_ekf = np.diag([1.0, 1.0])
    R_ekf = np.diag([1.0, 1.0])

    # MPC weight vectors
    Weight_U = np.array([0.1, 0.1])
    Weight_Y = np.array([1.0, 1.0])

    # initial state: x1 = 1.0, x2 = 0.0  (equilibrium with u = 0)
    X_initial = np.array([[1.0], [0.0]])

    Np = 20
    Nc = 2

    Number_of_Delay = 0

    # input constraints
    u1_limit = 5.0
    u2_limit = 5.0
    U_min = np.array([
        [-u1_limit],
        [-u2_limit]
    ])
    U_max = np.array([
        [u1_limit],
        [u2_limit]
    ])

    ada_mpc = AdaptiveMPC(
        delta_time=sim_delta_time,
        X=X, U=U,
        X_initial=X_initial,
        fxu=fxu, fxu_jacobian_X=fxu_jacobian_X,
        fxu_jacobian_U=fxu_jacobian_U,
        hx=hx, hx_jacobian=hx_jacobian,
        parameters_struct=controller_parameters,
        Np=Np, Nc=Nc,
        Weight_U=Weight_U,
        Weight_Y=Weight_Y,
        Q_kf=Q_ekf,
        R_kf=R_ekf,
        Number_of_Delay=Number_of_Delay,
        U_min=U_min,
        U_max=U_max
    )

    # You can create cpp header which can easily define MPC as C++ code
    deployed_file_names = AdaptiveMPC_Deploy.generate_Adaptive_MPC_cpp_code(
        ada_mpc)

    current_dir = os.path.dirname(__file__)
    generator = SIL_CodeGenerator(deployed_file_names, current_dir)
    generator.build_SIL_code()

    from test_sil.sqrt_2_2_model import Sqrt22ModelSIL
    Sqrt22ModelSIL.initialize()

    # X: x1, x2
    x_true = X_initial
    # U: u1, u2
    u = np.array([[0.0], [0.0]])

    # create reference
    y1_ref, y2_ref = create_reference(time)

    y_measured = np.array([[1.0], [0.0]])
    y_store = [y_measured] * (Number_of_Delay + 1)
    delay_index = 0

    tester = MCAPTester()
    NEAR_LIMIT = 0.5

    # simulation
    for i in range(round(simulation_time / sim_delta_time)):
        # system response
        if i > 0:
            u = np.copy(u_from_mpc)

        x_true = ada_mpc.state_space_initializer.fxu_function(
            x_true, u, plant_parameters)
        y_store[delay_index] = ada_mpc.state_space_initializer.hx_function(
            x_true, plant_parameters)

        # system delay
        delay_index += 1
        if delay_index > Number_of_Delay:
            delay_index = 0

        y_measured = y_store[delay_index]

        # controller
        ref = np.array([
            [y1_ref[i, 0]],
            [y2_ref[i, 0]]
        ])

        u_from_mpc = ada_mpc.update_manipulation(ref, y_measured)

        U_cpp = Sqrt22ModelSIL.update_manipulation(
            ref, y_measured)

        tester.expect_near(
            u_from_mpc, U_cpp, NEAR_LIMIT,
            "Adaptive MPC sqrt 2 2 model SIL, check update_manipulation.")

    tester.throw_error_if_test_failed()


if __name__ == "__main__":
    main()
