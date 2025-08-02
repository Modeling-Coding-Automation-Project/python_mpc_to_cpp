import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_mpc'))

import math
import numpy as np
import sympy as sp
from dataclasses import dataclass

from external_libraries.MCAP_python_mpc.python_mpc.adaptive_mpc import AdaptiveMPC_NoConstraints
from python_mpc.adaptive_mpc_deploy import AdaptiveMPC_Deploy

from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester


def create_model(delta_time: float):
    # define parameters and variables
    m, u, v, r, F_f, F_r = sp.symbols('m u v r F_f F_r', real=True)
    I, l_f, l_r, v_dot, r_dot, V, beta, beta_dot = sp.symbols(
        'I l_f l_r v_dot r_dot V beta beta_dot', real=True)

    # derive equations of two wheel vehicle model
    eq_1 = sp.Eq(m * (v_dot + u * r), F_f + F_r)
    eq_2 = sp.Eq(I * r_dot, l_f * F_f - l_r * F_r)

    lhs = eq_1.lhs.subs({u: V, v_dot: V * beta_dot})
    eq1 = sp.Eq(lhs, eq_1.rhs)

    K_f, K_r, delta, beta_f, beta_r = sp.symbols(
        'K_f K_r delta beta_f beta_r', real=True)

    rhs = eq1.rhs.subs({F_f: -2 * K_f * beta_f, F_r: -2 * K_r * beta_r})
    eq_1 = sp.Eq(eq1.lhs, rhs)

    rhs = eq_2.rhs.subs({F_f: -2 * K_f * beta_f, F_r: -2 * K_r * beta_r})
    eq_2 = sp.Eq(eq_2.lhs, rhs)

    rhs = eq_1.rhs.subs({
        beta_f: beta + (l_f / V) * r - delta,
        beta_r: beta - (l_r / V) * r
    })
    eq_1 = sp.Eq(eq_1.lhs, rhs)

    rhs = eq_2.rhs.subs({
        beta_f: beta + (l_f / V) * r - delta,
        beta_r: beta - (l_r / V) * r
    })
    eq_2 = sp.Eq(eq_2.lhs, rhs)

    eq_vec = [eq_1, eq_2]

    solution = sp.solve(eq_vec, beta_dot, dict=True)
    beta_dot_sol = sp.simplify(solution[0][beta_dot])

    solution = sp.solve(eq_vec, r_dot, dict=True)
    r_dot_sol = sp.simplify(solution[0][r_dot])

    # Define state space model
    accel = sp.symbols('accel', real=True)
    U = sp.Matrix([[delta], [accel]])

    theta, px, py = sp.symbols('theta px py', real=True)
    X = sp.Matrix([[px], [py], [theta], [r], [beta], [V]])
    Y = sp.Matrix([[px], [py], [theta], [r], [V]])

    fxu_continuous = sp.Matrix([
        [V * sp.cos(theta)],
        [V * sp.sin(theta)],
        [r],
        [r_dot_sol],
        [beta_dot_sol],
        [accel],
    ])
    fxu: sp.Matrix = X + fxu_continuous * delta_time

    hx = sp.Matrix([[X[0]], [X[1]], [X[2]], [X[3]], [X[5]]])

    # derive Jacobian
    fxu_jacobian_X = fxu.jacobian(X)
    fxu_jacobian_U = fxu.jacobian(U)
    hx_jacobian = hx.jacobian(X)

    return X, U, Y, \
        fxu, fxu_jacobian_X, fxu_jacobian_U, \
        hx, hx_jacobian


def create_reference(
        time: np.ndarray, delta_time: float, simulation_time: float):

    vehicle_speed = 15.0
    curve_yaw_rate = math.pi / 5.0
    curve_timing = 2.0

    yaw_ref = math.pi

    x_sequence = np.zeros((len(time), 1))
    y_sequence = np.zeros((len(time), 1))
    theta_sequence = np.zeros((len(time), 1))
    r_sequence = np.zeros((len(time), 1))
    V_sequence = np.zeros((len(time), 1))

    for i in range(len(time)):
        if time[i] < curve_timing:
            x_sequence[i, 0] = x_sequence[i - 1, 0] + \
                vehicle_speed * delta_time
            y_sequence[i, 0] = 0.0
            theta_sequence[i, 0] = 0.0
            r_sequence[i, 0] = 0.0
            V_sequence[i, 0] = vehicle_speed

        elif time[i] > curve_timing and theta_sequence[i - 1, 0] < yaw_ref:
            x_sequence[i, 0] = x_sequence[i - 1, 0] + \
                vehicle_speed * delta_time * math.cos(theta_sequence[i - 1, 0])
            y_sequence[i, 0] = y_sequence[i - 1, 0] + \
                vehicle_speed * delta_time * math.sin(theta_sequence[i - 1, 0])
            theta_sequence[i, 0] = theta_sequence[i - 1, 0] + \
                curve_yaw_rate * delta_time
            if theta_sequence[i, 0] > yaw_ref:
                theta_sequence[i, 0] = yaw_ref

            r_sequence[i, 0] = curve_yaw_rate
            V_sequence[i, 0] = vehicle_speed
        else:
            x_sequence[i, 0] = x_sequence[i - 1, 0] + \
                vehicle_speed * delta_time * math.cos(theta_sequence[i - 1, 0])
            y_sequence[i, 0] = y_sequence[i - 1, 0] + \
                vehicle_speed * delta_time * math.sin(theta_sequence[i - 1, 0])
            theta_sequence[i, 0] = theta_sequence[i - 1, 0]

            r_sequence[i, 0] = 0.0
            V_sequence[i, 0] = vehicle_speed

    return x_sequence, y_sequence, theta_sequence, r_sequence, V_sequence


@dataclass
class Parameter:
    m: float = 2000
    l_f: float = 1.4
    l_r: float = 1.6
    I: float = 4000
    K_f: float = 12e3
    K_r: float = 11e3


def main():
    # simulation setup
    sim_delta_time = 0.01
    simulation_time = 20.0

    time = np.arange(0, simulation_time, sim_delta_time)

    X, U, Y, \
        fxu, fxu_jacobian_X, fxu_jacobian_U, \
        hx, hx_jacobian = create_model(sim_delta_time)

    plant_parameters = Parameter()
    controller_parameters = Parameter()

    Q_ekf = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    R_ekf = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])

    Weight_U = np.array([0.1, 0.1])
    Weight_Y = np.array([1.0, 1.0, 0.05, 0.01, 1.0])

    X_initial = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [10.0]])

    Np = 4
    Nc = 1

    Number_of_Delay = 0

    ada_mpc = AdaptiveMPC_NoConstraints(
        delta_time=sim_delta_time,
        X=X, U=U, Y=Y,
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
        Number_of_Delay=Number_of_Delay)

    # You can create cpp header which can easily define MPC as C++ code
    deployed_file_names = AdaptiveMPC_Deploy.generate_Adaptive_MPC_NC_cpp_code(
        ada_mpc)

    current_dir = os.path.dirname(__file__)
    generator = SIL_CodeGenerator(deployed_file_names, current_dir)
    generator.build_SIL_code()

    from test_sil.adaptive_mpc_two_wheel_vehicle import AdaptiveMpcTwoWheelVehicleSIL
    AdaptiveMpcTwoWheelVehicleSIL.initialize()

    # X: px, py, theta, r, beta, V
    x_true = X_initial
    # U: delta, accel
    u = np.array([[0.0], [0.0]])

    # create reference
    x_sequence, y_sequence, theta_sequence, r_sequence, V_sequence = \
        create_reference(time, sim_delta_time, simulation_time)

    y_measured = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
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
            [x_sequence[i, 0]],
            [y_sequence[i, 0]],
            [theta_sequence[i, 0]],
            [r_sequence[i, 0]],
            [V_sequence[i, 0]]
        ])

        u_from_mpc = ada_mpc.update_manipulation(ref, y_measured)

        U_cpp = AdaptiveMpcTwoWheelVehicleSIL.update_manipulation(
            ref, y_measured)

        tester.expect_near(
            u_from_mpc, U_cpp, NEAR_LIMIT,
            "Adaptive MPC two wheel vehicle model SIL, check update_manipulation.")

    tester.throw_error_if_test_failed()


if __name__ == "__main__":
    main()
