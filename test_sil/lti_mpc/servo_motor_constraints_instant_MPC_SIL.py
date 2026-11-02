"""Check the generated servo iMPC against Python with identical cycle timing.

Run with --trajectory to also exercise every preview-reference column. Generated
headers, the build directory and the extension module stay in a temporary folder.
The plant, controller settings and reference signals are defined here so this
test can run without importing the sample scripts.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external_libraries" / "MCAP_python_optimization"))
sys.path.insert(0, str(ROOT / "external_libraries" / "MCAP_python_mpc"))
sys.path.insert(0, str(ROOT))

import control
import numpy as np

from external_libraries.MCAP_python_mpc.mpc_utility.state_space_utility import SymbolicStateSpace
from external_libraries.MCAP_python_mpc.python_mpc.linear_mpc_instant import InstantMPC_LTI
from python_mpc.linear_mpc_instant_deploy import InstantMPC_LTI_Deploy
from test_sil.SIL_operator import SIL_CodeGenerator

# Double-precision Python BLAS and scalar C++ accumulate in different orders.
# Use an absolute floor near zero and a small scale-dependent tolerance.
ATOL = 1.0e-9
RTOL = 1.0e-10


def create_plant_model():
    """Create the continuous servo motor plant used by this SIL test."""
    Lshaft = 1.0
    dshaft = 0.02
    shaftrho = 7850.0
    G = 81500.0 * 1.0e6

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

    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-Kth / JL, -Bload / JL, Kth / (gear * JL), 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [Kth / (JM * gear), 0.0, -Kth / (JM * gear ** 2),
         -(Bmotor + Kt ** 2 / R) / JM],
    ])
    B = np.array([[0.0], [0.0], [0.0], [Kt / (R * JM)]])
    C = np.array([[1.0, 0.0, 0.0, 0.0],
                  [Kth, 0.0, -Kth / gear, 0.0]])
    D = np.zeros((2, 1))
    return control.StateSpace(A, B, C, D)


def create_controller(is_reference_trajectory=False):
    """Return a fresh discrete plant and constrained InstantMPC controller."""
    dt = 0.05
    Number_of_Delay = 0
    sys_d = create_plant_model().sample(Ts=dt, method="euler")
    ideal_plant_model = SymbolicStateSpace(
        sys_d.A, sys_d.B, sys_d.C, delta_time=dt,
        Number_of_Delay=Number_of_Delay)

    impc = InstantMPC_LTI(
        ideal_plant_model,
        Np=20, Nc=2,
        Weight_U=np.diag([0.0005]),
        Weight_Y=np.diag([1.0, 0.005]),
        U_min=np.array([[-180.0]]), U_max=np.array([[180.0]]),
        zeta=50.0,
        near_zero_relative_limit=1.0e-5,
        is_reference_trajectory=is_reference_trajectory)
    return sys_d, impc


def create_input_signal(dt, time, Np, is_reference_trajectory=False):
    """Create a pulse target or a sinusoidal target with preview samples."""
    if is_reference_trajectory:
        latest_time = time[-1]
        for i in range(Np):
            time = np.append(time, latest_time + dt * (i + 1))

        freq = 0.1
        amplitude = 1.0
        input_signal = (amplitude * np.sin(2 * np.pi *
                        freq * time)).reshape(-1, 1)
    else:
        # A 20-second pulse switches from one to zero at 10 seconds.
        input_signal = np.where(np.remainder(time, 20.0) < 10.0,
                                1.0, 0.0).reshape(-1, 1)
    return input_signal


def get_reference_signal(input_signal, index, Np,
                         is_reference_trajectory=False):
    """Form the two-output reference; the torque target stays zero."""
    if is_reference_trajectory:
        reference = np.zeros((2, Np))
        for i in range(Np):
            reference[0, i] = input_signal[index + i, 0]
    else:
        reference = np.array([[input_signal[index, 0]], [0.0]])
    return reference


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def check_close(actual, expected, label, step, errors, atol=ATOL):
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise AssertionError(f"{label}, step {step}: non-finite values")
    np.testing.assert_allclose(
        actual, expected, atol=atol, rtol=RTOL,
        err_msg=f"{label}, step {step}",
    )
    errors[label] = max(errors.get(label, 0.0),
                        float(np.max(np.abs(actual - expected))))


def run_comparison(module, plant, initial_controller, references, independent):
    controller = copy.deepcopy(initial_controller)
    module.initialize(plant.A, plant.B, plant.C, plant.D, plant.dt)
    state = np.zeros((plant.nstates, 1))
    input_previous = np.zeros((plant.ninputs, 1))
    errors = {}
    # The same-input plant comparison has no feedback correction of roundoff.
    # Propagate its state tolerance through y=C*x (the torque row has gain >1e3).
    measurement_atol = (ATOL if independent else
                        ATOL * max(1.0, np.linalg.norm(plant.C, ord=np.inf)))

    for step, reference in enumerate(references):
        # Apply u[k-1], obtain x[k]/y[k], then compute u[k] in both languages.
        state = plant.A @ state + plant.B @ input_previous
        measurement = plant.C @ state + plant.D @ input_previous
        if independent:
            input_cpp, state_cpp, measurement_cpp = module.closed_loop_step(
                reference)
        else:
            # First isolate plant arithmetic using the exact same prior input.
            state_cpp, measurement_cpp = module.plant_step(input_previous)
            # Isolate controller arithmetic using identical reference/Y arrays.
            input_cpp = module.update(reference, measurement)

        input_python = controller.update(reference, measurement)
        check_close(state_cpp, state, "state", step, errors)
        check_close(measurement_cpp, measurement, "measurement", step, errors,
                    atol=measurement_atol)
        check_close(input_cpp, input_python, "control input", step, errors)
        input_previous = input_python

    mode = "independent closed loops" if independent else "identical inputs"
    print(f"Instant MPC SIL ({mode}), {len(references)} steps: {errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", action="store_true",
                        help="Use a sinusoidal preview reference.")
    args = parser.parse_args()

    plant, impc = create_controller(args.trajectory)
    time = np.arange(0.0, 20.0, plant.dt)
    signal = create_input_signal(plant.dt, time, impc.Np, args.trajectory)
    references = [get_reference_signal(signal, step, impc.Np, args.trajectory)
                  for step in range(len(time))]
    if not args.trajectory:
        # Guard the pulse transition at 10 seconds.
        np.testing.assert_array_equal(signal[:200, 0], np.ones(200))
        np.testing.assert_array_equal(signal[200:400, 0], np.zeros(200))

    with tempfile.TemporaryDirectory(prefix="instant-mpc-sil-") as temporary:
        with working_directory(temporary):
            deployed = InstantMPC_LTI_Deploy.generate_InstantMPC_LTI_cpp_code(
                impc, file_name="instant_mpc")
            generator = SIL_CodeGenerator(
                deployed, Path(__file__).resolve().parent,
                sil_name="instant_mpc", work_dir=temporary,
            )
            module_path = generator.build_SIL_code()

        spec = importlib.util.spec_from_file_location(
            "InstantMpcSIL", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run_comparison(module, plant, impc, references, independent=False)
        run_comparison(module, plant, impc, references, independent=True)


if __name__ == "__main__":
    main()
