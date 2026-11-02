"""Check the generated servo iMPC against Python with identical cycle timing.

Run with --trajectory to also exercise every preview-reference column. Generated
headers, the build directory and the extension module stay in a temporary folder.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external_libraries" / "MCAP_python_mpc"))
sys.path.insert(0, str(ROOT))

import numpy as np

from python_mpc.linear_mpc_instant_deploy import InstantMPC_LTI_Deploy
from sample.lti.servo_motor_constraints_instant_MPC import (
    create_controller, create_input_signal, get_reference_signal,
)
from test_sil.SIL_operator import SIL_CodeGenerator

# Double-precision Python BLAS and scalar C++ accumulate in different orders.
# Use an absolute floor near zero and a small scale-dependent tolerance.
ATOL = 1.0e-9
RTOL = 1.0e-10


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
                        help="Use the demo's sinusoidal preview reference.")
    args = parser.parse_args()

    plant, impc = create_controller(args.trajectory)
    time = np.arange(0.0, 20.0, plant.dt)
    signal = create_input_signal(plant.dt, time, impc.Np, args.trajectory)
    references = [get_reference_signal(signal, step, impc.Np, args.trajectory)
                  for step in range(len(time))]
    if not args.trajectory:
        # Guard the sample's pulse transition, which was absent in the C++ demo.
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
