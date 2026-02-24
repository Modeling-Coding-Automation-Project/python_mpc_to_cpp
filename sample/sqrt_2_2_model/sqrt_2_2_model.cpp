/**
 * @file sqrt_2_2_model.cpp
 *
 * @brief Simulation of a 2-input 2-output nonlinear model with adaptive MPC
 * (Model Predictive Control) and constraints.
 *
 * Model:
 *   State:  x = [x1, x2]^T
 *   Input:  u = [u1, u2]^T
 *   Output: y = [y1, y2]^T
 *
 *   dx1/dt = x2 + u1
 *   dx2/dt = -x1 + u2 + 1/sqrt(x1)
 *
 *   y1 = x1
 *   y2 = x2
 *
 * The reference trajectory is a piecewise-constant step signal:
 *   Phase 1 (0 – 3 s):  y1_ref = 1.0, y2_ref = 0.0
 *   Phase 2 (3 – 7 s):  y1_ref = 4.0, y2_ref = 0.0
 *   Phase 3 (7 – end):  y1_ref = 2.0, y2_ref = 0.0
 *
 * Usage:
 * - Compile and run after generating the required model files from Python.
 * - Outputs the measured states and control inputs at each simulation step.
 *
 * @note This code assumes the existence of generated C++ headers and types from
 * the Python modeling pipeline.
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

/* CAUTION */
// You need to run "sqrt_2_2_model.py" before running this code.
#include "sqrt_2_2_model_ada_mpc.hpp"

#include "python_control.hpp"
#include "python_mpc.hpp"
#include "python_numpy.hpp"

using namespace PythonNumpy;
using namespace PythonControl;
using namespace PythonMPC;

/* Create Reference */
struct ReferenceSequence {
  std::vector<double> y1_sequence;
  std::vector<double> y2_sequence;
};

ReferenceSequence create_reference(const std::vector<double> &time) {

  const size_t time_size = time.size();

  ReferenceSequence reference;
  reference.y1_sequence.resize(time_size, 0.0);
  reference.y2_sequence.resize(time_size, 0.0);

  for (size_t i = 0; i < time_size; ++i) {
    if (time[i] < 3.0) {
      reference.y1_sequence[i] = 1.0;
      reference.y2_sequence[i] = 0.0;
    } else if (time[i] < 7.0) {
      reference.y1_sequence[i] = 4.0;
      reference.y2_sequence[i] = 0.0;
    } else {
      reference.y1_sequence[i] = 2.0;
      reference.y2_sequence[i] = 0.0;
    }
  }

  return reference;
}

int main(void) {
  /* Simulation Setting */
  constexpr double SIMULATION_TIME = 10.0;
  constexpr double DELTA_TIME = 0.01;
  constexpr std::size_t MAX_STEP =
      static_cast<std::size_t>(SIMULATION_TIME / DELTA_TIME);

  std::vector<double> time = std::vector<double>(MAX_STEP, 0.0);
  for (std::size_t i = 0; i < MAX_STEP; ++i) {
    time[i] = i * DELTA_TIME;
  }

  /* Define MPC */
  constexpr std::size_t STATE_SIZE = sqrt_2_2_model_ada_mpc::STATE_SIZE;
  constexpr std::size_t INPUT_SIZE = sqrt_2_2_model_ada_mpc::INPUT_SIZE;
  constexpr std::size_t OUTPUT_SIZE = sqrt_2_2_model_ada_mpc::OUTPUT_SIZE;

  auto ada_mpc_nc = sqrt_2_2_model_ada_mpc::make();

  sqrt_2_2_model_ada_mpc_ekf_parameter::Parameter_Type parameters;

  StateSpaceState_Type<double, STATE_SIZE> X;
  X.template set<0, 0>(static_cast<double>(1.0));
  X.template set<1, 0>(static_cast<double>(0.0));

  StateSpaceInput_Type<double, INPUT_SIZE> U;
  StateSpaceOutput_Type<double, OUTPUT_SIZE> Y;

  sqrt_2_2_model_ada_mpc::Reference_Type reference;
  ReferenceSequence reference_sequence = create_reference(time);

  /* Simulation */
  for (std::size_t sim_step = 0; sim_step < MAX_STEP; ++sim_step) {
    /* system response */
    X = sqrt_2_2_model_ada_mpc_ekf_state_function::function(X, U, parameters);
    Y = sqrt_2_2_model_ada_mpc_ekf_measurement_function::function(X,
                                                                  parameters);

    /* controller */
    reference(0, 0) = reference_sequence.y1_sequence[sim_step];
    reference(1, 0) = reference_sequence.y2_sequence[sim_step];
    U = ada_mpc_nc.update_manipulation(reference, Y);

    std::cout << "Y_0: " << Y(0, 0) << ", ";
    std::cout << "Y_1: " << Y(1, 0) << ", ";
    std::cout << "U_0: " << U(0, 0) << ", ";
    std::cout << "U_1: " << U(1, 0) << ", ";

    std::cout << std::endl;
  }

  return 0;
}
