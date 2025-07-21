/**
 * @file two_wheel_vehicle_model.cpp
 * @brief Simulation of a two-wheel vehicle model with adaptive MPC (Model
 * Predictive Control) and no constraints.
 *
 * This program simulates the motion of a two-wheel vehicle using an adaptive
 * MPC controller. The reference trajectory is generated to include a straight
 * segment followed by a curve, and the controller attempts to track this
 * reference using the vehicle model.
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
// You need to run "two_wheel_vehicle_model.py" before running this code.
#include "two_wheel_vehicle_model_ada_mpc.hpp"

#include "python_control.hpp"
#include "python_mpc.hpp"
#include "python_numpy.hpp"

using namespace PythonNumpy;
using namespace PythonControl;
using namespace PythonMPC;

/* Create Reference */
struct ReferenceSequence {
  std::vector<double> x_sequence;
  std::vector<double> y_sequence;
  std::vector<double> theta_sequence;
  std::vector<double> r_sequence;
  std::vector<double> V_sequence;
};

ReferenceSequence create_reference(const std::vector<double> &time,
                                   double delta_time, double simulation_time) {

  constexpr double PI = 3.14159265358979323846;
  const double vehicle_speed = 15.0;
  const double curve_yaw_rate = PI / 5.0;
  const double curve_timing = 2.0;
  const double yaw_ref = PI;

  const size_t time_size = time.size();

  ReferenceSequence ref;
  ref.x_sequence.resize(time_size, 0.0);
  ref.y_sequence.resize(time_size, 0.0);
  ref.theta_sequence.resize(time_size, 0.0);
  ref.r_sequence.resize(time_size, 0.0);
  ref.V_sequence.resize(time_size, 0.0);

  for (size_t i = 0; i < time_size; ++i) {
    if (time[i] < curve_timing) {
      if (i > 0) {
        ref.x_sequence[i] = ref.x_sequence[i - 1] + vehicle_speed * delta_time;
      } else {
        ref.x_sequence[i] = vehicle_speed * delta_time;
      }
      ref.y_sequence[i] = 0.0;
      ref.theta_sequence[i] = 0.0;
      ref.r_sequence[i] = 0.0;
      ref.V_sequence[i] = vehicle_speed;

    } else if (time[i] > curve_timing &&
               (i == 0 || ref.theta_sequence[i - 1] < yaw_ref)) {

      double prev_theta = (i > 0) ? ref.theta_sequence[i - 1] : 0.0;
      double prev_x = (i > 0) ? ref.x_sequence[i - 1] : 0.0;
      double prev_y = (i > 0) ? ref.y_sequence[i - 1] : 0.0;

      ref.x_sequence[i] =
          prev_x + vehicle_speed * delta_time * std::cos(prev_theta);
      ref.y_sequence[i] =
          prev_y + vehicle_speed * delta_time * std::sin(prev_theta);
      ref.theta_sequence[i] = prev_theta + curve_yaw_rate * delta_time;

      if (ref.theta_sequence[i] > yaw_ref) {
        ref.theta_sequence[i] = yaw_ref;
      }

      ref.r_sequence[i] = curve_yaw_rate;
      ref.V_sequence[i] = vehicle_speed;

    } else {
      double prev_theta = (i > 0) ? ref.theta_sequence[i - 1] : 0.0;
      double prev_x = (i > 0) ? ref.x_sequence[i - 1] : 0.0;
      double prev_y = (i > 0) ? ref.y_sequence[i - 1] : 0.0;

      ref.x_sequence[i] =
          prev_x + vehicle_speed * delta_time * std::cos(prev_theta);
      ref.y_sequence[i] =
          prev_y + vehicle_speed * delta_time * std::sin(prev_theta);
      ref.theta_sequence[i] = prev_theta;

      ref.r_sequence[i] = 0.0;
      ref.V_sequence[i] = vehicle_speed;
    }
  }

  return ref;
}

int main(void) {
  /* Simulation Setting */
  constexpr double SIMULATION_TIME = 5.0;
  constexpr double DELTA_TIME = 0.01;
  constexpr std::size_t MAX_STEP =
      static_cast<std::size_t>(SIMULATION_TIME / DELTA_TIME);

  std::vector<double> time = std::vector<double>(MAX_STEP, 0.0);
  for (std::size_t i = 0; i < MAX_STEP; ++i) {
    time[i] = i * DELTA_TIME;
  }

  /* Define MPC */
  constexpr std::size_t STATE_SIZE =
      two_wheel_vehicle_model_ada_mpc::STATE_SIZE;
  constexpr std::size_t INPUT_SIZE =
      two_wheel_vehicle_model_ada_mpc::INPUT_SIZE;
  constexpr std::size_t OUTPUT_SIZE =
      two_wheel_vehicle_model_ada_mpc::OUTPUT_SIZE;

  auto ada_mpc_nc = two_wheel_vehicle_model_ada_mpc::make();

  two_wheel_vehicle_model_ada_mpc_ekf_parameter::Parameter_Type parameters;

  StateSpaceState_Type<double, STATE_SIZE> X;
  X.template set<0, 0>(static_cast<double>(0.0));
  X.template set<1, 0>(static_cast<double>(0.0));
  X.template set<2, 0>(static_cast<double>(0.0));
  X.template set<3, 0>(static_cast<double>(0.0));
  X.template set<4, 0>(static_cast<double>(0.0));
  X.template set<5, 0>(static_cast<double>(10.0));

  StateSpaceInput_Type<double, INPUT_SIZE> U;
  StateSpaceOutput_Type<double, OUTPUT_SIZE> Y;

  two_wheel_vehicle_model_ada_mpc::Ref_Type ref;
  ReferenceSequence reference_sequence =
      create_reference(time, DELTA_TIME, SIMULATION_TIME);

  /* Simulation */
  for (std::size_t sim_step = 0; sim_step < MAX_STEP; ++sim_step) {
    /* system response */
    X = two_wheel_vehicle_model_ada_mpc_ekf_state_function::function(
        X, U, parameters);
    Y = two_wheel_vehicle_model_ada_mpc_ekf_measurement_function::function(
        X, parameters);

    /* controller */
    ref(0, 0) = reference_sequence.x_sequence[sim_step];
    ref(1, 0) = reference_sequence.y_sequence[sim_step];
    ref(2, 0) = reference_sequence.theta_sequence[sim_step];
    ref(3, 0) = reference_sequence.r_sequence[sim_step];
    ref(4, 0) = reference_sequence.V_sequence[sim_step];

    U = ada_mpc_nc.update_manipulation(ref, Y);

    std::cout << "Y_0: " << Y(0, 0) << ", ";
    std::cout << "Y_1: " << Y(1, 0) << ", ";
    std::cout << "Y_2: " << Y(2, 0) << ", ";
    std::cout << "Y_3: " << Y(3, 0) << ", ";
    std::cout << "Y_4: " << Y(4, 0) << ", ";
    std::cout << "U_0: " << U(0, 0) << ", ";
    std::cout << "U_1: " << U(1, 0) << ", ";

    std::cout << std::endl;
  }

  return 0;
}
