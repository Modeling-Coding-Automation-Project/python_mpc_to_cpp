/**
 * @file kinematic_bicycle_model.cpp
 *
 * @brief Example of Nonlinear MPC for a kinematic bicycle model with nonlinear
 * dynamics.
 *
 * This code demonstrates the implementation of a Nonlinear Model Predictive
 * Control (MPC) system for a kinematic bicycle model. The vehicle dynamics are
 * defined using symbolic expressions, and the MPC controller is designed to
 * track a reference trajectory for the vehicle's position and orientation. The
 * simulation runs a closed-loop control scenario, where the MPC computes the
 * optimal control inputs at each time step based on the current state and
 * reference. The results are printed to the console, showing the vehicle's
 * position, orientation, control inputs, and the number of solver iterations at
 * each time step.
 */

// You must run "kinematic_bicycle_model.py" to use
// "kinematic_bicycle_model_nonlinear_mpc.hpp"
#include "kinematic_bicycle_model_nonlinear_mpc.hpp"

#include "python_control.hpp"
#include "python_mpc.hpp"
#include "python_numpy.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace PythonNumpy;
using namespace PythonControl;
using namespace PythonMPC;

int main(void) {
  /* Simulation Setting */
  constexpr double SIMULATION_TIME = 60.0;
  constexpr double DELTA_TIME = 0.1;
  constexpr std::size_t MAX_STEP =
      static_cast<std::size_t>(SIMULATION_TIME / DELTA_TIME);

  constexpr std::size_t NUMBER_OF_DELAY = 0;

  std::vector<double> time = std::vector<double>(MAX_STEP, 0.0);
  for (std::size_t i = 0; i < MAX_STEP; ++i) {
    time[i] = i * DELTA_TIME;
  }

  constexpr std::size_t NP = 10;

  /* Define Nonlinear MPC */
  constexpr std::size_t STATE_SIZE =
      kinematic_bicycle_model_nonlinear_mpc::STATE_SIZE;
  constexpr std::size_t INPUT_SIZE =
      kinematic_bicycle_model_nonlinear_mpc::INPUT_SIZE;
  constexpr std::size_t OUTPUT_SIZE =
      kinematic_bicycle_model_nonlinear_mpc::OUTPUT_SIZE;

  auto nonlinear_mpc = kinematic_bicycle_model_nonlinear_mpc::make();

  nonlinear_mpc.set_solver_max_iteration(5);

  kinematic_bicycle_model_nonlinear_mpc::Parameter_Type parameters;

  auto X_initial = nonlinear_mpc.get_X();

  StateSpaceState_Type<double, STATE_SIZE> X = X_initial;
  StateSpaceInput_Type<double, INPUT_SIZE> U;
  StateSpaceOutput_Type<double, OUTPUT_SIZE> Y;

  kinematic_bicycle_model_nonlinear_mpc::ReferenceTrajectory_Type
      reference_trajectory;

  // You must run "kinematic_bicycle_model.py" to generate "reference_path.csv".
  // --- Load reference CSV (px,py,q0,q3) ---
  std::vector<std::array<double, 4>> reference_data;
  {
    std::ifstream ifs("reference_path.csv");
    if (!ifs) {
      std::cerr << "Warning: could not open reference_path.csv, using zeros"
                << std::endl;
    } else {
      std::string line;
      // skip header if present
      if (std::getline(ifs, line)) {
        // check if header contains non-numeric characters
        bool header = false;
        for (char c : line) {
          if (!std::isdigit(c) && c != '+' && c != '-' && c != '.' &&
              c != 'e' && c != 'E' && c != ',') {
            header = true;
            break;
          }
        }
        if (!header) {
          // first line is data
          std::stringstream ss(line);
          std::array<double, 4> row{0.0, 0.0, 0.0, 0.0};
          for (int k = 0; k < 4; ++k) {
            std::string cell;
            if (!std::getline(ss, cell, ','))
              break;
            row[k] = std::stod(cell);
          }
          reference_data.push_back(row);
        }
      }

      // read remaining lines
      while (std::getline(ifs, line)) {
        if (line.empty())
          continue;
        std::stringstream ss(line);
        std::array<double, 4> row{0.0, 0.0, 0.0, 0.0};
        for (int k = 0; k < 4; ++k) {
          std::string cell;
          if (!std::getline(ss, cell, ','))
            break;
          try {
            row[k] = std::stod(cell);
          } catch (...) {
            row[k] = 0.0;
          }
        }
        reference_data.push_back(row);
      }
    }
  }

  const std::size_t reference_length = reference_data.size();

  for (std::size_t step = 0; step < MAX_STEP; ++step) {
    /* system response */
    X = kinematic_bicycle_model_nonlinear_mpc_ekf_state_function::function(
        X, U, parameters);

    double q_norm = std::sqrt(X(2, 0) * X(2, 0) + X(3, 0) * X(3, 0));
    X(2, 0) = X(2, 0) / q_norm;
    X(3, 0) = X(3, 0) / q_norm;

    Y = kinematic_bicycle_model_nonlinear_mpc_ekf_measurement_function::
        function(X, parameters);

    /* controller */
    // Set reference trajectory for NonlinearMPC (OUTPUT_SIZE x NP)
    // reference_trajectory is (OUTPUT_SIZE, NP)
    // We'll fill each column j with the appropriate reference at time index =
    // step + j
    for (std::size_t row = 0; row < OUTPUT_SIZE; ++row) {
      for (std::size_t j = 0; j < NP; ++j) {
        std::size_t index = step + j;
        if (reference_length == 0) {
          reference_trajectory(row, j) = 0.0;
          continue;
        }
        if (index >= reference_length)
          index = reference_length - 1;
        double value = 0.0;
        switch (row) {
        case 0:
          value = reference_data[index][0];
          break;
        case 1:
          value = reference_data[index][1];
          break;
        case 2:
          value = reference_data[index][2];
          break;
        case 3:
          value = reference_data[index][3];
          break;
        default:
          value = 0.0;
        }
        reference_trajectory(row, j) = value;
      }
    }

    U = nonlinear_mpc.update_manipulation(reference_trajectory, Y);

    std::size_t solver_iteration =
        nonlinear_mpc.get_solver_step_iterated_number();

    double yaw = 2.0 * std::atan2(Y(3, 0), Y(2, 0));

    std::cout << "px: " << Y(0, 0) << ", ";
    std::cout << "py: " << Y(1, 0) << ", ";
    std::cout << "yaw: " << yaw << ", ";
    std::cout << "v: " << U(0, 0) << ", ";
    std::cout << "delta: " << U(1, 0) << ", ";
    std::cout << "iteration: " << solver_iteration << ", ";

    std::cout << std::endl;
  }

  return 0;
}
