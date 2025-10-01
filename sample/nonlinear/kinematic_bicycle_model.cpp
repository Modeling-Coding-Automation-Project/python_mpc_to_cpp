#include <iostream>

#include "kinematic_bicycle_model_nonlinear_mpc.hpp"

#include "python_control.hpp"
#include "python_mpc.hpp"
#include "python_numpy.hpp"

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

  StateSpaceState_Type<double, STATE_SIZE> X;
  StateSpaceInput_Type<double, INPUT_SIZE> U;
  StateSpaceOutput_Type<double, OUTPUT_SIZE> Y;

  kinematic_bicycle_model_nonlinear_mpc::Reference_Type reference;

  for (std::size_t step = 0; step < MAX_STEP; ++step) {
    /* system response */
    X = kinematic_bicycle_model_nonlinear_mpc_ekf_state_function::function(
        X, U, parameters);
    Y = kinematic_bicycle_model_nonlinear_mpc_ekf_measurement_function::
        function(X, parameters);

    /* controller */
    U = nonlinear_mpc.update_manipulation(reference, Y);

    std::size_t solver_iteration =
        nonlinear_mpc.get_solver_step_iterated_number();

    std::cout << "Y_0: " << Y(0, 0) << ", ";
    std::cout << "Y_1: " << Y(0, 1) << ", ";
    std::cout << "Y_2: " << Y(0, 2) << ", ";
    std::cout << "Y_3: " << Y(0, 3) << ", ";
    std::cout << "U_0: " << U(0, 0) << ", ";
    std::cout << "U_1: " << U(1, 0) << ", ";
    std::cout << "iteration: " << solver_iteration << ", ";

    std::cout << std::endl;
  }

  return 0;
}
