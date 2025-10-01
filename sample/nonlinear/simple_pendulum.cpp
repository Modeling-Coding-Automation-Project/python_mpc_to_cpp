#include <iostream>

#include "simple_pendulum_nonlinear_mpc.hpp"

#include "python_control.hpp"
#include "python_mpc.hpp"
#include "python_numpy.hpp"

using namespace PythonNumpy;
using namespace PythonControl;
using namespace PythonMPC;

int main(void) {
  /* Simulation Setting */
  constexpr double SIMULATION_TIME = 10.0;
  constexpr double DELTA_TIME = 0.05;
  constexpr std::size_t MAX_STEP =
      static_cast<std::size_t>(SIMULATION_TIME / DELTA_TIME);

  constexpr std::size_t NUMBER_OF_DELAY = 0;

  std::vector<double> time = std::vector<double>(MAX_STEP, 0.0);
  for (std::size_t i = 0; i < MAX_STEP; ++i) {
    time[i] = i * DELTA_TIME;
  }

  constexpr std::size_t NP = 10;

  /* Define Nonlinear MPC */
  constexpr std::size_t STATE_SIZE = simple_pendulum_nonlinear_mpc::STATE_SIZE;
  constexpr std::size_t INPUT_SIZE = simple_pendulum_nonlinear_mpc::INPUT_SIZE;
  constexpr std::size_t OUTPUT_SIZE =
      simple_pendulum_nonlinear_mpc::OUTPUT_SIZE;

  auto nonlinear_mpc = simple_pendulum_nonlinear_mpc::make();

  nonlinear_mpc.set_solver_max_iteration(10);

  simple_pendulum_nonlinear_mpc::Parameter_Type parameters;

  auto X_initial = nonlinear_mpc.get_X();

  StateSpaceState_Type<double, STATE_SIZE> X;
  StateSpaceInput_Type<double, INPUT_SIZE> U;
  StateSpaceOutput_Type<double, OUTPUT_SIZE> Y;

  simple_pendulum_nonlinear_mpc::Reference_Type reference;

  for (std::size_t step = 0; step < MAX_STEP; ++step) {
    /* system response */
    X = simple_pendulum_nonlinear_mpc_ekf_state_function::function(X, U,
                                                                   parameters);
    Y = simple_pendulum_nonlinear_mpc_ekf_measurement_function::function(
        X, parameters);

    /* controller */
    U = nonlinear_mpc.update_manipulation(reference, Y);

    std::size_t solver_iteration =
        nonlinear_mpc.get_solver_step_iterated_number();

    std::cout << "Y_0: " << Y(0, 0) << ", ";
    std::cout << "U_0: " << U(0, 0) << ", ";
    std::cout << "iteration: " << solver_iteration << ", ";

    std::cout << std::endl;
  }

  return 0;
}
