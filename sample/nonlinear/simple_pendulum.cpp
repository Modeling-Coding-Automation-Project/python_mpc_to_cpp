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

  auto nonlinear_mpc = simple_pendulum_nonlinear_mpc::make();

  return 0;
}
