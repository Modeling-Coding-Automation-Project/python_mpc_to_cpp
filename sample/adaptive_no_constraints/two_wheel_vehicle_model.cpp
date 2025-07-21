#include <iostream>

/* CAUTION */
// You need to run "two_wheel_vehicle_model.py" before running this code.
#include "two_wheel_vehicle_model_ada_mpc.hpp"

#include "python_control.hpp"
#include "python_mpc.hpp"
#include "python_numpy.hpp"

using namespace PythonNumpy;
using namespace PythonControl;
using namespace PythonMPC;

int main(void) {
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

  /* Simulation */
  for (std::size_t sim_step = 0; sim_step < 1; ++sim_step) {
    /* system response */
    X = two_wheel_vehicle_model_ada_mpc_ekf_state_function::function(
        X, U, parameters);
    Y = two_wheel_vehicle_model_ada_mpc_ekf_measurement_function::function(
        X, parameters);

    /* controller */

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
