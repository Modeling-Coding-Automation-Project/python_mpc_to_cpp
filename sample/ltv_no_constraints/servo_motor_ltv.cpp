/**
 * @file servo_motor_ltv.cpp
 * @brief Example of discrete-time state-space simulation and LTV MPC control
 *        for a servo motor system with parameter adaptation.
 *
 * This file demonstrates how to set up and simulate a discrete-time state-space
 * model of a servo motor, and how to apply a Linear Time-Varying Model
 * Predictive Controller (LTV MPC) to control the system. The code initializes
 * the state-space matrices (A, B, C, D), constructs the system, and runs a
 * closed-loop simulation where the controller computes the control input at
 * each time step to track a reference signal. During the simulation, plant and
 * controller parameters are updated at specified steps to demonstrate adaptive
 * control and model updating.
 *
 * The simulation prints the system outputs at each step, showing the response
 * of the controlled servo motor to reference changes and parameter updates.
 */
#include <iostream>

/* CAUTION */
// You need to run "servo_motor_ltv.py" before running this code.
#include "mpc_state_space_updater.hpp"
#include "servo_motor_ltv_ltv_mpc.hpp"
#include "servo_motor_ltv_parameters.hpp"

#include "python_control.hpp"
#include "python_mpc.hpp"
#include "python_numpy.hpp"

using namespace PythonNumpy;
using namespace PythonControl;
using namespace PythonMPC;

int main(void) {
  /* Define State Space */
  using A_Type = SparseAvailable<ColumnAvailable<true, true, false, false>,
                                 ColumnAvailable<true, true, true, false>,
                                 ColumnAvailable<false, false, true, true>,
                                 ColumnAvailable<true, false, true, true>>;

  auto A = make_SparseMatrix<A_Type>(
      static_cast<double>(1.0), static_cast<double>(0.05),
      static_cast<double>(-2.5603853840856576),
      static_cast<double>(0.9500002466138069),
      static_cast<double>(0.12801926920428286), static_cast<double>(1.0),
      static_cast<double>(0.05), static_cast<double>(6.400995031689203),
      static_cast<double>(-0.3200497515844601),
      static_cast<double>(0.4900000000000001));

  using B_Type = SparseAvailable<ColumnAvailable<false>, ColumnAvailable<false>,
                                 ColumnAvailable<false>, ColumnAvailable<true>>;

  auto B = make_SparseMatrix<B_Type>(static_cast<double>(0.04999999999999999));

  using C_Type = SparseAvailable<ColumnAvailable<true, false, false, false>,
                                 ColumnAvailable<true, false, true, false>>;

  auto C = make_SparseMatrix<C_Type>(static_cast<double>(1.0),
                                     static_cast<double>(1280.1990063378407),
                                     static_cast<double>(-64.00995031689203));

  using D_Type =
      SparseAvailable<ColumnAvailable<false>, ColumnAvailable<false>>;

  auto D = make_SparseMatrixEmpty<double, 2, 1>();

  double dt = 0.05;

  constexpr std::size_t INPUT_SIZE = 1;

  auto sys = make_DiscreteStateSpace(A, B, C, D, dt);

  /* Define parameters */
  servo_motor_ltv_parameters::Parameter plant_parameters;
  servo_motor_ltv_parameters::Parameter controller_parameters;

  /* Define controller */
  servo_motor_ltv_ltv_mpc::Ref_Type ref;
  for (std::size_t i = 0; i < ref.rows(); ++i) {
    ref(0, i) = 1.0;
  }

  auto lti_mpc_nc = servo_motor_ltv_ltv_mpc::make();

  auto U = make_StateSpaceInput<INPUT_SIZE>(0.0);

  /* State Space Simulation */

  constexpr std::size_t PARAMETER_CHANGE_STEP = 200;
  bool parameter_changed = false;
  constexpr double MPC_UPDATE_STEP = 400;
  bool MPC_updated = false;

  for (std::size_t sim_step = 0; sim_step < 800; ++sim_step) {
    if (!parameter_changed && sim_step >= PARAMETER_CHANGE_STEP) {
      plant_parameters.Mmotor = 250.0;
      mpc_state_space_updater::MPC_StateSpace_Updater::update(plant_parameters,
                                                              sys);
      parameter_changed = true;

      for (std::size_t i = 0; i < ref.rows(); ++i) {
        ref(0, i) = -1.0;
      }
    }

    /* system response */
    sys.update(U);

    /* controller */
    if (!MPC_updated && sim_step >= MPC_UPDATE_STEP) {
      controller_parameters.Mmotor = 250.0;
      lti_mpc_nc.update_parameters(controller_parameters);
      MPC_updated = true;

      for (std::size_t i = 0; i < ref.rows(); ++i) {
        ref(0, i) = 1.0;
      }
    }

    U = lti_mpc_nc.update_manipulation(ref, sys.get_Y());

    std::cout << "Y_0: " << sys.get_Y()(0, 0) << ", ";
    std::cout << "Y_1: " << sys.get_Y()(1, 0) << ", ";
    std::cout << std::endl;
  }

  return 0;
}
