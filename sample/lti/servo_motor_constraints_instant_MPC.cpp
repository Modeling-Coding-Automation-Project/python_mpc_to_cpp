/**
 * @file servo_motor_constraints_instant_MPC.cpp
 * @brief Example of discrete-time state-space simulation and Instant MPC
 * control for a servo motor system with constraints.
 *
 * This file demonstrates how to set up and simulate a discrete-time state-space
 * model of a servo motor, and apply an Instant Model Predictive Controller
 * (iMPC) with constraints. The code initializes the system matrices, constructs
 * the state-space system, and runs a closed-loop simulation where the iMPC
 * computes the control input at each time step to track a reference signal.
 */
#include <cmath>
#include <iostream>

/* CAUTION */
// You need to run "servo_motor_constraints_instant_MPC.py" before running this
// code.
#include "servo_motor_constraints_instant_MPC_impc.hpp"

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

  auto D = make_SparseMatrixEmpty<double, 2, 1>();

  double dt = 0.05;

  constexpr std::size_t INPUT_SIZE = 1;

  auto sys = make_DiscreteStateSpace(A, B, C, D, dt);

  /* Define controller */
  servo_motor_constraints_instant_MPC_impc::Reference_Type reference;

  auto impc = servo_motor_constraints_instant_MPC_impc::make();

  auto U = make_StateSpaceInput<INPUT_SIZE>(0.0);

  /* State Space Simulation */
  for (std::size_t sim_step = 0; sim_step < 400; ++sim_step) {
    /* Apply the previous input, then measure the updated state. update()
     * stores its output before advancing X, so get_Y() is one sample older. */
    sys.update(U);
    const auto y_measured = sys.output_equation(sys.get_X(), U);

    /* controller */
    for (std::size_t j = 0; j < reference.cols(); ++j) {
      const double time = dt * static_cast<double>(sim_step + j);
      reference(0, j) = reference.cols() == 1
                            ? (std::fmod(time, 20.0) < 10.0 ? 1.0 : 0.0)
                            : std::sin(2.0 * std::acos(-1.0) * 0.1 * time);
      reference(1, j) = 0.0;
    }

    U = impc.update(reference, y_measured);

    std::cout << "Y_0: " << y_measured(0) << ", ";
    std::cout << "Y_1: " << y_measured(1) << ", ";
    std::cout << "U_0: " << U(0) << ", ";
    std::cout << std::endl;
  }

  return 0;
}
