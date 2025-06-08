#include <iostream>

/* CAUTION */
// You need to run "servo_motor.py" before running this code.
#include "servo_motor_constraints_lti_mpc.hpp"

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

  /* Define controller */
  servo_motor_constraints_lti_mpc::Ref_Type ref;

  auto lti_mpc = servo_motor_constraints_lti_mpc::make();

  auto U = make_StateSpaceInput<INPUT_SIZE>(0.0);

  /* State Space Simulation */
  for (std::size_t sim_step = 0; sim_step < 200; ++sim_step) {
    /* system response */
    sys.update(U);

    /* controller */
    for (std::size_t i = 0; i < ref.rows(); ++i) {
      ref(0, i) = 1.0;
    }

    U = lti_mpc.update(ref, sys.get_Y());

    std::cout << "Y_0: " << sys.get_Y()(0, 0) << ", ";
    std::cout << "Y_1: " << sys.get_Y()(1, 0) << ", ";
    std::cout << "U_0: " << U(0, 0) << ", ";
    std::cout << std::endl;
  }

  return 0;
}
