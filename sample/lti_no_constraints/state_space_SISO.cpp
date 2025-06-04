#include <iostream>

/* CAUTION */
// You need to run "state_space_SISO.py" before running this code.
#include "state_space_SISO_lti_mpc.hpp"

#include "python_control.hpp"
#include "python_mpc.hpp"
#include "python_numpy.hpp"

using namespace PythonNumpy;
using namespace PythonControl;
using namespace PythonMPC;

int main(void) {
  /* Define State Space */
  auto A = make_DenseMatrix<2, 2>(0.7, 0.2, -0.3, 0.8);
  auto B = make_DenseMatrix<2, 1>(0.1, 0.2);
  auto C = make_DenseMatrix<1, 2>(1.0, 0.0);
  auto D = make_DenseMatrix<1, 1>(0.0);
  double dt = 0.01;

  constexpr std::size_t INPUT_SIZE = 1;
  constexpr std::size_t NUMBER_OF_DELAY = 5;

  auto sys = make_DiscreteStateSpace<NUMBER_OF_DELAY>(A, B, C, D, dt);

  /* Define controller */
  state_space_SISO_lti_mpc::Ref_Type ref;

  auto lti_mpc_nc = state_space_SISO_lti_mpc::make();

  auto U = make_StateSpaceInput<INPUT_SIZE>(0.0);

  /* State Space Simulation */
  for (std::size_t sim_step = 0; sim_step < 50; ++sim_step) {
    /* system response */
    sys.update(U);

    /* controller */
    for (std::size_t i = 0; i < ref.rows(); ++i) {
      ref(0, i) = 1.0;
    }

    U = lti_mpc_nc.update(ref, sys.get_Y());

    std::cout << "X_0: " << sys.get_X()(0, 0) << ", ";
    std::cout << "X_1: " << sys.get_X()(1, 0) << ", ";
    std::cout << "Y: " << sys.get_Y()(0, 0) << ", ";
    std::cout << std::endl;
  }

  return 0;
}
