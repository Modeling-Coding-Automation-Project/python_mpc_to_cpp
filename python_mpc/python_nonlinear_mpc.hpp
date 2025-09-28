#ifndef __PYTHON_NONLINEAR_MPC_HPP__
#define __PYTHON_NONLINEAR_MPC_HPP__

#include "mpc_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <functional>
#include <type_traits>

namespace PythonMPC {

template <typename Cost_Matrices_Type> class NonlinearMPC_TwiceDifferentiable {
public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE = Cost_Matrices_Type::STATE_SIZE;
  static constexpr std::size_t INPUT_SIZE = Cost_Matrices_Type::INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = Cost_Matrices_Type::OUTPUT_SIZE;

  static constexpr std::size_t NP = Cost_Matrices_Type::NP;

public:
  /* Type */

public:
  /* Constructor */

public:
  /* Function */

public:
  /* Variable */

protected:
  /* Variable */
};

} // namespace PythonMPC

#endif // __PYTHON_NONLINEAR_MPC_HPP__
