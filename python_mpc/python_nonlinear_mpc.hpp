#ifndef __PYTHON_NONLINEAR_MPC_HPP__
#define __PYTHON_NONLINEAR_MPC_HPP__

#include "mpc_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"
#include "python_optimization.hpp"

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
  using Value_Type = typename Cost_Matrices_Type::Value_Type;

  using X_Type = typename Cost_Matrices_Type::X_Type;
  using U_Type = typename Cost_Matrices_Type::U_Type;
  using Y_Type = typename Cost_Matrices_Type::Y_Type;

  using U_Horizon_Type = typename Cost_Matrices_Type::U_Horizon_Type;

  using Weight_X_Type = PythonNumpy::DiagMatrix_Type<Value_Type, STATE_SIZE>;
  using Weight_U_Type = PythonNumpy::DiagMatrix_Type<Value_Type, INPUT_SIZE>;
  using Weight_Y_Type = PythonNumpy::DiagMatrix_Type<Value_Type, OUTPUT_SIZE>;

protected:
  /* Type */
  using _Solver_Type = SQP_ActiveSet_PCG_PLS<CostMatrices_Type>;

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
