#ifndef __PYTHON_NONLINEAR_MPC_HPP__
#define __PYTHON_NONLINEAR_MPC_HPP__

#include "mpc_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"
#include "python_optimization.hpp"

#include <functional>
#include <type_traits>

namespace PythonMPC {

template <typename EKF_Type_In, typename CostMatrices_Type_In,
          typename ReferenceTrajectory_Type_In>
class NonlinearMPC_TwiceDifferentiable {
protected:
  /* Type */
  using _T = typename EKF_Type_In::Value_Type;

public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE = CostMatrices_Type_In::STATE_SIZE;
  static constexpr std::size_t INPUT_SIZE = CostMatrices_Type_In::INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = CostMatrices_Type_In::OUTPUT_SIZE;

  static constexpr std::size_t NP = CostMatrices_Type_In::NP;

  static constexpr std::size_t NUMBER_OF_DELAY = EKF_Type_In::NUMBER_OF_DELAY;

public:
  /* Type */
  using EKF_Type = EKF_Type_In;
  using CostMatrices_Type = CostMatrices_Type_In;

  using X_Type = typename CostMatrices_Type::X_Type;
  using U_Type = typename CostMatrices_Type::U_Type;
  using Y_Type = typename CostMatrices_Type::Y_Type;

  using U_Horizon_Type = typename CostMatrices_Type::U_Horizon_Type;
  using Y_Store_Type =
      PythonControl::DelayedVectorObject<Y_Type, NUMBER_OF_DELAY>;

  using Weight_X_Type = PythonNumpy::DiagMatrix_Type<_T, STATE_SIZE>;
  using Weight_U_Type = PythonNumpy::DiagMatrix_Type<_T, INPUT_SIZE>;
  using Weight_Y_Type = PythonNumpy::DiagMatrix_Type<_T, OUTPUT_SIZE>;

protected:
  /* Type */
  using _Solver_Type =
      PythonOptimization::SQP_ActiveSet_PCG_PLS_Type<CostMatrices_Type>;

public:
  /* Constructor */
  NonlinearMPC_TwiceDifferentiable() {}

public:
  /* Function */

public:
  /* Variable */
  U_Horizon_Type U_horizon;

protected:
  /* Variable */
  EKF_Type _kalman_filter;
  _T delta_time;

  X_Type _X_inner_model;
  Y_Store_Type _Y_store;

  _Solver_Type _solver;
};

} // namespace PythonMPC

#endif // __PYTHON_NONLINEAR_MPC_HPP__
