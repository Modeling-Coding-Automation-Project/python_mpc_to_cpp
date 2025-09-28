#ifndef __PYTHON_NONLINEAR_MPC_HPP__
#define __PYTHON_NONLINEAR_MPC_HPP__

#include "mpc_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"
#include "python_optimization.hpp"

#include <functional>
#include <type_traits>

namespace PythonMPC {

static constexpr std::size_t NMPC_SOLVER_MAX_ITERATION_DEFAULT = 20;

template <typename EKF_Type_In, typename Cost_Matrices_Type_In,
          typename ReferenceTrajectory_Type_In>
class NonlinearMPC_TwiceDifferentiable {
protected:
  /* Type */
  using _T = typename EKF_Type_In::Value_Type;

public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE = Cost_Matrices_Type_In::STATE_SIZE;
  static constexpr std::size_t INPUT_SIZE = Cost_Matrices_Type_In::INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = Cost_Matrices_Type_In::OUTPUT_SIZE;

  static constexpr std::size_t NP = Cost_Matrices_Type_In::NP;

  static constexpr std::size_t NUMBER_OF_DELAY = EKF_Type_In::NUMBER_OF_DELAY;

public:
  /* Type */
  using EKF_Type = EKF_Type_In;
  using Cost_Matrices_Type = Cost_Matrices_Type_In;

  using X_Type = typename Cost_Matrices_Type::X_Type;
  using U_Type = typename Cost_Matrices_Type::U_Type;
  using Y_Type = typename Cost_Matrices_Type::Y_Type;

  using U_Horizon_Type = typename Cost_Matrices_Type::U_Horizon_Type;
  using Y_Store_Type =
      PythonControl::DelayedVectorObject<Y_Type, NUMBER_OF_DELAY>;

  using Weight_X_Type = PythonNumpy::DiagMatrix_Type<_T, STATE_SIZE>;
  using Weight_U_Type = PythonNumpy::DiagMatrix_Type<_T, INPUT_SIZE>;
  using Weight_Y_Type = PythonNumpy::DiagMatrix_Type<_T, OUTPUT_SIZE>;

  using ReferenceTrajectory_Type = ReferenceTrajectory_Type_In;

protected:
  /* Type */
  using _Solver_Type =
      PythonOptimization::SQP_ActiveSet_PCG_PLS_Type<Cost_Matrices_Type>;

  using _Gradient_Type = U_Horizon_Type;
  using _V_Horizon_Type = U_Horizon_Type;
  using _HVP_Type = U_Horizon_Type;

  using _ConstFunction_Object_Type =
      PythonOptimization::CostFunction_Object<X_Type, U_Horizon_Type>;
  using _CostAndGradientFunction_Object_Type =
      PythonOptimization::CostAndGradientFunction_Object<X_Type, U_Horizon_Type,
                                                         _Gradient_Type>;
  using _HVP_Function_Object_Type =
      PythonOptimization::HVP_Function_Object<X_Type, U_Horizon_Type,
                                              _V_Horizon_Type, _HVP_Type>;

public:
  /* Constructor */
  NonlinearMPC_TwiceDifferentiable()
      : U_horizon(), _cost_matrices(), _kalman_filter(), _delta_time(0),
        _X_inner_model(), _Y_store(), _cost_function(nullptr),
        _cost_and_gradient_function(nullptr), _hvp_function(nullptr),
        _solver() {}

  NonlinearMPC_TwiceDifferentiable(EKF_Type &kalman_filter,
                                   Cost_Matrices_Type &cost_matrices,
                                   _T delta_time, X_Type X_initial)
      : U_horizon(), _cost_matrices(cost_matrices),
        _kalman_filter(kalman_filter), _delta_time(delta_time),
        _X_inner_model(X_initial), _Y_store(), _cost_function(),
        _cost_and_gradient_function(), _hvp_function(), _solver() {

    this->_cost_function = [this](const X_Type &X, const U_Horizon_Type &U) ->
        typename X_Type::Value_Type {
          return this->_cost_matrices.compute_cost(X, U);
        };

    this->_cost_and_gradient_function =
        [this](const X_Type &X, const U_Horizon_Type &U,
               typename X_Type::Value_Type &J, _Gradient_Type &gradient) {
          this->_cost_matrices.compute_cost_and_gradient(X, U, J, gradient);
        };

    this->_hvp_function = [this](const X_Type &X, const U_Horizon_Type &U,
                                 const _V_Horizon_Type &V) -> _HVP_Type {
      return this->_cost_matrices.hvp_analytic(X, U, V);
    };

    this->_solver =
        PythonOptimization::make_SQP_ActiveSet_PCG_PLS<Cost_Matrices_Type>();

    this->_solver.X_initial = X_initial;
    this->_solver.set_solver_max_iteration(NMPC_SOLVER_MAX_ITERATION_DEFAULT);
  }

public:
  /* Setter */
  inline void set_solver_max_iteration(std::size_t max_iteration) {
    this->_solver.set_solver_max_iteration(max_iteration);
  }

  /* Function */
  inline auto update_manipulation(ReferenceTrajectory_Type &reference,
                                  const Y_Type &Y) -> U_Type {}

public:
  /* Variable */
  U_Horizon_Type U_horizon;

protected:
  /* Variable */
  EKF_Type _kalman_filter;
  Cost_Matrices_Type _cost_matrices;

  _T _delta_time;

  X_Type _X_inner_model;
  Y_Store_Type _Y_store;

  _ConstFunction_Object_Type _cost_function;
  _CostAndGradientFunction_Object_Type _cost_and_gradient_function;
  _HVP_Function_Object_Type _hvp_function;

  _Solver_Type _solver;
};

} // namespace PythonMPC

#endif // __PYTHON_NONLINEAR_MPC_HPP__
