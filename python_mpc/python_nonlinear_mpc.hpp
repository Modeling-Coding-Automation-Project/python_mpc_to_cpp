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

namespace NonlinearMPC_ReferenceTrajectoryOperation {

// Unroll nested loops for copying reference -> reference_trajectory
namespace SubstituteReference {

template <typename ReferenceTrajectory_Type, typename Reference_Type,
          std::size_t M, std::size_t N, std::size_t I, std::size_t J_idx>
struct Column {
  static inline void compute(ReferenceTrajectory_Type &reference_trajectory,
                             const Reference_Type &reference) {
    reference_trajectory.template set<I, J_idx>(
        reference.template get<I, J_idx>());
    Column<ReferenceTrajectory_Type, Reference_Type, M, N, I,
           (J_idx - 1)>::compute(reference_trajectory, reference);
  }
};

template <typename ReferenceTrajectory_Type, typename Reference_Type,
          std::size_t M, std::size_t N, std::size_t I>
struct Column<ReferenceTrajectory_Type, Reference_Type, M, N, I, 0> {
  static inline void compute(ReferenceTrajectory_Type &reference_trajectory,
                             const Reference_Type &reference) {
    reference_trajectory.template set<I, 0>(reference.template get<I, 0>());
  }
};

template <typename ReferenceTrajectory_Type, typename Reference_Type,
          std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static inline void compute(ReferenceTrajectory_Type &reference_trajectory,
                             const Reference_Type &reference) {
    Column<ReferenceTrajectory_Type, Reference_Type, M, N, I_idx,
           (N - 1)>::compute(reference_trajectory, reference);
    Row<ReferenceTrajectory_Type, Reference_Type, M, N, (I_idx - 1)>::compute(
        reference_trajectory, reference);
  }
};

template <typename ReferenceTrajectory_Type, typename Reference_Type,
          std::size_t M, std::size_t N>
struct Row<ReferenceTrajectory_Type, Reference_Type, M, N, 0> {
  static inline void compute(ReferenceTrajectory_Type &reference_trajectory,
                             const Reference_Type &reference) {
    Column<ReferenceTrajectory_Type, Reference_Type, M, N, 0, (N - 1)>::compute(
        reference_trajectory, reference);
  }
};

template <typename ReferenceTrajectory_Type, typename Reference_Type>
inline void substitute(ReferenceTrajectory_Type &reference_trajectory,
                       const Reference_Type &reference) {
  constexpr std::size_t M = Reference_Type::COLS; // columns (i)
  constexpr std::size_t N = Reference_Type::ROWS; // rows (j)
  static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");
  Row<ReferenceTrajectory_Type, Reference_Type, M, N, (M - 1)>::compute(
      reference_trajectory, reference);
}

} // namespace SubstituteReference

template <std::size_t ROWS, std::size_t Np, typename ReferenceTrajectory_Type,
          typename Reference_Type>
inline typename std::enable_if<(ROWS > 1), void>::type
substitute_reference(ReferenceTrajectory_Type &reference_trajectory,
                     const Reference_Type &reference) {
  static_assert(ROWS == (Np + 1), "ROWS must be equal to Np + 1 when ROWS > 1");
  SubstituteReference::substitute(reference_trajectory, reference);
}

template <std::size_t ROWS, std::size_t Np, typename ReferenceTrajectory_Type,
          typename Reference_Type>
inline typename std::enable_if<(ROWS == 1), void>::type
calculate_each_ref_sub_Y(ReferenceTrajectory_Type &reference_trajectory,
                         const Reference_Type &reference) {
  static_assert(ROWS == 1, "ROWS must be equal to 1");

  for (std::size_t i = 0; i < Reference_Type::COLS; ++i) {
    for (std::size_t j = 0; j < Reference_Type::ROWS; ++j) {
      reference_trajectory.template set<i, j>(reference.template get<i, 0>());
    }
  }
}

} // namespace NonlinearMPC_ReferenceTrajectoryOperation

template <typename EKF_Type_In, typename Cost_Matrices_Type_In>
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

  using ReferenceTrajectory_Type =
      PythonNumpy::DenseMatrix_Type<_T, OUTPUT_SIZE, (NP + 1)>;

protected:
  /* Type */
  using _Parameter_Type = typename Cost_Matrices_Type::Parameter_Type;

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

  template <typename Reference_Type_In>
  inline void set_reference_trajectory(const Reference_Type_In &reference) {

    ReferenceTrajectory_Type reference_trajectory;

    this->_cost_matrices.set_reference_trajectory(reference_trajectory);
  }

  /* Getter */
  inline auto get_solver_step_iterated_number(void) const -> std::size_t {
    return this->_solver.get_solver_step_iterated_number();
  }

  /* Function */
  inline auto calculate_this_U(const U_Horizon_Type &U_horizon) -> U_Type {

    auto U = PythonNumpy::get_row<0, _T, INPUT_SIZE, 1>(U_horizon);

    return U;
  }

  inline void update_parameters(const _Parameter_Type &parameters) {

    this->_kalman_filter.parameters = parameters;
    this->_cost_matrices.state_space_parameters = parameters;
  }

  template <typename Reference_Type_In>
  inline auto update_manipulation(Reference_Type_In &reference, const Y_Type &Y)
      -> U_Type {

    auto U_latest = this->calculate_this_U(this->U_horizon);

    this->_kalman_filter.predict_and_update(U_latest, Y);

    auto X = this->_kalman_filter.get_x_hat();

    X_Type X_compensated;
    Y_Type Y_compensated;
    this->_compensate_X_Y_delay(X, Y, X_compensated, Y_compensated);

    this->set_reference_trajectory(reference);

    this->U_horizon = this->_solver.solve(
        this->U_horizon, this->_cost_and_gradient_function,
        this->_cost_function, this->_hvp_function, X_compensated,
        this->_cost_matrices.get_U_min_matrix(),
        this->_cost_matrices.get_U_max_matrix());

    U_latest = this->calculate_this_U(this->U_horizon);

    return U_latest;
  }

protected:
  /* Function */
  inline void _compensate_X_Y_delay(const X_Type &X_in, const Y_Type &Y_in,
                                    X_Type &X_out, Y_Type &Y_out) {

    AdaptiveMPC_Operation::compensate_X_Y_delay<NUMBER_OF_DELAY>(
        X_in, Y_in, X_out, Y_out, this->_Y_store, this->_kalman_filter);
  }

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
