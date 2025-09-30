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
namespace SubstituteReferenceTrajectory {

template <typename ReferenceTrajectory_Type, typename Reference_Type,
          std::size_t I, std::size_t J_idx>
struct Column {
  static inline void compute(ReferenceTrajectory_Type &reference_trajectory,
                             const Reference_Type &reference) {
    reference_trajectory.template set<I, J_idx>(
        reference.template get<I, J_idx>());
    Column<ReferenceTrajectory_Type, Reference_Type, I, (J_idx - 1)>::compute(
        reference_trajectory, reference);
  }
};

template <typename ReferenceTrajectory_Type, typename Reference_Type,
          std::size_t I>
struct Column<ReferenceTrajectory_Type, Reference_Type, I, 0> {
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
    Column<ReferenceTrajectory_Type, Reference_Type, I_idx, (N - 1)>::compute(
        reference_trajectory, reference);
    Row<ReferenceTrajectory_Type, Reference_Type, M, N, (I_idx - 1)>::compute(
        reference_trajectory, reference);
  }
};

template <typename ReferenceTrajectory_Type, typename Reference_Type,
          std::size_t M, std::size_t N>
struct Row<ReferenceTrajectory_Type, Reference_Type, M, N, 0> {
  static inline void compute(ReferenceTrajectory_Type &reference_trajectory,
                             const Reference_Type &reference) {
    Column<ReferenceTrajectory_Type, Reference_Type, 0, (N - 1)>::compute(
        reference_trajectory, reference);
  }
};

template <std::size_t COLS, std::size_t Np, typename ReferenceTrajectory_Type,
          typename Reference_Type>
inline void substitute(ReferenceTrajectory_Type &reference_trajectory,
                       const Reference_Type &reference) {

  Row<ReferenceTrajectory_Type, Reference_Type, COLS, Np, (COLS - 1)>::compute(
      reference_trajectory, reference);
}

} // namespace SubstituteReferenceTrajectory

template <std::size_t ROWS, std::size_t NP, typename ReferenceTrajectory_Type,
          typename Reference_Type>
inline typename std::enable_if<(ROWS > 1), void>::type
substitute_reference(ReferenceTrajectory_Type &reference_trajectory,
                     const Reference_Type &reference) {
  static_assert(
      ReferenceTrajectory_Type::ROWS == (NP + 1),
      "ROWS of ReferenceTrajectory_Type must be equal to NP + 1 when ROWS > 1");

  constexpr std::size_t M = ReferenceTrajectory_Type::COLS;

  SubstituteReferenceTrajectory::substitute<M, NP>(reference_trajectory,
                                                   reference);

  PythonNumpy::set_row<NP>(reference_trajectory,
                           PythonNumpy::get_row<NP - 1>(reference));
}

namespace SubstituteReferenceVector {

template <typename ReferenceTrajectory_Type, typename Reference_Type,
          std::size_t M, std::size_t N, std::size_t I, std::size_t J_idx>
struct Column {
  static inline void compute(ReferenceTrajectory_Type &reference_trajectory,
                             const Reference_Type &reference) {
    reference_trajectory.template set<I, J_idx>(reference.template get<I, 0>());
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

  constexpr std::size_t M = ReferenceTrajectory_Type::COLS;
  constexpr std::size_t N = ReferenceTrajectory_Type::ROWS;

  static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");
  Row<ReferenceTrajectory_Type, Reference_Type, M, N, (M - 1)>::compute(
      reference_trajectory, reference);
}

} // namespace SubstituteReferenceVector

template <std::size_t ROWS, std::size_t Np, typename ReferenceTrajectory_Type,
          typename Reference_Type>
inline typename std::enable_if<(ROWS == 1), void>::type
substitute_reference(ReferenceTrajectory_Type &reference_trajectory,
                     const Reference_Type &reference) {
  static_assert(ROWS == 1, "ROWS must be equal to 1");

  SubstituteReferenceVector::substitute(reference_trajectory, reference);
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

  using CostMatricesReferenceTrajectory_Type =
      PythonNumpy::DenseMatrix_Type<_T, OUTPUT_SIZE, (NP + 1)>;

protected:
  /* Type */
  using _R_Type = Weight_U_Type;

  using _Parameter_Type = typename EKF_Type::Parameter_Type;

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
      : U_horizon(), _kalman_filter(), _sqp_cost_matrices(), _delta_time(0),
        _X_inner_model(), _Y_store(), _cost_function(nullptr),
        _cost_and_gradient_function(nullptr), _hvp_function(nullptr),
        _solver() {}

  NonlinearMPC_TwiceDifferentiable(EKF_Type &kalman_filter,
                                   Cost_Matrices_Type &cost_matrices,
                                   _T delta_time, X_Type X_initial)
      : U_horizon(), _kalman_filter(kalman_filter),
        _sqp_cost_matrices(cost_matrices), _delta_time(delta_time),
        _X_inner_model(X_initial), _Y_store(), _cost_function(),
        _cost_and_gradient_function(), _hvp_function(), _solver() {

    this->_kalman_filter.set_x_hat(X_initial);

    this->_initialize_solver(X_initial);
  }

  /* Copy Constructor */
  NonlinearMPC_TwiceDifferentiable(
      const NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type>
          &input)
      : U_horizon(input.U_horizon), _kalman_filter(input._kalman_filter),
        _sqp_cost_matrices(input._sqp_cost_matrices),
        _delta_time(input._delta_time), _X_inner_model(input._X_inner_model),
        _Y_store(input._Y_store), _cost_function(input._cost_function),
        _cost_and_gradient_function(input._cost_and_gradient_function),
        _hvp_function(input._hvp_function), _solver(input._solver) {}

  NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type> &
  operator=(const NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type>
                &input) {
    if (this != &input) {
      this->U_horizon = input.U_horizon;
      this->_kalman_filter = input._kalman_filter;
      this->_sqp_cost_matrices = input._sqp_cost_matrices;
      this->_delta_time = input._delta_time;
      this->_X_inner_model = input._X_inner_model;
      this->_Y_store = input._Y_store;
      this->_cost_function = input._cost_function;
      this->_cost_and_gradient_function = input._cost_and_gradient_function;
      this->_hvp_function = input._hvp_function;
      this->_solver = input._solver;
    }
    return *this;
  }

  /* Move Constructor */
  NonlinearMPC_TwiceDifferentiable(
      NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type>
          &&input) noexcept
      : U_horizon(std::move(input.U_horizon)),
        _kalman_filter(std::move(input._kalman_filter)),
        _sqp_cost_matrices(std::move(input._sqp_cost_matrices)),
        _delta_time(std::move(input._delta_time)),
        _X_inner_model(std::move(input._X_inner_model)),
        _Y_store(std::move(input._Y_store)),
        _cost_function(std::move(input._cost_function)),
        _cost_and_gradient_function(
            std::move(input._cost_and_gradient_function)),
        _hvp_function(std::move(input._hvp_function)),
        _solver(std::move(input._solver)) {}

  NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type> &
  operator=(NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type>
                &&input) noexcept {
    if (this != &input) {
      this->U_horizon = std::move(input.U_horizon);
      this->_kalman_filter = std::move(input._kalman_filter);
      this->_sqp_cost_matrices = std::move(input._sqp_cost_matrices);
      this->_delta_time = std::move(input._delta_time);
      this->_X_inner_model = std::move(input._X_inner_model);
      this->_Y_store = std::move(input._Y_store);
      this->_cost_function = std::move(input._cost_function);
      this->_cost_and_gradient_function =
          std::move(input._cost_and_gradient_function);
      this->_hvp_function = std::move(input._hvp_function);
      this->_solver = std::move(input._solver);
    }
    return *this;
  }

public:
  /* Setter */
  inline void set_solver_max_iteration(std::size_t max_iteration) {
    this->_solver.set_solver_max_iteration(max_iteration);
  }

  template <typename Reference_Type_In>
  inline void set_reference_trajectory(const Reference_Type_In &reference) {

    static_assert(Reference_Type_In::COLS == OUTPUT_SIZE,
                  "COLS of Reference_Type_In must be equal to OUTPUT_SIZE");
    static_assert((Reference_Type_In::ROWS == NP) ||
                      (Reference_Type_In::ROWS == 1),
                  "ROWS of Reference_Type_In must be equal to NP, or 1");
    // If the input is reference vector, it is copied to all time steps.
    // Reference trajectory of sqp_cost_matrices is (OUTPUT_SIZE, NP + 1),
    // but reference input is (OUTPUT_SIZE, NP)

    CostMatricesReferenceTrajectory_Type reference_trajectory;

    NonlinearMPC_ReferenceTrajectoryOperation::substitute_reference<
        Reference_Type_In::ROWS, NP>(reference_trajectory, reference);

    this->_sqp_cost_matrices.reference_trajectory = reference_trajectory;
  }

  /* Getter */
  inline auto get_solver_step_iterated_number(void) const -> std::size_t {
    return this->_solver.get_solver_step_iterated_number();
  }

  /* Function */
  inline auto calculate_this_U(const U_Horizon_Type &U_horizon_in) -> U_Type {

    auto U = PythonNumpy::get_row<0>(U_horizon_in);

    return U;
  }

  inline void update_parameters(const _Parameter_Type &parameters) {
    // when you use this function, parameters type of EKF and CostMatrices must
    // be same

    this->_kalman_filter.parameters = parameters;
    this->_sqp_cost_matrices.state_space_parameters = parameters;
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
        this->_sqp_cost_matrices.get_U_min_matrix(),
        this->_sqp_cost_matrices.get_U_max_matrix());

    U_latest = this->calculate_this_U(this->U_horizon);

    return U_latest;
  }

protected:
  /* Function */
  inline void _initialize_solver(const X_Type &X_initial) {

    this->_cost_function = [this](const X_Type &X, const U_Horizon_Type &U) ->
        typename X_Type::Value_Type {
          return this->_sqp_cost_matrices.compute_cost(X, U);
        };

    this->_cost_and_gradient_function =
        [this](const X_Type &X, const U_Horizon_Type &U,
               typename X_Type::Value_Type &J, _Gradient_Type &gradient) {
          this->_sqp_cost_matrices.compute_cost_and_gradient(X, U, J, gradient);
        };

    this->_hvp_function = [this](const X_Type &X, const U_Horizon_Type &U,
                                 const _V_Horizon_Type &V) -> _HVP_Type {
      return this->_sqp_cost_matrices.hvp_analytic(X, U, V);
    };

    this->_solver =
        PythonOptimization::make_SQP_ActiveSet_PCG_PLS<Cost_Matrices_Type>();

    auto diag_R = this->_sqp_cost_matrices.get_R().diagonal_vector();
    this->_solver.set_diag_R_full(
        PythonNumpy::concatenate_tile<1, NP, decltype(diag_R)>(diag_R));

    this->_solver.X_initial = X_initial;
    this->_solver.set_solver_max_iteration(NMPC_SOLVER_MAX_ITERATION_DEFAULT);
  }

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
  Cost_Matrices_Type _sqp_cost_matrices;

  _T _delta_time;

  X_Type _X_inner_model;
  Y_Store_Type _Y_store;

  _ConstFunction_Object_Type _cost_function;
  _CostAndGradientFunction_Object_Type _cost_and_gradient_function;
  _HVP_Function_Object_Type _hvp_function;

  _Solver_Type _solver;
};

/* make NonlinearMPC_TwiceDifferentiable */
template <typename EKF_Type, typename Cost_Matrices_Type, typename T>
inline auto make_NonlinearMPC_TwiceDifferentiable(
    EKF_Type &kalman_filter, Cost_Matrices_Type &cost_matrices, T delta_time,
    typename Cost_Matrices_Type::X_Type X_initial)
    -> NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type> {

  static_assert(std::is_same<typename EKF_Type::Value_Type, T>::value,
                "Data type of EKF must be same type as T.");
  static_assert(std::is_same<typename Cost_Matrices_Type::Value_Type, T>::value,
                "Data type of CostMatrices must be same type as T.");

  using NonlinearMPC_TwiceDifferentiable_Type =
      NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type>;

  NonlinearMPC_TwiceDifferentiable_Type nonlinear_mpc(
      kalman_filter, cost_matrices, static_cast<T>(delta_time), X_initial);

  return nonlinear_mpc;
}

/* NonlinearMPC_TwiceDifferentiable Type */
template <typename EKF_Type, typename Cost_Matrices_Type>
using NonlinearMPC_TwiceDifferentiable_Type =
    NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type>;

} // namespace PythonMPC

#endif // __PYTHON_NONLINEAR_MPC_HPP__
