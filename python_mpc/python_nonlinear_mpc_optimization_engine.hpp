/**
 * @file python_nonlinear_mpc_optimization_engine.hpp
 *
 * @brief Nonlinear Model Predictive Control (NMPC) using ALM/PM Optimization
 * Engine
 *
 * This header file defines the NonlinearMPC_OptimizationEngine class template,
 * which implements a Nonlinear Model Predictive Control (MPC) algorithm using
 * the ALM/PM (Augmented Lagrangian Method / Penalty Method) optimization engine
 * with PANOC as the inner solver.
 *
 * The class integrates an Extended Kalman Filter (EKF) for state estimation
 * and utilizes only first-order gradient information (no Hessian required).
 *
 * Unlike NonlinearMPC_TwiceDifferentiable, this implementation does not require
 * second-order derivative (Hessian) computations, as PANOC uses L-BFGS
 * quasi-Newton approximations and only needs first-order gradient information.
 *
 * The ALM/PM optimizer is used as the unified solver for both constrained and
 * unconstrained problems:
 * - When output constraints are present (N1 > 0), the full ALM outer loop is
 *   activated with Lagrange multiplier updates and penalty parameter
 * adjustment.
 * - When no output constraints exist (N1 = 0), the ALM outer loop reduces to a
 *   single PANOC solve with no augmentation overhead.
 */
#ifndef __PYTHON_NONLINEAR_MPC_OPTIMIZATION_ENGINE_HPP__
#define __PYTHON_NONLINEAR_MPC_OPTIMIZATION_ENGINE_HPP__

#include "mpc_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"
#include "python_optimization.hpp"

#include <functional>
#include <tuple>
#include <type_traits>

namespace PythonMPC {

/* ==========================================================================
 * NMPC Optimization Engine Constants
 * ========================================================================== */
namespace NonlinearMPC_OptimizationEngine_Constants {

static constexpr double PANOC_TOLERANCE_DEFAULT = 1e-4;
static constexpr std::size_t PANOC_LBFGS_MEMORY_DEFAULT = 5;

static constexpr std::size_t ALM_INNER_MAX_ITERATION_DEFAULT = 20;
static constexpr std::size_t ALM_OUTER_MAX_ITERATIONS_DEFAULT = 10;
static constexpr double ALM_EPSILON_TOLERANCE_DEFAULT = 1e-4;
static constexpr double ALM_DELTA_TOLERANCE_DEFAULT = 1e-3;
static constexpr double ALM_INITIAL_INNER_TOLERANCE_DEFAULT = 1e-2;
static constexpr double ALM_INITIAL_PENALTY_DEFAULT = 10.0;

static constexpr double BALL_PROJECTION_RADIUS_DEFAULT = 1e12;

} // namespace NonlinearMPC_OptimizationEngine_Constants

/* ==========================================================================
 * Reference Trajectory Operations (reuse from NonlinearMPC)
 * ========================================================================== */
namespace NonlinearMPC_OptEng_ReferenceTrajectoryOperation {

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

/**
 * @brief Substitutes a reference trajectory based on the number of rows.
 */
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

/**
 * @brief Substitutes a reference trajectory when the reference is a vector.
 */
template <std::size_t ROWS, std::size_t Np, typename ReferenceTrajectory_Type,
          typename Reference_Type>
inline typename std::enable_if<(ROWS == 1), void>::type
substitute_reference(ReferenceTrajectory_Type &reference_trajectory,
                     const Reference_Type &reference) {
  static_assert(ROWS == 1, "ROWS must be equal to 1");

  SubstituteReferenceVector::substitute(reference_trajectory, reference);
}

} // namespace NonlinearMPC_OptEng_ReferenceTrajectoryOperation

/* ==========================================================================
 * NonlinearMPC_OptimizationEngine Class Template
 * ========================================================================== */

/**
 * @brief Nonlinear Model Predictive Control (NMPC) class template using
 * ALM/PM optimization engine with PANOC as the inner solver.
 *
 * This class template implements a Nonlinear Model Predictive Control (MPC)
 * algorithm that uses ALM/PM (Augmented Lagrangian Method / Penalty Method)
 * for optimization. It integrates an Extended Kalman Filter (EKF) for state
 * estimation and only requires first-order gradient information.
 *
 * The ALM/PM optimizer is used as the unified solver regardless of whether
 * output constraints are present. When no output constraints exist, ALM
 * reduces to a single PANOC solve with no augmentation overhead.
 *
 * @tparam EKF_Type_In Type of the Extended Kalman Filter (EKF) used for state
 * estimation. Must provide methods for state prediction and update.
 * @tparam Cost_Matrices_Type_In Type of the cost matrices used in the MPC
 * formulation. Must define state, input, and output sizes, as well as horizon
 * length.
 * @tparam HasOutputConstraints Boolean indicating whether output constraints
 * are present. When true, N1 = OUTPUT_SIZE * (NP + 1); when false, N1 = 0.
 * @tparam LBFGSMemory L-BFGS memory size for PANOC (default: 5).
 */
template <typename EKF_Type_In, typename Cost_Matrices_Type_In,
          bool HasOutputConstraints = false, std::size_t LBFGSMemory = 5>
class NonlinearMPC_OptimizationEngine {
protected:
  /* Type */
  using _T = typename EKF_Type_In::Value_Type;

public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE = Cost_Matrices_Type_In::STATE_SIZE;
  static constexpr std::size_t INPUT_SIZE = Cost_Matrices_Type_In::INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = Cost_Matrices_Type_In::OUTPUT_SIZE;

  enum : std::size_t { NP = Cost_Matrices_Type_In::NP };

  static constexpr std::size_t NUMBER_OF_DELAY = EKF_Type_In::NUMBER_OF_DELAY;

  static constexpr std::size_t PROBLEM_SIZE = INPUT_SIZE * NP;

  /** @brief N1 = OUTPUT_SIZE * (NP + 1) if HasOutputConstraints, else 0. */
  static constexpr std::size_t N1 =
      HasOutputConstraints ? OUTPUT_SIZE * (NP + 1) : 0;

  /** @brief N2 = 0 (no PM-type equality constraints). */
  static constexpr std::size_t N2 = 0;

public:
  /* Type */
  using Value_Type = _T;
  using EKF_Type = EKF_Type_In;
  using Cost_Matrices_Type = Cost_Matrices_Type_In;

  using X_Type = typename Cost_Matrices_Type::X_Type;
  using U_Type = typename Cost_Matrices_Type::U_Type;
  using Y_Type = typename Cost_Matrices_Type::Y_Type;

  using U_Horizon_Type = typename Cost_Matrices_Type::U_Horizon_Type;
  using Y_Horizon_Type = typename Cost_Matrices_Type::Y_Horizon_Type;
  using Y_Store_Type =
      PythonControl::DelayedVectorObject<Y_Type, NUMBER_OF_DELAY>;

  using CostMatricesReferenceTrajectory_Type =
      PythonNumpy::DenseMatrix_Type<_T, OUTPUT_SIZE, (NP + 1)>;

protected:
  /* Type */
  using _Parameter_Type = typename EKF_Type::Parameter_Type;

  using _Gradient_Type = U_Horizon_Type;

  /* ALM/PM types */
  using _ALM_PM_Optimizer_Type =
      PythonOptimization::ALM_PM_Optimizer<Cost_Matrices_Type, N1, N2,
                                           LBFGSMemory>;
  using _ALM_Problem_Type =
      PythonOptimization::ALM_Problem<Cost_Matrices_Type, N1, N2>;
  using _ALM_Factory_Type =
      PythonOptimization::ALM_Factory<Cost_Matrices_Type, N1, N2>;
  using _ALM_SolverStatus_Type = PythonOptimization::ALM_SolverStatus<_T, N1>;

  using _BoxProjection_U_Type =
      PythonOptimization::BoxProjectionOperator<_T, PROBLEM_SIZE>;
  using _BoxProjection_Y_Type =
      PythonOptimization::BoxProjectionOperator<_T, N1>;
  using _BallProjection_Y_Type =
      PythonOptimization::BallProjectionOperator<_T, N1>;

  /* Flat vector types for constraints/projections */
  using _U_Min_Flat_Type = PythonNumpy::DenseMatrix_Type<_T, PROBLEM_SIZE, 1>;
  using _U_Max_Flat_Type = PythonNumpy::DenseMatrix_Type<_T, PROBLEM_SIZE, 1>;
  using _Y_Horizon_Flat_Type =
      PythonNumpy::DenseMatrix_Type<_T, (N1 > 0 ? N1 : 1), 1>;

  /* Function object types */
  using _CostFunction_Type = std::function<_T(const U_Horizon_Type &)>;
  using _GradientFunction_Type =
      std::function<_Gradient_Type(const U_Horizon_Type &)>;

  using _MappingF1_Type = typename _ALM_Factory_Type::Mapping_F1_Type;
  using _JacobianF1Trans_Type =
      typename _ALM_Factory_Type::Jacobian_F1_Trans_Type;
  using _SetCProject_Type = typename _ALM_Factory_Type::Set_C_Project_Type;
  using _SetYProject_Type = typename _ALM_Problem_Type::Set_Y_Project_Type;

public:
  /* Constructor */
  NonlinearMPC_OptimizationEngine()
      : U_horizon(), _kalman_filter(), _cost_matrices(), _delta_time(0),
        _Y_store(),
        _solver_outer_max_iteration(NonlinearMPC_OptimizationEngine_Constants::
                                        ALM_OUTER_MAX_ITERATIONS_DEFAULT),
        _solver_inner_max_iteration(NonlinearMPC_OptimizationEngine_Constants::
                                        ALM_INNER_MAX_ITERATION_DEFAULT),
        _last_iteration_count(0), _cost_function(nullptr),
        _gradient_function(nullptr), _alm_problem(), _solver(),
        _solver_status() {}

  NonlinearMPC_OptimizationEngine(EKF_Type &kalman_filter,
                                  Cost_Matrices_Type &cost_matrices,
                                  _T delta_time, X_Type X_initial)
      : U_horizon(), _kalman_filter(kalman_filter),
        _cost_matrices(cost_matrices), _delta_time(delta_time), _Y_store(),
        _solver_outer_max_iteration(NonlinearMPC_OptimizationEngine_Constants::
                                        ALM_OUTER_MAX_ITERATIONS_DEFAULT),
        _solver_inner_max_iteration(NonlinearMPC_OptimizationEngine_Constants::
                                        ALM_INNER_MAX_ITERATION_DEFAULT),
        _last_iteration_count(0), _cost_function(), _gradient_function(),
        _alm_problem(), _solver(), _solver_status() {

    this->_kalman_filter.set_x_hat(X_initial);

    this->_initialize_solver(X_initial);
  }

  /* Copy Constructor */
  NonlinearMPC_OptimizationEngine(
      const NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type,
                                            HasOutputConstraints, LBFGSMemory>
          &input)
      : U_horizon(input.U_horizon), _kalman_filter(input._kalman_filter),
        _cost_matrices(input._cost_matrices), _delta_time(input._delta_time),
        _Y_store(input._Y_store),
        _solver_outer_max_iteration(input._solver_outer_max_iteration),
        _solver_inner_max_iteration(input._solver_inner_max_iteration),
        _last_iteration_count(input._last_iteration_count),
        _cost_function(input._cost_function),
        _gradient_function(input._gradient_function),
        _alm_problem(input._alm_problem), _solver(input._solver),
        _solver_status(input._solver_status) {
    this->_bind_cost_functions();
    this->_setup_alm_problem();
  }

  NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type,
                                  HasOutputConstraints, LBFGSMemory> &
  operator=(const NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type,
                                                  HasOutputConstraints,
                                                  LBFGSMemory> &input) {
    if (this != &input) {
      this->U_horizon = input.U_horizon;
      this->_kalman_filter = input._kalman_filter;
      this->_cost_matrices = input._cost_matrices;
      this->_delta_time = input._delta_time;
      this->_Y_store = input._Y_store;
      this->_solver_outer_max_iteration = input._solver_outer_max_iteration;
      this->_solver_inner_max_iteration = input._solver_inner_max_iteration;
      this->_last_iteration_count = input._last_iteration_count;

      this->_cost_function = {};
      this->_gradient_function = {};

      this->_alm_problem = input._alm_problem;
      this->_solver = input._solver;
      this->_solver_status = input._solver_status;

      this->_bind_cost_functions();
      this->_setup_alm_problem();
    }
    return *this;
  }

  /* Move Constructor */
  NonlinearMPC_OptimizationEngine(
      NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type,
                                      HasOutputConstraints, LBFGSMemory>
          &&input) noexcept
      : U_horizon(std::move(input.U_horizon)),
        _kalman_filter(std::move(input._kalman_filter)),
        _cost_matrices(std::move(input._cost_matrices)),
        _delta_time(std::move(input._delta_time)),
        _Y_store(std::move(input._Y_store)),
        _solver_outer_max_iteration(input._solver_outer_max_iteration),
        _solver_inner_max_iteration(input._solver_inner_max_iteration),
        _last_iteration_count(input._last_iteration_count),
        _cost_function(std::move(input._cost_function)),
        _gradient_function(std::move(input._gradient_function)),
        _alm_problem(std::move(input._alm_problem)),
        _solver(std::move(input._solver)),
        _solver_status(std::move(input._solver_status)) {

    this->_bind_cost_functions();
    this->_setup_alm_problem();
  }

  NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type,
                                  HasOutputConstraints, LBFGSMemory> &
  operator=(NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type,
                                            HasOutputConstraints, LBFGSMemory>
                &&input) noexcept {
    if (this != &input) {
      this->U_horizon = std::move(input.U_horizon);
      this->_kalman_filter = std::move(input._kalman_filter);
      this->_cost_matrices = std::move(input._cost_matrices);
      this->_delta_time = std::move(input._delta_time);
      this->_Y_store = std::move(input._Y_store);
      this->_solver_outer_max_iteration = input._solver_outer_max_iteration;
      this->_solver_inner_max_iteration = input._solver_inner_max_iteration;
      this->_last_iteration_count = input._last_iteration_count;

      this->_cost_function = {};
      this->_gradient_function = {};

      this->_alm_problem = std::move(input._alm_problem);
      this->_solver = std::move(input._solver);
      this->_solver_status = std::move(input._solver_status);

      this->_bind_cost_functions();
      this->_setup_alm_problem();
    }
    return *this;
  }

public:
  /* Setter */

  /**
   * @brief Sets the maximum number of iterations for the solver.
   *
   * @param outer_max_iteration The maximum number of outer iterations the
   * solver is allowed to perform.
   * @param inner_max_iteration The maximum number of inner iterations the
   * solver is allowed to perform.
   */
  inline void
  set_solver_max_iteration(std::size_t outer_max_iteration =
                               NonlinearMPC_OptimizationEngine_Constants::
                                   ALM_OUTER_MAX_ITERATIONS_DEFAULT,
                           std::size_t inner_max_iteration =
                               NonlinearMPC_OptimizationEngine_Constants::
                                   ALM_INNER_MAX_ITERATION_DEFAULT) {
    this->_solver_outer_max_iteration = outer_max_iteration;
    this->_solver_inner_max_iteration = inner_max_iteration;

    this->_solver.set_solver_max_iteration(this->_solver_outer_max_iteration,
                                           this->_solver_inner_max_iteration);
  }

  /**
   * @brief Sets the reference trajectory for the MPC.
   *
   * @tparam Reference_Type_In Type of the reference input.
   * @param reference The reference input (trajectory matrix or single vector).
   */
  template <typename Reference_Type_In>
  inline void set_reference_trajectory(const Reference_Type_In &reference) {

    static_assert(Reference_Type_In::COLS == OUTPUT_SIZE,
                  "COLS of Reference_Type_In must be equal to OUTPUT_SIZE");
    static_assert((Reference_Type_In::ROWS == NP) ||
                      (Reference_Type_In::ROWS == 1),
                  "ROWS of Reference_Type_In must be equal to NP, or 1");

    CostMatricesReferenceTrajectory_Type reference_trajectory;

    NonlinearMPC_OptEng_ReferenceTrajectoryOperation::substitute_reference<
        Reference_Type_In::ROWS, NP>(reference_trajectory, reference);

    this->_cost_matrices.reference_trajectory = reference_trajectory;
  }

  /* Getter */

  /**
   * @brief Retrieves the number of iterations performed in the last solver
   * step.
   *
   * @return std::size_t The number of inner iterations.
   */
  inline auto get_solver_step_iterated_number(void) const -> std::size_t {
    return this->_last_iteration_count;
  }

  /**
   * @brief Retrieves the current estimated state from the Kalman filter.
   *
   * @return X_Type The current estimated state.
   */
  inline auto get_X(void) const -> X_Type {
    return this->_kalman_filter.get_x_hat();
  }

  /**
   * @brief Retrieves the solver status from the last solve call.
   *
   * @return const _ALM_SolverStatus_Type& Reference to the solver status.
   */
  inline auto get_solver_status(void) const -> const _ALM_SolverStatus_Type & {
    return this->_solver_status;
  }

  /* Function */

  /**
   * @brief Updates the parameters of the Kalman filter and cost matrices.
   *
   * @param parameters The new parameters to be set.
   */
  inline void update_parameters(const _Parameter_Type &parameters) {
    this->_kalman_filter.parameters = parameters;
    this->_cost_matrices.state_space_parameters = parameters;
  }

  /**
   * @brief Updates the control input based on the current reference and
   * measurement.
   *
   * This function performs a complete update cycle for the MPC:
   * 1. Calculates the latest control input based on the current control
   * horizon.
   * 2. Updates the internal Kalman filter state estimate.
   * 3. Compensates the state estimate for any delay.
   * 4. Sets the reference trajectory for the MPC optimization.
   * 5. Solves the MPC optimization problem using ALM/PM.
   * 6. Calculates and returns the latest control input after optimization.
   *
   * @tparam Reference_Type_In Type of the reference input.
   * @param reference The current reference input for the MPC.
   * @param Y The current measurement used for state estimation.
   * @return U_Type The updated control input to be applied.
   */
  template <typename Reference_Type_In>
  inline auto update_manipulation(Reference_Type_In &reference, const Y_Type &Y)
      -> U_Type {

    auto U_latest = this->_calculate_this_U(this->U_horizon);

    this->_kalman_filter.predict_and_update(U_latest, Y);

    auto X = this->_kalman_filter.get_x_hat();

    X_Type X_compensated;
    Y_Type Y_compensated;
    std::tie(X_compensated, Y_compensated) = this->_compensate_X_Y_delay(X, Y);

    this->set_reference_trajectory(reference);

    this->_cost_matrices.X_initial = X_compensated;

    this->U_horizon = this->_solver.solve(this->U_horizon);

    this->_solver_status = this->_solver.get_solver_status();
    this->_last_iteration_count = this->_solver_status.num_inner_iterations;

    U_latest = this->_calculate_this_U(this->U_horizon);

    return U_latest;
  }

protected:
  /* Function */

  /**
   * @brief Converts a matrix to a flat vector for optimization.
   *
   * @tparam Matrix_Type Type of the input matrix.
   * @param matrix The input matrix to be flattened.
   * @return PythonNumpy::DenseMatrix_Type<_T, Matrix_Type::COLS *
   * Matrix_Type::ROWS, 1> The flattened matrix as a column vector.
   */
  template <typename Matrix_Type>
  static inline auto _to_flat(Matrix_Type matrix)
      -> PythonNumpy::DenseMatrix_Type<
          _T, Matrix_Type::COLS * Matrix_Type::ROWS, 1> {

    return PythonNumpy::reshape<Matrix_Type::COLS * Matrix_Type::ROWS, 1>(
        matrix);
  }

  /**
   * @brief Converts a flat vector back to a matrix of specified dimensions.
   *
   * @tparam M Number of rows in the output matrix.
   * @tparam N Number of columns in the output matrix.
   * @param vector The input flat vector to be reshaped.
   * @return PythonNumpy::DenseMatrix_Type<_T, M, N> The reshaped matrix with
   * dimensions M x N.
   */
  template <std::size_t M, std::size_t N, typename Vector_Type>
  static inline auto _from_flat(Vector_Type vector)
      -> PythonNumpy::DenseMatrix_Type<_T, M, N> {

    return PythonNumpy::reshape<M, N>(vector);
  }

  /**
   * @brief Calculates the current control input from the control input horizon.
   *
   * @param U_horizon_in The control input horizon.
   * @return U_Type The current control input (first element of horizon).
   */
  inline auto _calculate_this_U(const U_Horizon_Type &U_horizon_in) -> U_Type {
    auto U = PythonNumpy::get_row<0>(U_horizon_in);
    return U;
  }

  /**
   * @brief Binds the cost functions for the optimizer.
   */
  inline void _bind_cost_functions() {
    this->_cost_function = [this](const U_Horizon_Type &U) ->
        typename X_Type::Value_Type {
          return this->_cost_matrices.compute_cost(U);
        };

    this->_gradient_function =
        [this](const U_Horizon_Type &U) -> _Gradient_Type {
      return this->_cost_matrices.compute_gradient(U);
    };
  }

  /**
   * @brief Sets up the ALM problem definition.
   * Specialization for HasOutputConstraints == false: no output constraints.
   */
  template <bool _HasOutputConstraints = HasOutputConstraints,
            typename std::enable_if<!_HasOutputConstraints, int>::type = 0>
  inline void _setup_alm_problem() {

    _ALM_Factory_Type alm_factory;
    alm_factory.set_cost_function(this->_cost_function);
    alm_factory.set_gradient_function(this->_gradient_function);

    this->_alm_problem.set_parametric_cost(
        [alm_factory](const U_Horizon_Type &u,
                      const typename _ALM_Factory_Type::Xi_Type &xi) -> _T {
          return alm_factory.psi(u, xi);
        });

    this->_alm_problem.set_parametric_gradient(
        [alm_factory](const U_Horizon_Type &u,
                      const typename _ALM_Factory_Type::Xi_Type &xi)
            -> _Gradient_Type { return alm_factory.d_psi(u, xi); });

    this->_alm_problem.set_u_min_matrix(
        this->_cost_matrices.get_U_min_matrix());
    this->_alm_problem.set_u_max_matrix(
        this->_cost_matrices.get_U_max_matrix());

    this->_solver.set_problem(this->_alm_problem);
  }

  /**
   * @brief Sets up the ALM problem definition.
   * Specialization for HasOutputConstraints == true: with output constraints.
   */
  template <bool _HasOutputConstraints = HasOutputConstraints,
            typename std::enable_if<_HasOutputConstraints, int>::type = 0>
  inline void _setup_alm_problem() {

    /* Create output constraint box projection */
    _Y_Horizon_Flat_Type Y_min_flat = NonlinearMPC_OptimizationEngine::_to_flat(
        this->_cost_matrices.get_Y_min_matrix());
    _Y_Horizon_Flat_Type Y_max_flat = NonlinearMPC_OptimizationEngine::_to_flat(
        this->_cost_matrices.get_Y_max_matrix());

    _BoxProjection_Y_Type set_c_project(Y_min_flat, Y_max_flat);

    /* Create Lagrange multiplier projection (ball of large radius) */
    _BallProjection_Y_Type set_y_project(
        static_cast<_T>(NonlinearMPC_OptimizationEngine_Constants::
                            BALL_PROJECTION_RADIUS_DEFAULT));

    /* Create mapping F1 and Jacobian transpose */
    _MappingF1_Type mapping_f1 =
        [this](const U_Horizon_Type &u) -> _Y_Horizon_Flat_Type {
      Y_Horizon_Type Y_horizon = this->_cost_matrices.compute_output_mapping(u);
      return NonlinearMPC_OptimizationEngine::_to_flat(Y_horizon);
    };

    _JacobianF1Trans_Type jacobian_f1_trans =
        [this](const U_Horizon_Type &u,
               const _Y_Horizon_Flat_Type &d) -> _Gradient_Type {
      Y_Horizon_Type D_reshaped =
          NonlinearMPC_OptimizationEngine::_from_flat<Y_Horizon_Type::COLS,
                                                      Y_Horizon_Type::ROWS>(d);
      return this->_cost_matrices.compute_output_jacobian_trans(u, D_reshaped);
    };

    _SetCProject_Type set_c_project_func =
        [set_c_project](_Y_Horizon_Flat_Type &x) mutable {
          set_c_project.project(x);
        };

    _SetYProject_Type set_y_project_func =
        [set_y_project](_Y_Horizon_Flat_Type &x) mutable {
          set_y_project.project(x);
        };

    _ALM_Factory_Type alm_factory;
    alm_factory.set_cost_function(this->_cost_function);
    alm_factory.set_gradient_function(this->_gradient_function);
    alm_factory.set_mapping_f1(mapping_f1);
    alm_factory.set_jacobian_f1_trans(jacobian_f1_trans);
    alm_factory.set_c_projection(set_c_project_func);

    this->_alm_problem.set_parametric_cost(
        [alm_factory](const U_Horizon_Type &u,
                      const typename _ALM_Factory_Type::Xi_Type &xi) -> _T {
          return alm_factory.psi(u, xi);
        });

    this->_alm_problem.set_parametric_gradient(
        [alm_factory](const U_Horizon_Type &u,
                      const typename _ALM_Factory_Type::Xi_Type &xi)
            -> _Gradient_Type { return alm_factory.d_psi(u, xi); });

    this->_alm_problem.set_u_min_matrix(
        this->_cost_matrices.get_U_min_matrix());
    this->_alm_problem.set_u_max_matrix(
        this->_cost_matrices.get_U_max_matrix());

    this->_alm_problem.set_mapping_f1(mapping_f1);
    this->_alm_problem.set_c_projection(set_c_project_func);
    this->_alm_problem.set_y_projection(set_y_project_func);

    this->_solver.set_problem(this->_alm_problem);
  }

  /**
   * @brief Initializes the solver with the initial state.
   *
   * @param X_initial The initial state for solver initialization.
   */
  inline void _initialize_solver(const X_Type &X_initial) {
    this->_bind_cost_functions();

    this->_cost_matrices.X_initial = X_initial;

    this->_setup_alm_problem();

    this->_solver.set_solver_max_iteration(this->_solver_outer_max_iteration,
                                           this->_solver_inner_max_iteration);

    this->_solver.set_epsilon_tolerance(
        static_cast<_T>(NonlinearMPC_OptimizationEngine_Constants::
                            ALM_EPSILON_TOLERANCE_DEFAULT));
    this->_solver.set_delta_tolerance(
        static_cast<_T>(NonlinearMPC_OptimizationEngine_Constants::
                            ALM_DELTA_TOLERANCE_DEFAULT));
    this->_solver.set_initial_inner_tolerance(
        static_cast<_T>(NonlinearMPC_OptimizationEngine_Constants::
                            ALM_INITIAL_INNER_TOLERANCE_DEFAULT));
    this->_solver.set_initial_penalty(
        static_cast<_T>(NonlinearMPC_OptimizationEngine_Constants::
                            ALM_INITIAL_PENALTY_DEFAULT));
  }

  /**
   * @brief Compensates for delays in the state and measurement.
   *
   * @param X_in The input state to be compensated.
   * @param Y_in The input measurement to be compensated.
   * @return std::tuple<X_Type, Y_Type> Compensated state and measurement.
   */
  inline auto _compensate_X_Y_delay(const X_Type &X_in, const Y_Type &Y_in)
      -> std::tuple<X_Type, Y_Type> {

    return AdaptiveMPC_Operation::compensate_X_Y_delay<NUMBER_OF_DELAY>(
        X_in, Y_in, this->_Y_store, this->_kalman_filter);
  }

public:
  /* Variable */
  U_Horizon_Type U_horizon;

protected:
  /* Variable */
  EKF_Type _kalman_filter;
  Cost_Matrices_Type _cost_matrices;

  _T _delta_time;

  Y_Store_Type _Y_store;

  std::size_t _solver_outer_max_iteration;
  std::size_t _solver_inner_max_iteration;
  std::size_t _last_iteration_count;

  _CostFunction_Type _cost_function;
  _GradientFunction_Type _gradient_function;

  _ALM_Problem_Type _alm_problem;
  _ALM_PM_Optimizer_Type _solver;
  _ALM_SolverStatus_Type _solver_status;
};

/* ==========================================================================
 * Factory Functions
 * ========================================================================== */

/**
 * @brief Factory function to create a NonlinearMPC_OptimizationEngine object.
 *
 * @tparam EKF_Type Type of the Extended Kalman Filter (EKF).
 * @tparam Cost_Matrices_Type Type of the cost matrices.
 * @tparam HasOutputConstraints Whether output constraints are present.
 * @tparam LBFGSMemory L-BFGS memory size for PANOC.
 * @tparam T Data type for the EKF and cost matrices.
 * @param kalman_filter Reference to an EKF object.
 * @param cost_matrices Reference to a cost matrices object.
 * @param delta_time Time step for the MPC updates.
 * @param X_initial Initial state.
 * @return NonlinearMPC_OptimizationEngine instance.
 */
template <typename EKF_Type, typename Cost_Matrices_Type,
          bool HasOutputConstraints = false, std::size_t LBFGSMemory = 5,
          typename T>
inline auto make_NonlinearMPC_OptimizationEngine(
    EKF_Type &kalman_filter, Cost_Matrices_Type &cost_matrices, T delta_time,
    typename Cost_Matrices_Type::X_Type X_initial)
    -> NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type,
                                       HasOutputConstraints, LBFGSMemory> {

  static_assert(std::is_same<typename EKF_Type::Value_Type, T>::value,
                "Data type of EKF must be same type as T.");
  static_assert(std::is_same<typename Cost_Matrices_Type::Value_Type, T>::value,
                "Data type of CostMatrices must be same type as T.");

  using NonlinearMPC_OptimizationEngine_Type =
      NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type,
                                      HasOutputConstraints, LBFGSMemory>;

  NonlinearMPC_OptimizationEngine_Type nonlinear_mpc(
      kalman_filter, cost_matrices, static_cast<T>(delta_time), X_initial);

  return nonlinear_mpc;
}

/**
 * @brief Factory function to create a NonlinearMPC_OptimizationEngine object
 * without output constraints (convenience wrapper).
 */
template <typename EKF_Type, typename Cost_Matrices_Type,
          std::size_t LBFGSMemory = 5, typename T>
inline auto make_NonlinearMPC_OptimizationEngine_NoConstraints(
    EKF_Type &kalman_filter, Cost_Matrices_Type &cost_matrices, T delta_time,
    typename Cost_Matrices_Type::X_Type X_initial)
    -> NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type, false,
                                       LBFGSMemory> {

  return make_NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type,
                                              false, LBFGSMemory>(
      kalman_filter, cost_matrices, delta_time, X_initial);
}

/**
 * @brief Factory function to create a NonlinearMPC_OptimizationEngine object
 * with output constraints (convenience wrapper).
 */
template <typename EKF_Type, typename Cost_Matrices_Type,
          std::size_t LBFGSMemory = 5, typename T>
inline auto make_NonlinearMPC_OptimizationEngine_WithConstraints(
    EKF_Type &kalman_filter, Cost_Matrices_Type &cost_matrices, T delta_time,
    typename Cost_Matrices_Type::X_Type X_initial)
    -> NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type, true,
                                       LBFGSMemory> {

  return make_NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type,
                                              true, LBFGSMemory>(
      kalman_filter, cost_matrices, delta_time, X_initial);
}

/* ==========================================================================
 * Type Aliases
 * ========================================================================== */

/**
 * @brief Type alias for NonlinearMPC_OptimizationEngine.
 *
 * @tparam EKF_Type Type of the Extended Kalman Filter.
 * @tparam Cost_Matrices_Type Type of the cost matrices.
 * @tparam HasOutputConstraints Whether output constraints are present.
 * @tparam LBFGSMemory L-BFGS memory size.
 */
template <typename EKF_Type, typename Cost_Matrices_Type,
          bool HasOutputConstraints = false, std::size_t LBFGSMemory = 5>
using NonlinearMPC_OptimizationEngine_Type =
    NonlinearMPC_OptimizationEngine<EKF_Type, Cost_Matrices_Type,
                                    HasOutputConstraints, LBFGSMemory>;

} // namespace PythonMPC

#endif // __PYTHON_NONLINEAR_MPC_OPTIMIZATION_ENGINE_HPP__
