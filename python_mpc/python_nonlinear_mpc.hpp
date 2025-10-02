/**
 * @file python_nonlinear_mpc.hpp
 *
 * @brief Nonlinear Model Predictive Control (Nonlinear MPC) library
 *
 * This header file defines the NonlinearMPC_TwiceDifferentiable class template,
 * which implements a Nonlinear Model Predictive Control (MPC) algorithm using
 * a twice-differentiable cost function. The class integrates an Extended
 * Kalman Filter (EKF) for state estimation and utilizes a Sequential
 * Quadratic Programming (SQP) solver for optimization.
 *
 * The NonlinearMPC_TwiceDifferentiable class supports setting reference
 * trajectories and updating control inputs in a flexible manner. The
 * implementation is designed to work with Python-based state-space models,
 * making it suitable for applications that require real-time control of
 * nonlinear systems.
 */
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

/**
 * @brief Recursively copies elements from a reference object to a reference
 * trajectory at a specific row (I) and column (J_idx).
 *
 * This struct template defines a static inline function `compute` that copies
 * the element at position (I, J_idx) from the `reference` object to the
 * `reference_trajectory` object using their respective `get` and `set` methods.
 * The recursion proceeds by decrementing the column index `J_idx` until the
 * base case is reached (not shown here).
 *
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory object,
 * which must provide a `set<I, J_idx>()` method.
 * @tparam Reference_Type Type of the reference object, which must provide a
 * `get<I, J_idx>()` method.
 * @tparam I Row index for the element to be copied.
 * @tparam J_idx Column index for the element to be copied; decremented
 * recursively.
 *
 * @param reference_trajectory The object to which elements are copied.
 * @param reference The object from which elements are copied.
 */
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

/**
 * @brief Base case for the recursive copying of elements from a reference
 * object to a reference trajectory at a specific row (I) and column (0).
 *
 * This struct template specializes the `Column` struct for the case when
 * `J_idx` is 0. It defines a static inline function `compute` that copies
 * the element at position (I, 0) from the `reference` object to the
 * `reference_trajectory` object using their respective `get` and `set` methods.
 *
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory object,
 * which must provide a `set<I, 0>()` method.
 * @tparam Reference_Type Type of the reference object, which must provide a
 * `get<I, 0>()` method.
 * @tparam I Row index for the element to be copied.
 *
 * @param reference_trajectory The object to which elements are copied.
 * @param reference The object from which elements are copied.
 */
template <typename ReferenceTrajectory_Type, typename Reference_Type,
          std::size_t I>
struct Column<ReferenceTrajectory_Type, Reference_Type, I, 0> {
  static inline void compute(ReferenceTrajectory_Type &reference_trajectory,
                             const Reference_Type &reference) {
    reference_trajectory.template set<I, 0>(reference.template get<I, 0>());
  }
};

/**
 * @brief Recursively processes rows of the reference trajectory to copy
 * elements from the reference object.
 *
 * This struct template defines a static inline function `compute` that
 * processes each row of the `reference_trajectory` object by invoking the
 * `Column` struct to copy elements from the `reference` object. The recursion
 * proceeds by decrementing the row index `I_idx` until the base case is
 * reached (not shown here).
 *
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory object,
 * which must provide a `set<I, J_idx>()` method.
 * @tparam Reference_Type Type of the reference object, which must provide a
 * `get<I, J_idx>()` method.
 * @tparam M Total number of rows in the reference trajectory.
 * @tparam N Total number of columns in the reference trajectory.
 * @tparam I_idx Current row index being processed; decremented recursively.
 *
 * @param reference_trajectory The object to which elements are copied.
 * @param reference The object from which elements are copied.
 */
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

/**
 * @brief Base case for the recursive processing of rows in the reference
 * trajectory.
 *
 * This struct template specializes the `Row` struct for the case when
 * `I_idx` is 0. It defines a static inline function `compute` that processes
 * the first row of the `reference_trajectory` object by invoking the `Column`
 * struct to copy elements from the `reference` object.
 *
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory object,
 * which must provide a `set<0, J_idx>()` method.
 * @tparam Reference_Type Type of the reference object, which must provide a
 * `get<0, J_idx>()` method.
 * @tparam M Total number of rows in the reference trajectory.
 * @tparam N Total number of columns in the reference trajectory.
 *
 * @param reference_trajectory The object to which elements are copied.
 * @param reference The object from which elements are copied.
 */
template <typename ReferenceTrajectory_Type, typename Reference_Type,
          std::size_t M, std::size_t N>
struct Row<ReferenceTrajectory_Type, Reference_Type, M, N, 0> {
  static inline void compute(ReferenceTrajectory_Type &reference_trajectory,
                             const Reference_Type &reference) {
    Column<ReferenceTrajectory_Type, Reference_Type, 0, (N - 1)>::compute(
        reference_trajectory, reference);
  }
};

/**
 * @brief Copies elements from a reference object to a reference trajectory.
 *
 * This function initiates the recursive copying of elements from the
 * `reference` object to the `reference_trajectory` object by invoking the
 * `Row` struct. It assumes that both objects provide appropriate `get` and
 * `set` methods for element access.
 *
 * @tparam COLS Number of columns in the reference trajectory.
 * @tparam Np Number of prediction steps (rows - 1) in the reference
 * trajectory.
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory object,
 * which must provide a `set<I, J_idx>()` method.
 * @tparam Reference_Type Type of the reference object, which must provide a
 * `get<I, J_idx>()` method.
 *
 * @param reference_trajectory The object to which elements are copied.
 * @param reference The object from which elements are copied.
 */
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
 *
 * This function substitutes the reference trajectory based on the number of
 * rows in the reference input. If the number of rows is greater than 1, it
 * copies the reference values directly to the corresponding rows in the
 * reference trajectory. If the number of rows is 1, it replicates the single
 * reference value across all rows of the reference trajectory.
 *
 * @tparam ROWS Number of rows in the reference input.
 * @tparam Np Number of prediction steps (rows - 1) in the reference
 * trajectory.
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory object,
 * which must provide a `set<I, J_idx>()` method.
 * @tparam Reference_Type Type of the reference object, which must provide a
 * `get<I, J_idx>()` method.
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

/**
 * @brief Recursively copies elements from a reference vector to a reference
 * trajectory.
 *
 * This struct template defines a static inline function `compute` that
 * copies elements from the `reference` vector to the `reference_trajectory`
 * matrix. The recursion proceeds by decrementing the row index `I` and column
 * index `J_idx` until the base case is reached (not shown here).
 *
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory object,
 * which must provide a `set<I, J_idx>()` method.
 * @tparam Reference_Type Type of the reference vector, which must provide a
 * `get<I, 0>()` method.
 * @tparam M Total number of rows in the reference trajectory.
 * @tparam N Total number of columns in the reference trajectory.
 * @tparam I Current row index being processed; decremented recursively.
 * @tparam J_idx Current column index being processed; decremented recursively.
 *
 * @param reference_trajectory The object to which elements are copied.
 * @param reference The object from which elements are copied.
 */
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

/**
 * @brief Base case for the recursive copying of elements from a reference
 * vector to a reference trajectory.
 *
 * This struct template specializes the `Column` struct for the case when
 * `J_idx` is 0. It defines a static inline function `compute` that copies
 * the element at position (I, 0) from the `reference` vector to the
 * `reference_trajectory` object using their respective `get` and `set` methods.
 *
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory object,
 * which must provide a `set<I, 0>()` method.
 * @tparam Reference_Type Type of the reference vector, which must provide a
 * `get<I, 0>()` method.
 * @tparam M Total number of rows in the reference trajectory.
 * @tparam N Total number of columns in the reference trajectory.
 * @tparam I Row index for the element to be copied.
 *
 * @param reference_trajectory The object to which elements are copied.
 * @param reference The object from which elements are copied.
 */
template <typename ReferenceTrajectory_Type, typename Reference_Type,
          std::size_t M, std::size_t N, std::size_t I>
struct Column<ReferenceTrajectory_Type, Reference_Type, M, N, I, 0> {
  static inline void compute(ReferenceTrajectory_Type &reference_trajectory,
                             const Reference_Type &reference) {
    reference_trajectory.template set<I, 0>(reference.template get<I, 0>());
  }
};

/**
 * @brief Recursively processes rows of the reference trajectory to copy
 * elements from the reference vector.
 *
 * This struct template defines a static inline function `compute` that
 * processes each row of the `reference_trajectory` object by invoking the
 * `Column` struct to copy elements from the `reference` vector. The recursion
 * proceeds by decrementing the row index `I_idx` until the base case is
 * reached (not shown here).
 *
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory object,
 * which must provide a `set<I, J_idx>()` method.
 * @tparam Reference_Type Type of the reference vector, which must provide a
 * `get<I, 0>()` method.
 * @tparam M Total number of rows in the reference trajectory.
 * @tparam N Total number of columns in the reference trajectory.
 * @tparam I_idx Current row index being processed; decremented recursively.
 *
 * @param reference_trajectory The object to which elements are copied.
 * @param reference The object from which elements are copied.
 */
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

/**
 * @brief Base case for the recursive processing of rows in the reference
 * trajectory.
 *
 * This struct template specializes the `Row` struct for the case when
 * `I_idx` is 0. It defines a static inline function `compute` that processes
 * the first row of the `reference_trajectory` object by invoking the `Column`
 * struct to copy elements from the `reference` vector.
 *
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory object,
 * which must provide a `set<0, J_idx>()` method.
 * @tparam Reference_Type Type of the reference vector, which must provide a
 * `get<0, 0>()` method.
 * @tparam M Total number of rows in the reference trajectory.
 * @tparam N Total number of columns in the reference trajectory.
 *
 * @param reference_trajectory The object to which elements are copied.
 * @param reference The object from which elements are copied.
 */
template <typename ReferenceTrajectory_Type, typename Reference_Type,
          std::size_t M, std::size_t N>
struct Row<ReferenceTrajectory_Type, Reference_Type, M, N, 0> {
  static inline void compute(ReferenceTrajectory_Type &reference_trajectory,
                             const Reference_Type &reference) {
    Column<ReferenceTrajectory_Type, Reference_Type, M, N, 0, (N - 1)>::compute(
        reference_trajectory, reference);
  }
};

/**
 * @brief Copies elements from a reference vector to a reference trajectory.
 *
 * This function initiates the recursive copying of elements from the
 * `reference` vector to the `reference_trajectory` matrix by invoking the
 * `Row` struct. It assumes that both objects provide appropriate `get` and
 * `set` methods for element access.
 *
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory object,
 * which must provide a `set<I, J_idx>()` method.
 * @tparam Reference_Type Type of the reference vector, which must provide a
 * `get<I, 0>()` method.
 *
 * @param reference_trajectory The object to which elements are copied.
 * @param reference The object from which elements are copied.
 */
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
 *
 * This function substitutes the reference trajectory by copying the values
 * from a reference vector to each column of the reference trajectory matrix.
 * It assumes that the reference vector has a single column and the same number
 * of rows as the reference trajectory.
 *
 * @tparam ROWS Number of rows in the reference input.
 * @tparam Np Number of prediction steps (rows - 1) in the reference
 * trajectory.
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory object,
 * which must provide a `set<I, J_idx>()` method.
 * @tparam Reference_Type Type of the reference vector, which must provide a
 * `get<I, 0>()` method.
 */
template <std::size_t ROWS, std::size_t Np, typename ReferenceTrajectory_Type,
          typename Reference_Type>
inline typename std::enable_if<(ROWS == 1), void>::type
substitute_reference(ReferenceTrajectory_Type &reference_trajectory,
                     const Reference_Type &reference) {
  static_assert(ROWS == 1, "ROWS must be equal to 1");

  SubstituteReferenceVector::substitute(reference_trajectory, reference);
}

} // namespace NonlinearMPC_ReferenceTrajectoryOperation

/**
 * @brief Nonlinear Model Predictive Control (Nonlinear MPC) class template
 * using a twice-differentiable cost function.
 *
 * This class template implements a Nonlinear Model Predictive Control (MPC)
 * algorithm that utilizes a twice-differentiable cost function. It integrates
 * an Extended Kalman Filter (EKF) for state estimation and employs a
 * Sequential Quadratic Programming (SQP) solver for optimization.
 *
 * The NonlinearMPC_TwiceDifferentiable class supports setting reference
 * trajectories and updating control inputs in a flexible manner. It is designed
 * to work with Python-based state-space models, making it suitable for
 * applications that require real-time control of nonlinear systems.
 *
 * @tparam EKF_Type_In Type of the Extended Kalman Filter (EKF) used for state
 * estimation. Must provide methods for state prediction and update.
 * @tparam Cost_Matrices_Type_In Type of the cost matrices used in the MPC
 * formulation. Must define state, input, and output sizes, as well as horizon
 * length.
 */
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
        _Y_store(), _cost_function(nullptr),
        _cost_and_gradient_function(nullptr), _hvp_function(nullptr),
        _solver() {}

  NonlinearMPC_TwiceDifferentiable(EKF_Type &kalman_filter,
                                   Cost_Matrices_Type &cost_matrices,
                                   _T delta_time, X_Type X_initial)
      : U_horizon(), _kalman_filter(kalman_filter),
        _sqp_cost_matrices(cost_matrices), _delta_time(delta_time), _Y_store(),
        _cost_function(), _cost_and_gradient_function(), _hvp_function(),
        _solver() {

    this->_kalman_filter.set_x_hat(X_initial);

    this->_initialize_solver(X_initial);
  }

  /* Copy Constructor */
  NonlinearMPC_TwiceDifferentiable(
      const NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type>
          &input)
      : U_horizon(input.U_horizon), _kalman_filter(input._kalman_filter),
        _sqp_cost_matrices(input._sqp_cost_matrices),
        _delta_time(input._delta_time), _Y_store(input._Y_store),
        _cost_function(input._cost_function),
        _cost_and_gradient_function(input._cost_and_gradient_function),
        _hvp_function(input._hvp_function), _solver(input._solver) {
    this->_bind_cost_functions();
  }

  NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type> &
  operator=(const NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type>
                &input) {
    if (this != &input) {
      this->U_horizon = input.U_horizon;
      this->_kalman_filter = input._kalman_filter;
      this->_sqp_cost_matrices = input._sqp_cost_matrices;
      this->_delta_time = input._delta_time;
      this->_Y_store = input._Y_store;

      this->_cost_function = {};
      this->_cost_and_gradient_function = {};
      this->_hvp_function = {};

      this->_solver = input._solver;

      this->_bind_cost_functions();
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
        _Y_store(std::move(input._Y_store)),
        _cost_function(std::move(input._cost_function)),
        _cost_and_gradient_function(
            std::move(input._cost_and_gradient_function)),
        _hvp_function(std::move(input._hvp_function)),
        _solver(std::move(input._solver)) {

    this->_bind_cost_functions();
  }

  NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type> &
  operator=(NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type>
                &&input) noexcept {
    if (this != &input) {
      this->U_horizon = std::move(input.U_horizon);
      this->_kalman_filter = std::move(input._kalman_filter);
      this->_sqp_cost_matrices = std::move(input._sqp_cost_matrices);
      this->_delta_time = std::move(input._delta_time);
      this->_Y_store = std::move(input._Y_store);

      this->_cost_function = {};
      this->_cost_and_gradient_function = {};
      this->_hvp_function = {};

      this->_solver = std::move(input._solver);
    }
    return *this;
  }

public:
  /* Setter */

  /**
   * @brief Sets the maximum number of iterations for the solver.
   *
   * This function configures the solver to limit the number of iterations it
   * performs during the optimization process. Setting an appropriate maximum
   * can help control computation time and ensure timely convergence.
   *
   * @param max_iteration The maximum number of iterations the solver is allowed
   * to perform.
   */
  inline void set_solver_max_iteration(std::size_t max_iteration) {
    this->_solver.set_solver_max_iteration(max_iteration);
  }

  /**
   * @brief Sets the reference trajectory for the MPC.
   *
   * This function allows setting the reference trajectory that the MPC will
   * use to compute control inputs. The reference can be provided as either a
   * trajectory matrix or a single reference vector. The function ensures that
   * the input dimensions are compatible with the expected output size and
   * prediction horizon.
   *
   * @tparam Reference_Type_In Type of the reference input, which must have
   * static members `ROWS` and `COLS`.
   * @param reference The reference input, which can be a trajectory matrix or
   * a single vector.
   *
   * @note The function uses static assertions to validate the dimensions of
   * the input reference against the expected output size and prediction
   * horizon.
   */
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

  /**
   * @brief Retrieves the number of iterations performed in the last solver
   * step.
   *
   * This function returns the number of iterations that the underlying solver
   * executed during the most recent solver step. It can be useful for
   * monitoring solver performance or debugging convergence issues.
   *
   * @return std::size_t The number of iterations performed in the last solver
   * step.
   */
  inline auto get_solver_step_iterated_number(void) const -> std::size_t {
    return this->_solver.get_solver_step_iterated_number();
  }

  /**
   * @brief Retrieves the current estimated state from the Kalman filter.
   *
   * This function returns the current state estimate maintained by the
   * Extended Kalman Filter (EKF) integrated within the MPC framework. The
   * state estimate is used for prediction and control input calculation.
   *
   * @return X_Type The current estimated state from the Kalman filter.
   */
  inline auto get_X(void) const -> X_Type {
    return this->_kalman_filter.get_x_hat();
  }

  /* Function */

  /**
   * @brief Calculates the current control input from the control input
   * horizon.
   *
   * This function extracts the first control input from the control input
   * horizon, which represents the immediate action to be taken by the system.
   * The control input horizon is typically a sequence of future control inputs
   * computed by the MPC algorithm.
   *
   * @param U_horizon_in The control input horizon from which to extract the
   * current control input.
   * @return U_Type The current control input extracted from the horizon.
   */
  inline auto calculate_this_U(const U_Horizon_Type &U_horizon_in) -> U_Type {

    auto U = PythonNumpy::get_row<0>(U_horizon_in);

    return U;
  }

  /**
   * @brief Updates the parameters of the Kalman filter and cost matrices.
   *
   * This function allows updating the parameters used by both the Extended
   * Kalman Filter (EKF) and the cost matrices in the MPC formulation. It is
   * important that the parameter types for both components are compatible to
   * ensure consistent behavior.
   *
   * @param parameters The new parameters to be set for the EKF and cost
   * matrices.
   *
   * @note The function assumes that the parameter types of the EKF and cost
   * matrices are the same.
   */
  inline void update_parameters(const _Parameter_Type &parameters) {
    // when you use this function, parameters type of EKF and CostMatrices must
    // be same

    this->_kalman_filter.parameters = parameters;
    this->_sqp_cost_matrices.state_space_parameters = parameters;
  }

  /**
   * @brief Updates the control input based on the current reference and
   * measurement.
   *
   * This function performs a complete update cycle for the MPC, including
   * state estimation using the Extended Kalman Filter (EKF) and optimization
   * of the control input horizon. It takes the current reference and
   * measurement as inputs, compensates for any delays, and computes the
   * optimal control input to be applied to the system.
   *
   * @tparam Reference_Type_In Type of the reference input, which must have
   * static members `ROWS` and `COLS`.
   * @param reference The current reference input for the MPC.
   * @param Y The current measurement used for state estimation.
   * @return U_Type The updated control input to be applied to the system.
   *
   * @note The function assumes that the reference input dimensions are
   * compatible with the expected output size and prediction horizon.
   */
  template <typename Reference_Type_In>
  inline auto update_manipulation(Reference_Type_In &reference,
                                  const Y_Type &Y) -> U_Type {

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

  /**
   * @brief Binds the cost functions to the solver.
   *
   * This function sets up the cost function, cost gradient function, and
   * Hessian-vector product (HVP) function for the solver using lambda
   * expressions. These functions are essential for the optimization process
   * and are defined based on the provided cost matrices.
   */
  inline void _bind_cost_functions() {
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
  }

  /**
   * @brief Initializes the solver with the initial state and cost functions.
   *
   * This function sets up the solver by defining the cost function, cost
   * gradient function, and Hessian-vector product (HVP) function based on the
   * provided cost matrices. It also configures the solver with the initial
   * state and sets a default maximum number of iterations.
   *
   * @param X_initial The initial state to be used for solver initialization.
   */
  inline void _initialize_solver(const X_Type &X_initial) {

    this->_bind_cost_functions();

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

  Y_Store_Type _Y_store;

  _ConstFunction_Object_Type _cost_function;
  _CostAndGradientFunction_Object_Type _cost_and_gradient_function;
  _HVP_Function_Object_Type _hvp_function;

  _Solver_Type _solver;
}; // namespace NonlinearMPC_ReferenceTrajectoryOperation

/* make NonlinearMPC_TwiceDifferentiable */

/**
 * @brief Factory function to create a NonlinearMPC_TwiceDifferentiable object.
 *
 * This function constructs and returns a NonlinearMPC_TwiceDifferentiable
 * object, initializing it with the provided Extended Kalman Filter (EKF),
 * cost matrices, time step, and initial state. It ensures that the data types
 * of the EKF and cost matrices match the specified type T.
 *
 * @tparam EKF_Type Type of the Extended Kalman Filter (EKF) used for state
 * estimation. Must provide methods for state prediction and update.
 * @tparam Cost_Matrices_Type Type of the cost matrices used in the MPC
 * formulation. Must define state, input, and output sizes, as well as horizon
 * length.
 * @tparam T Data type for the EKF and cost matrices (e.g., float, double).
 * @param kalman_filter Reference to an EKF object for state estimation.
 * @param cost_matrices Reference to a cost matrices object for MPC.
 * @param delta_time Time step for the MPC updates.
 * @param X_initial Initial state for the EKF.
 * @return NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type>
 * Initialized NonlinearMPC_TwiceDifferentiable object.
 *
 * @note The function uses static assertions to ensure that the data types of
 * the EKF and cost matrices match the specified type T.
 */
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

/**
 * @brief Type alias for NonlinearMPC_TwiceDifferentiable.
 *
 * This alias simplifies the usage of the NonlinearMPC_TwiceDifferentiable
 * class template by providing a shorter and more convenient name. It can be
 * used to instantiate NonlinearMPC_TwiceDifferentiable objects with specific
 * EKF and cost matrices types.
 *
 * @tparam EKF_Type Type of the Extended Kalman Filter (EKF) used for state
 * estimation. Must provide methods for state prediction and update.
 * @tparam Cost_Matrices_Type Type of the cost matrices used in the MPC
 * formulation. Must define state, input, and output sizes, as well as horizon
 * length.
 */
template <typename EKF_Type, typename Cost_Matrices_Type>
using NonlinearMPC_TwiceDifferentiable_Type =
    NonlinearMPC_TwiceDifferentiable<EKF_Type, Cost_Matrices_Type>;

} // namespace PythonMPC

#endif // __PYTHON_NONLINEAR_MPC_HPP__
