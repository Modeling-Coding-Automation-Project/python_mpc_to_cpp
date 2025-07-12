/**
 * @file python_linear_mpc.hpp
 * @brief Linear Model Predictive Control (MPC) implementation for Python/C++
 * integration.
 *
 * This header provides a set of template classes and utilities for implementing
 * Linear Model Predictive Control (MPC) algorithms, with and without
 * constraints, in C++. The code is designed to be flexible and extensible,
 * supporting various types of Kalman filters, prediction matrices, and solver
 * factors, and is intended for use in projects that bridge Python and C++ for
 * control system applications.
 */
#ifndef __PYTHON_LINEAR_MPC_HPP__
#define __PYTHON_LINEAR_MPC_HPP__

#include "mpc_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <functional>
#include <type_traits>

namespace PythonMPC {

class SolverFactor_Empty {};

namespace LMPC_Operation {

/**
 * @brief Compensates the delay in the state and output vectors for the
 * Linear Model Predictive Control (LMPC) operation.
 *
 * This function adjusts the state and output vectors to account for delays
 * in the system, ensuring that the control inputs are correctly aligned with
 * the measured outputs.
 *
 * @tparam Number_Of_Delay The number of delays to compensate for.
 * @tparam X_Type Type of the state vector.
 * @tparam Y_Type Type of the output vector.
 * @tparam Y_Store_Type Type of the delayed output storage.
 * @tparam LKF_Type Type of the Kalman filter used in LMPC.
 */
template <std::size_t Number_Of_Delay, typename X_Type, typename Y_Type,
          typename Y_Store_Type, typename LKF_Type>
inline typename std::enable_if<(Number_Of_Delay > 0), void>::type
compensate_X_Y_delay(const X_Type &X_in, const Y_Type &Y_in, X_Type &X_out,
                     Y_Type &Y_out, Y_Store_Type &Y_store,
                     LKF_Type &kalman_filter) {

  static_cast<void>(X_in);

  Y_Type Y_measured = Y_in;

  X_out = kalman_filter.get_x_hat_without_delay();
  auto Y = kalman_filter.state_space.C * X_out;

  Y_store.push(Y);
  auto Y_diff = Y_measured - Y_store.get();

  Y_out = Y + Y_diff;
}

/**
 * @brief Specialization of the compensate_X_Y_delay function for the case
 * where there are no delays.
 *
 * This specialization simply copies the input state and output vectors to
 * the output vectors without any delay compensation.
 *
 * @tparam Number_Of_Delay The number of delays (should be 0 for this
 * specialization).
 * @tparam X_Type Type of the state vector.
 * @tparam Y_Type Type of the output vector.
 * @tparam Y_Store_Type Type of the delayed output storage (not used here).
 * @tparam LKF_Type Type of the Kalman filter (not used here).
 */
template <std::size_t Number_Of_Delay, typename X_Type, typename Y_Type,
          typename Y_Store_Type, typename LKF_Type>
inline typename std::enable_if<(Number_Of_Delay == 0), void>::type
compensate_X_Y_delay(const X_Type &X_in, const Y_Type &Y_in, X_Type &X_out,
                     Y_Type &Y_out, Y_Store_Type &Y_store,
                     LKF_Type &kalman_filter) {

  static_cast<void>(Y_store);
  static_cast<void>(kalman_filter);

  X_out = X_in;
  Y_out = Y_in;
}

template <typename U_Type, typename U_Horizon_Type, std::size_t Index>
struct Integrate_U {
  /**
   * @brief Recursively integrates the control input over the horizon.
   *
   * This function updates the control input U by adding the corresponding
   * delta_U_Horizon value for the given index.
   *
   * @param U The current control input to be updated.
   * @param delta_U_Horizon The delta control input for the horizon.
   */
  static void calculate(U_Type &U, const U_Horizon_Type &delta_U_Horizon) {

    U.template set<Index, 0>(U.template get<Index, 0>() +
                             delta_U_Horizon.template get<Index, 0>());
    Integrate_U<U_Type, U_Horizon_Type, Index - 1>::calculate(U,
                                                              delta_U_Horizon);
  }
};

template <typename U_Type, typename U_Horizon_Type>
struct Integrate_U<U_Type, U_Horizon_Type, 0> {
  /**
   * @brief Base case for the recursive integration of control input.
   *
   * This function updates the first element of the control input U by adding
   * the corresponding delta_U_Horizon value.
   *
   * @param U The current control input to be updated.
   * @param delta_U_Horizon The delta control input for the horizon.
   */
  static void calculate(U_Type &U, const U_Horizon_Type &delta_U_Horizon) {

    U.template set<0, 0>(U.template get<0, 0>() +
                         delta_U_Horizon.template get<0, 0>());
  }
};

/**
 * @brief Solves the Linear Model Predictive Control (LMPC) problem without
 * constraints.
 *
 * This function computes the control input delta_U by applying the solver
 * factor to the reference trajectory, adjusted by the system dynamics matrix F
 * and the augmented state vector X_augmented.
 *
 * @tparam SolverFactor_Type Type of the solver factor used in LMPC.
 * @tparam F_Type Type of the system dynamics matrix.
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory.
 * @tparam X_Augmented_Type Type of the augmented state vector.
 * @tparam U_Horizon_Type Type of the control input horizon.
 *
 * @param solver_factor The solver factor used to scale the reference
 * trajectory.
 * @param F The system dynamics matrix.
 * @param reference_trajectory The reference trajectory object that provides
 * delta_U calculation.
 * @param X_augmented The augmented state vector containing current state and
 * output information.
 * @param delta_U The output control input for the horizon, which will be
 * modified in place.
 */
template <typename SolverFactor_Type, typename F_Type,
          typename ReferenceTrajectory_Type, typename X_Augmented_Type,
          typename U_Horizon_Type>
inline void solve_LMPC_No_Constraints(
    const SolverFactor_Type &solver_factor, const F_Type &F,
    ReferenceTrajectory_Type &reference_trajectory,
    const X_Augmented_Type &X_augmented, U_Horizon_Type &delta_U) {

  delta_U = solver_factor * reference_trajectory.calculate_dif(F * X_augmented);
}

/**
 * @brief Calculates the control input U based on the latest control input and
 * the delta control input.
 *
 * This function computes the new control input U by adding the delta_U to the
 * latest control input U_latest.
 *
 * @tparam U_Type Type of the control input.
 *
 * @param U_latest The latest control input.
 * @param delta_U The change in control input to be applied.
 * @return The updated control input U.
 */
template <typename U_Type>
inline auto calculate_this_U(const U_Type &U_latest, const U_Type &delta_U)
    -> U_Type {

  auto U = U_latest + delta_U;

  return U;
}

} // namespace LMPC_Operation

/**
 * @brief Linear Model Predictive Control (MPC) class without constraints.
 *
 * This class implements a basic linear MPC algorithm that does not enforce
 * constraints on the control inputs or outputs. It uses a Kalman filter for
 * state estimation and prediction matrices for system dynamics.
 *
 * @tparam LKF_Type_In Type of the Kalman filter used in the MPC.
 * @tparam PredictionMatrices_Type_In Type of the prediction matrices used in
 * the MPC.
 * @tparam ReferenceTrajectory_Type_In Type of the reference trajectory used in
 * the MPC.
 * @tparam SolverFactor_Type_In Type of the solver factor used in the MPC (can
 * be empty).
 */
template <typename LKF_Type_In, typename PredictionMatrices_Type_In,
          typename ReferenceTrajectory_Type_In,
          typename SolverFactor_Type_In = SolverFactor_Empty>
class LTI_MPC_NoConstraints {
public:
  /* Type */
  using LKF_Type = LKF_Type_In;
  using PredictionMatrices_Type = PredictionMatrices_Type_In;
  using ReferenceTrajectory_Type = ReferenceTrajectory_Type_In;

protected:
  /* Type */
  using _T = typename PredictionMatrices_Type::Value_Type;

public:
  /* Type */
  using Value_Type = _T;

  static constexpr std::size_t INPUT_SIZE = LKF_Type::INPUT_SIZE;
  static constexpr std::size_t STATE_SIZE = LKF_Type::STATE_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = LKF_Type::OUTPUT_SIZE;

  static constexpr std::size_t NP = PredictionMatrices_Type::NP;
  static constexpr std::size_t NC = PredictionMatrices_Type::NC;

  static constexpr std::size_t NUMBER_OF_DELAY = LKF_Type::NUMBER_OF_DELAY;

  using U_Type = PythonControl::StateSpaceInput_Type<_T, INPUT_SIZE>;
  using X_Type = PythonControl::StateSpaceState_Type<_T, STATE_SIZE>;
  using Y_Type = PythonControl::StateSpaceOutput_Type<_T, OUTPUT_SIZE>;

  using U_Horizon_Type =
      PythonControl::StateSpaceInput_Type<_T, (INPUT_SIZE * NC)>;
  using X_Augmented_Type =
      PythonControl::StateSpaceState_Type<_T, (STATE_SIZE + OUTPUT_SIZE)>;
  using Y_Store_Type =
      PythonControl::DelayedVectorObject<Y_Type, NUMBER_OF_DELAY>;

  typedef typename std::conditional<
      std::is_same<SolverFactor_Type_In, SolverFactor_Empty>::value,
      PythonNumpy::DenseMatrix_Type<_T, (INPUT_SIZE * NC), (OUTPUT_SIZE * NP)>,
      SolverFactor_Type_In>::type SolverFactor_Type;

  static_assert(SolverFactor_Type::COLS == (INPUT_SIZE * NC),
                "SolverFactor_Type::COLS must be equal to (INPUT_SIZE * NC)");
  static_assert(SolverFactor_Type::ROWS == (OUTPUT_SIZE * NP),
                "SolverFactor_Type::ROWS must be equal to (OUTPUT_SIZE * "
                "NP)");

public:
  /* Constructor */
  LTI_MPC_NoConstraints()
      : _kalman_filter(), _prediction_matrices(), _reference_trajectory(),
        _solver_factor(), _X_inner_model(), _U_latest(), _Y_store() {}

  template <typename LKF_Type, typename PredictionMatrices_Type,
            typename ReferenceTrajectory_Type,
            typename SolverFactor_Type_In_Constructor>
  LTI_MPC_NoConstraints(
      const LKF_Type &kalman_filter,
      const PredictionMatrices_Type &prediction_matrices,
      const ReferenceTrajectory_Type &reference_trajectory,
      const SolverFactor_Type_In_Constructor &solver_factor_in)
      : _kalman_filter(kalman_filter),
        _prediction_matrices(prediction_matrices),
        _reference_trajectory(reference_trajectory), _solver_factor(),
        _X_inner_model(), _U_latest(), _Y_store() {

    static_assert(SolverFactor_Type::COLS ==
                      SolverFactor_Type_In_Constructor::COLS,
                  "SolverFactor_Type::COL must be equal to "
                  "SolverFactor_Type_In_Constructor::COL");
    static_assert(SolverFactor_Type::ROWS ==
                      SolverFactor_Type_In_Constructor::ROWS,
                  "SolverFactor_Type::ROW must be equal to "
                  "SolverFactor_Type_In_Constructor::ROW");

    // This is because the solver_factor_in can be different type from
    // "SolverFactor_Type".
    PythonNumpy::substitute_matrix(this->_solver_factor, solver_factor_in);
  }

  /* Copy Constructor */
  LTI_MPC_NoConstraints(
      const LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                                  ReferenceTrajectory_Type> &input)
      : _kalman_filter(input._kalman_filter),
        _prediction_matrices(input._prediction_matrices),
        _reference_trajectory(input._reference_trajectory),
        _solver_factor(input._solver_factor),
        _X_inner_model(input._X_inner_model), _U_latest(input._U_latest),
        _Y_store(input._Y_store) {}

  LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                        ReferenceTrajectory_Type> &
  operator=(const LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                                        ReferenceTrajectory_Type> &input) {
    if (this != &input) {
      this->_kalman_filter = input._kalman_filter;
      this->_prediction_matrices = input._prediction_matrices;
      this->_reference_trajectory = input._reference_trajectory;
      this->_solver_factor = input._solver_factor;
      this->_X_inner_model = input._X_inner_model;
      this->_U_latest = input._U_latest;
      this->_Y_store = input._Y_store;
    }
    return *this;
  }

  /* Move Constructor */
  LTI_MPC_NoConstraints(
      LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                            ReferenceTrajectory_Type> &&input) noexcept
      : _kalman_filter(std::move(input._kalman_filter)),
        _prediction_matrices(std::move(input._prediction_matrices)),
        _reference_trajectory(std::move(input._reference_trajectory)),
        _solver_factor(std::move(input._solver_factor)),
        _X_inner_model(std::move(input._X_inner_model)),
        _U_latest(std::move(input._U_latest)),
        _Y_store(std::move(input._Y_store)) {}

  LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                        ReferenceTrajectory_Type> &
  operator=(LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                                  ReferenceTrajectory_Type> &&input) noexcept {
    if (this != &input) {
      this->_kalman_filter = std::move(input._kalman_filter);
      this->_prediction_matrices = std::move(input._prediction_matrices);
      this->_reference_trajectory = std::move(input._reference_trajectory);
      this->_solver_factor = std::move(input._solver_factor);
      this->_X_inner_model = std::move(input._X_inner_model);
      this->_U_latest = std::move(input._U_latest);
      this->_Y_store = std::move(input._Y_store);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Sets the reference trajectory for the MPC.
   *
   * This function updates the reference trajectory used by the MPC to
   * calculate control inputs based on the provided reference vector.
   *
   * @tparam Ref_Type Type of the reference vector.
   * @param ref The reference vector to be set.
   */
  template <typename Ref_Type>
  inline void set_reference_trajectory(const Ref_Type &ref) {

    static_assert(std::is_same<typename Ref_Type::Value_Type, _T>::value,
                  "Ref_Type::Value_Type must be equal to Value_Type");

    this->_reference_trajectory.reference_vector = ref;
  }

  /**
   * @brief Updates the control input based on the current state and reference.
   *
   * This function performs a prediction step using the Kalman filter,
   * compensates for delays in the state and output vectors, and calculates the
   * new control input based on the reference trajectory and the current state.
   *
   * @tparam Ref_Type Type of the reference vector.
   * @param reference The reference vector to be used for updating control
   * input.
   * @param Y The measured output vector.
   * @return The updated control input vector.
   */
  template <typename Ref_Type>
  inline auto update(const Ref_Type &reference, const Y_Type &Y) -> U_Type {

    this->_kalman_filter.predict_and_update_with_fixed_G(this->_U_latest, Y);

    X_Type X = this->_kalman_filter.get_x_hat();

    X_Type X_compensated;
    Y_Type Y_compensated;
    this->_compensate_X_Y_delay(X, Y, X_compensated, Y_compensated);

    auto delta_X = X_compensated - this->_X_inner_model;
    auto X_augmented =
        PythonNumpy::concatenate_vertically(delta_X, Y_compensated);

    this->_reference_trajectory.reference = reference;

    auto delta_U = this->_solve(X_augmented);

    LMPC_Operation::Integrate_U<U_Type, U_Horizon_Type,
                                (INPUT_SIZE - 1)>::calculate(this->_U_latest,
                                                             delta_U);

    this->_X_inner_model = X_compensated;

    return this->_U_latest;
  }

protected:
  /* Function */

  /**
   * @brief Compensates for delays in the state and output vectors.
   *
   * This function adjusts the state and output vectors to account for delays
   * in the system, ensuring that the control inputs are correctly aligned with
   * the measured outputs.
   *
   * @param X_in The input state vector.
   * @param Y_in The input output vector.
   * @param X_out The compensated output state vector.
   * @param Y_out The compensated output vector.
   */
  inline void _compensate_X_Y_delay(const X_Type &X_in, const Y_Type &Y_in,
                                    X_Type &X_out, Y_Type &Y_out) {

    LMPC_Operation::compensate_X_Y_delay<NUMBER_OF_DELAY>(
        X_in, Y_in, X_out, Y_out, this->_Y_store, this->_kalman_filter);
  }

  /**
   * @brief Solves the MPC optimization problem to calculate the control input.
   *
   * This function computes the change in control input (delta_U) based on the
   * augmented state vector (X_augmented) and the reference trajectory.
   *
   * @param X_augmented The augmented state vector containing the current state
   * and output.
   * @return The calculated change in control input (delta_U).
   */
  virtual inline auto _solve(const X_Augmented_Type &X_augmented)
      -> U_Horizon_Type {

    U_Horizon_Type delta_U;

    LMPC_Operation::solve_LMPC_No_Constraints(
        this->_solver_factor, this->_prediction_matrices.F,
        this->_reference_trajectory, X_augmented, delta_U);

    return delta_U;
  }

  /**
   * @brief Calculates the new control input based on the latest control input
   * and the change in control input (delta_U).
   *
   * This function updates the control input by adding the change in control
   * input to the latest control input.
   *
   * @param delta_U The change in control input to be applied.
   * @return The updated control input.
   */
  inline auto _calculate_this_U(const U_Type &delta_U) -> U_Type {

    auto U = LMPC_Operation::calculate_this_U(this->_U_latest, delta_U);

    return U;
  }

protected:
  /* Variables */
  LKF_Type _kalman_filter;
  PredictionMatrices_Type _prediction_matrices;
  ReferenceTrajectory_Type _reference_trajectory;

  SolverFactor_Type _solver_factor;

  X_Type _X_inner_model;
  U_Type _U_latest;
  Y_Store_Type _Y_store;
};

/* make LTI MPC No Constraints */

/**
 * @brief Factory function to create an instance of LTI_MPC_NoConstraints.
 *
 * This function initializes the LTI_MPC_NoConstraints class with the provided
 * Kalman filter, prediction matrices, reference trajectory, and solver factor.
 *
 * @tparam LKF_Type Type of the Kalman filter.
 * @tparam PredictionMatrices_Type Type of the prediction matrices.
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory.
 * @tparam SolverFactor_Type Type of the solver factor (optional).
 * @param kalman_filter The Kalman filter to be used in the MPC.
 * @param prediction_matrices The prediction matrices for the MPC.
 * @param reference_trajectory The reference trajectory for the MPC.
 * @param solver_factor The solver factor for the MPC (optional).
 * @return An instance of LTI_MPC_NoConstraints initialized with the provided
 * parameters.
 */
template <typename LKF_Type, typename PredictionMatrices_Type,
          typename ReferenceTrajectory_Type, typename SolverFactor_Type>
inline auto
make_LTI_MPC_NoConstraints(const LKF_Type &kalman_filter,
                           const PredictionMatrices_Type &prediction_matrices,
                           const ReferenceTrajectory_Type &reference_trajectory,
                           const SolverFactor_Type &solver_factor)
    -> LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                             ReferenceTrajectory_Type, SolverFactor_Type> {

  return LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                               ReferenceTrajectory_Type, SolverFactor_Type>(
      kalman_filter, prediction_matrices, reference_trajectory, solver_factor);
}

/* LTI MPC No Constraints Type */
template <typename LKF_Type, typename PredictionMatrices_Type,
          typename ReferenceTrajectory_Type,
          typename SolverFactor_Type = SolverFactor_Empty>
using LTI_MPC_NoConstraints_Type =
    LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                          ReferenceTrajectory_Type, SolverFactor_Type>;

/* LTI MPC */

/**
 * @brief Linear Model Predictive Control (MPC) class with constraints.
 *
 * This class extends the LTI_MPC_NoConstraints class to include constraints on
 * the control inputs and outputs, using a quadratic programming solver to
 * compute the optimal control inputs while respecting these constraints.
 *
 * @tparam LKF_Type Type of the Kalman filter used in the MPC.
 * @tparam PredictionMatrices_Type Type of the prediction matrices used in the
 * MPC.
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory used in
 * the MPC.
 * @tparam Weight_U_Nc_Type Type for the weight matrix for control input
 * changes.
 * @tparam Delta_U_Min_Type Type for the minimum change in control input.
 * @tparam Delta_U_Max_Type Type for the maximum change in control input.
 * @tparam U_Min_Type Type for the minimum control input.
 * @tparam U_Max_Type Type for the maximum control input.
 * @tparam Y_Min_Type Type for the minimum output constraint.
 * @tparam Y_Max_Type Type for the maximum output constraint.
 * @tparam SolverFactor_Type_In Type of the solver factor used in the MPC (can
 * be empty).
 */
template <typename LKF_Type, typename PredictionMatrices_Type,
          typename ReferenceTrajectory_Type, typename Weight_U_Nc_Type,
          typename Delta_U_Min_Type, typename Delta_U_Max_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type,
          typename SolverFactor_Type_In = SolverFactor_Empty>
class LTI_MPC : public LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                                             ReferenceTrajectory_Type,
                                             SolverFactor_Type_In> {

protected:
  /* Type */
  using _LTI_MPC_NoConstraints_Type =
      LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                            ReferenceTrajectory_Type, SolverFactor_Type_In>;

  using _U_Horizon_Type = typename _LTI_MPC_NoConstraints_Type::U_Horizon_Type;

  using _X_Augmented_Type =
      typename _LTI_MPC_NoConstraints_Type::X_Augmented_Type;

  using _Solver_Type = LTI_MPC_QP_Solver_Type<
      _U_Horizon_Type::COLS, _LTI_MPC_NoConstraints_Type::OUTPUT_SIZE,
      typename _LTI_MPC_NoConstraints_Type::U_Type, _X_Augmented_Type,
      typename PredictionMatrices_Type::Phi_Type,
      typename PredictionMatrices_Type::F_Type, Weight_U_Nc_Type,
      Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type, U_Max_Type, Y_Min_Type,
      Y_Max_Type>;

public:
  /* Constructor */
  LTI_MPC() : _LTI_MPC_NoConstraints_Type(), _solver() {}

  template <typename SolverFactor_Type>
  LTI_MPC(const LKF_Type &kalman_filter,
          const PredictionMatrices_Type &prediction_matrices,
          const ReferenceTrajectory_Type &reference_trajectory,
          const Weight_U_Nc_Type &Weight_U_Nc,
          const Delta_U_Min_Type &delta_U_min,
          const Delta_U_Max_Type &delta_U_max, const U_Min_Type &U_min,
          const U_Max_Type &U_max, const Y_Min_Type &Y_min,
          const Y_Max_Type &Y_max, const SolverFactor_Type &solver_factor_in)
      : _LTI_MPC_NoConstraints_Type(kalman_filter, prediction_matrices,
                                    reference_trajectory, solver_factor_in),
        _solver() {

    _U_Horizon_Type delta_U_Nc;

    auto X_augmented = PythonNumpy::concatenate_vertically(
        this->_X_inner_model, this->_Y_store.get());

    this->_solver =
        make_LTI_MPC_QP_Solver<_U_Horizon_Type::COLS,
                               _LTI_MPC_NoConstraints_Type::OUTPUT_SIZE>(
            this->_U_latest, X_augmented, this->_prediction_matrices.Phi,
            this->_prediction_matrices.F, Weight_U_Nc, delta_U_min, delta_U_max,
            U_min, U_max, Y_min, Y_max);
  }

  /* Copy Constructor */
  LTI_MPC(const LTI_MPC &other)
      : LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                              ReferenceTrajectory_Type, SolverFactor_Type_In>(
            other),
        _solver(other._solver) {}

  LTI_MPC &operator=(const LTI_MPC &other) {
    if (this != &other) {
      this->LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                                  ReferenceTrajectory_Type,
                                  SolverFactor_Type_In>::operator=(other);
      this->_solver = other._solver;
    }
    return *this;
  }

  /* Move Constructor */
  LTI_MPC(LTI_MPC &&other)
  noexcept
      : LTI_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type,
                              ReferenceTrajectory_Type, SolverFactor_Type_In>(
            std::move(other)),
        _solver(std::move(other._solver)) {}

  LTI_MPC &operator=(LTI_MPC &&other) noexcept {
    if (this != &other) {
      this->LTI_MPC_NoConstraints<
          LKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,
          SolverFactor_Type_In>::operator=(std::move(other));
      this->_solver = std::move(other._solver);
    }
    return *this;
  }

protected:
  /* Function */

  /**
   * @brief Solves the MPC optimization problem to calculate the control input
   * with constraints.
   *
   * This function computes the change in control input (delta_U) based on the
   * augmented state vector (X_augmented) and the reference trajectory, while
   * considering constraints on the control inputs and outputs.
   *
   * @param X_augmented The augmented state vector containing the current state
   * and output.
   * @return The calculated change in control input (delta_U).
   */
  inline auto _solve(const _X_Augmented_Type &X_augmented)
      -> _U_Horizon_Type override {

    this->_solver.update_constraints(this->_U_latest, X_augmented,
                                     this->_prediction_matrices.Phi,
                                     this->_prediction_matrices.F);

    auto delta_U = this->_solver.solve(
        this->_prediction_matrices.Phi, this->_prediction_matrices.F,
        this->_reference_trajectory, X_augmented);

    return delta_U;
  }

protected:
  /* Variables */
  _Solver_Type _solver;
};

/* make LTI MPC */

/**
 * @brief Factory function to create an instance of LTI_MPC.
 *
 * This function initializes the LTI_MPC class with the provided Kalman filter,
 * prediction matrices, reference trajectory, and various constraints and
 * solver factors.
 *
 * @tparam LKF_Type Type of the Kalman filter.
 * @tparam PredictionMatrices_Type Type of the prediction matrices.
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory.
 * @tparam Weight_U_Nc_Type Type for the weight matrix for control input
 * changes.
 * @tparam Delta_U_Min_Type Type for the minimum change in control input.
 * @tparam Delta_U_Max_Type Type for the maximum change in control input.
 * @tparam U_Min_Type Type for the minimum control input.
 * @tparam U_Max_Type Type for the maximum control input.
 * @tparam Y_Min_Type Type for the minimum output constraint.
 * @tparam Y_Max_Type Type for the maximum output constraint.
 * @tparam SolverFactor_Type_In Type of the solver factor used in the MPC (can
 * be empty).
 * @return An instance of LTI_MPC initialized with the provided parameters.
 */
template <typename LKF_Type, typename PredictionMatrices_Type,
          typename ReferenceTrajectory_Type, typename Weight_U_Nc_Type,
          typename Delta_U_Min_Type, typename Delta_U_Max_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type,
          typename SolverFactor_Type_In = SolverFactor_Empty>
inline auto make_LTI_MPC(const LKF_Type &kalman_filter,
                         const PredictionMatrices_Type &prediction_matrices,
                         const ReferenceTrajectory_Type &reference_trajectory,
                         const Weight_U_Nc_Type &Weight_U_Nc,
                         const Delta_U_Min_Type &delta_U_min,
                         const Delta_U_Max_Type &delta_U_max,
                         const U_Min_Type &U_min, const U_Max_Type &U_max,
                         const Y_Min_Type &Y_min, const Y_Max_Type &Y_max,
                         const SolverFactor_Type_In &solver_factor_in)
    -> LTI_MPC<LKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,
               Weight_U_Nc_Type, Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
               U_Max_Type, Y_Min_Type, Y_Max_Type, SolverFactor_Type_In> {

  return LTI_MPC<LKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,
                 Weight_U_Nc_Type, Delta_U_Min_Type, Delta_U_Max_Type,
                 U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
                 SolverFactor_Type_In>(
      kalman_filter, prediction_matrices, reference_trajectory, Weight_U_Nc,
      delta_U_min, delta_U_max, U_min, U_max, Y_min, Y_max, solver_factor_in);
}

/* LTI MPC Type */
template <typename LKF_Type, typename PredictionMatrices_Type,
          typename ReferenceTrajectory_Type, typename Weight_U_Nc_Type,
          typename Delta_U_Min_Type, typename Delta_U_Max_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type,
          typename SolverFactor_Type_In = SolverFactor_Empty>
using LTI_MPC_Type =
    LTI_MPC<LKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,
            Weight_U_Nc_Type, Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
            U_Max_Type, Y_Min_Type, Y_Max_Type, SolverFactor_Type_In>;

/* LTV MPC Function Object */

template <typename Parameter_Type, typename MPC_StateSpace_Updater_Output_Type>
using MPC_StateSpace_Updater_Function_Object = std::function<void(
    const Parameter_Type &, MPC_StateSpace_Updater_Output_Type &)>;

template <typename StateSpace_Type, typename Parameter_Type, typename Phi_Type,
          typename F_Type>
using LTV_MPC_Phi_F_Updater_Function_Object =
    std::function<void(const Parameter_Type &, Phi_Type &, F_Type &)>;

/* LTV MPC No Constraints */

template <typename LKF_Type_In, typename PredictionMatrices_Type_In,
          typename ReferenceTrajectory_Type_In, typename Parameter_Type_In,
          typename SolverFactor_Type_In = SolverFactor_Empty>
class LTV_MPC_NoConstraints {
public:
  /* Type */
  using LKF_Type = LKF_Type_In;
  using PredictionMatrices_Type = PredictionMatrices_Type_In;
  using ReferenceTrajectory_Type = ReferenceTrajectory_Type_In;
  using Parameter_Type = Parameter_Type_In;

protected:
  /* Type */
  using _T = typename PredictionMatrices_Type::Value_Type;

public:
  /* Type */
  using Value_Type = _T;

  static constexpr std::size_t INPUT_SIZE = LKF_Type::INPUT_SIZE;
  static constexpr std::size_t STATE_SIZE = LKF_Type::STATE_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = LKF_Type::OUTPUT_SIZE;

  static constexpr std::size_t NP = PredictionMatrices_Type::NP;
  static constexpr std::size_t NC = PredictionMatrices_Type::NC;

  static constexpr std::size_t NUMBER_OF_DELAY = LKF_Type::NUMBER_OF_DELAY;

  using U_Type = PythonControl::StateSpaceInput_Type<_T, INPUT_SIZE>;
  using X_Type = PythonControl::StateSpaceState_Type<_T, STATE_SIZE>;
  using Y_Type = PythonControl::StateSpaceOutput_Type<_T, OUTPUT_SIZE>;

  using U_Horizon_Type =
      PythonControl::StateSpaceInput_Type<_T, (INPUT_SIZE * NC)>;
  using X_Augmented_Type =
      PythonControl::StateSpaceState_Type<_T, (STATE_SIZE + OUTPUT_SIZE)>;
  using Y_Store_Type =
      PythonControl::DelayedVectorObject<Y_Type, NUMBER_OF_DELAY>;

  typedef typename std::conditional<
      std::is_same<SolverFactor_Type_In, SolverFactor_Empty>::value,
      PythonNumpy::DenseMatrix_Type<_T, (INPUT_SIZE * NC), (OUTPUT_SIZE * NP)>,
      SolverFactor_Type_In>::type SolverFactor_Type;

  static_assert(SolverFactor_Type::COLS == (INPUT_SIZE * NC),
                "SolverFactor_Type::COLS must be equal to (INPUT_SIZE * NC)");
  static_assert(SolverFactor_Type::ROWS == (OUTPUT_SIZE * NP),
                "SolverFactor_Type::ROWS must be equal to (OUTPUT_SIZE * "
                "NP)");

  using _MPC_StateSpace_Updater_Function_Object =
      MPC_StateSpace_Updater_Function_Object<
          Parameter_Type, typename LKF_Type::DiscreteStateSpace_Type>;

  using _LTV_MPC_Phi_F_Updater_Function_Object =
      LTV_MPC_Phi_F_Updater_Function_Object<
          typename LKF_Type::DiscreteStateSpace_Type, Parameter_Type,
          typename PredictionMatrices_Type::Phi_Type,
          typename PredictionMatrices_Type::F_Type>;

public:
  /* Constructor */
  LTV_MPC_NoConstraints() {}

protected:
  /* Variables */
  LKF_Type _kalman_filter;
  PredictionMatrices_Type _prediction_matrices;
  ReferenceTrajectory_Type _reference_trajectory;

  SolverFactor_Type _solver_factor;

  X_Type _X_inner_model;
  U_Type _U_latest;
  Y_Store_Type _Y_store;

  _MPC_StateSpace_Updater_Function_Object _state_space_updater_function;
  _LTV_MPC_Phi_F_Updater_Function_Object _phi_f_updater_function;
};

} // namespace PythonMPC

#endif // __PYTHON_LINEAR_MPC_HPP__
