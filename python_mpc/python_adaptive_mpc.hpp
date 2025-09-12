
/**
 * @file python_adaptive_mpc.hpp
 * @brief Adaptive Model Predictive Control (MPC) implementation for
 * Python-based models in C++.
 *
 * This header defines classes and utilities for adaptive MPC, including delay
 * compensation, recursive integration of control inputs, and a flexible,
 * template-based MPC controller without constraints. The implementation is
 * designed to work with Python-based state-space models and Kalman filters,
 * supporting runtime adaptation of prediction matrices and reference
 * trajectories.
 *
 * Usage:
 * - Instantiate AdaptiveMPC_NoConstraints or use make_AdaptiveMPC_NoConstraints
 * for adaptive MPC without constraints.
 * - Provide appropriate types for system, prediction, and solver components.
 * - Use update_parameters() and update_manipulation() for runtime control.
 */
#ifndef __PYTHON_ADAPTIVE_MPC_HPP__
#define __PYTHON_ADAPTIVE_MPC_HPP__

#include "mpc_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <functional>
#include <type_traits>

namespace PythonMPC {

namespace AdaptiveMPC_Operation {

/**
 * @brief Compensates the delay in the state and output vectors for the
 * Adaptive Model Predictive Control (Adaptive MPC) operation.
 * This function adjusts the state and output vectors to account for delays
 * in the system, ensuring that the control inputs are correctly aligned with
 * the measured outputs.
 * @tparam Number_Of_Delay The number of delays to compensate for.
 * @tparam X_Type Type of the state vector.
 * @tparam Y_Type Type of the output vector.
 * @tparam Y_Store_Type Type of the delayed output storage.
 * @tparam EKF_Type Type of the Kalman filter used in Adaptive MPC.
 */
template <std::size_t Number_Of_Delay, typename X_Type, typename Y_Type,
          typename Y_Store_Type, typename EKF_Type>
inline typename std::enable_if<(Number_Of_Delay > 0), void>::type
compensate_X_Y_delay(const X_Type &X_in, const Y_Type &Y_in, X_Type &X_out,
                     Y_Type &Y_out, Y_Store_Type &Y_store,
                     EKF_Type &kalman_filter) {

  static_cast<void>(X_in);

  Y_Type Y_measured = Y_in;

  X_out = kalman_filter.get_x_hat_without_delay();
  auto Y = kalman_filter.calculate_measurement_function(X_out);

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
 * @tparam Number_Of_Delay The number of delays to compensate for.
 * @tparam X_Type Type of the state vector.
 * @tparam Y_Type Type of the output vector.
 * @tparam Y_Store_Type Type of the delayed output storage (not used here).
 * @tparam EKF_Type Type of the Kalman filter (not used here).
 */
template <std::size_t Number_Of_Delay, typename X_Type, typename Y_Type,
          typename Y_Store_Type, typename EKF_Type>
inline typename std::enable_if<(Number_Of_Delay == 0), void>::type
compensate_X_Y_delay(const X_Type &X_in, const Y_Type &Y_in, X_Type &X_out,
                     Y_Type &Y_out, Y_Store_Type &Y_store,
                     EKF_Type &kalman_filter) {

  static_cast<void>(kalman_filter);

  Y_store.push(Y_in);

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

} // namespace AdaptiveMPC_Operation

/* Adaptive MPC Function Object */

template <typename X_Type, typename U_Type, typename Parameter_Type,
          typename Phi_Type, typename F_Type, typename StateSpace_Type>
using Adaptive_MPC_Phi_F_Updater_Function_Object =
    std::function<void(const X_Type &, const U_Type &, const Parameter_Type &,
                       Phi_Type &, F_Type &)>;

/* Adaptive MPC No Constraints */

/**
 * @brief Adaptive Model Predictive Control (MPC) without constraints.
 *
 * This class implements an adaptive MPC controller that can update its
 * prediction matrices and reference trajectory based on the provided
 * parameters. It supports runtime adaptation and is designed to work with
 * Python-based state-space models.
 *
 * @tparam B_Type Type of the system input matrix.
 * @tparam EKF_Type Type of the Extended Kalman Filter used for state
 * estimation.
 * @tparam PredictionMatrices_Type Type of the prediction matrices used in MPC.
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory used in
 * MPC.
 * @tparam Parameter_Type Type of the parameters used for updating the MPC.
 * @tparam SolverFactor_Type Type of the solver factor used in MPC (default is
 * empty).
 */
template <typename B_Type_In, typename EKF_Type_In,
          typename PredictionMatrices_Type_In,
          typename ReferenceTrajectory_Type_In, typename Parameter_Type_In,
          typename SolverFactor_Type_In = SolverFactor_Empty>
class AdaptiveMPC_NoConstraints {
public:
  /* Type */
  using B_Type = B_Type_In;
  using EKF_Type = EKF_Type_In;
  using PredictionMatrices_Type = PredictionMatrices_Type_In;
  using ReferenceTrajectory_Type = ReferenceTrajectory_Type_In;
  using Parameter_Type = Parameter_Type_In;

protected:
  /* Type */
  using _T = typename PredictionMatrices_Type::Value_Type;

public:
  /* Type */
  using Value_Type = _T;

  static constexpr std::size_t INPUT_SIZE = EKF_Type::INPUT_SIZE;
  static constexpr std::size_t STATE_SIZE = EKF_Type::STATE_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = EKF_Type::OUTPUT_SIZE;

  static constexpr std::size_t NP = PredictionMatrices_Type::NP;
  static constexpr std::size_t NC = PredictionMatrices_Type::NC;

  static constexpr std::size_t NUMBER_OF_DELAY = EKF_Type::NUMBER_OF_DELAY;

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

  using EmbeddedIntegratorStateSpace_Type = typename EmbeddedIntegratorTypes<
      typename EKF_Type::A_Type, B_Type,
      typename EKF_Type::C_Type>::StateSpace_Type;

  using Phi_Type = typename PredictionMatrices_Type::Phi_Type;
  using F_Type = typename PredictionMatrices_Type::F_Type;

  using Weight_U_Nc_Type = PythonNumpy::DiagMatrix_Type<_T, (INPUT_SIZE * NC)>;

protected:
  /* Type */
  using _Adaptive_MPC_Phi_F_Updater_Function_Object =
      Adaptive_MPC_Phi_F_Updater_Function_Object<
          X_Type, U_Type, Parameter_Type, Phi_Type, F_Type,
          EmbeddedIntegratorStateSpace_Type>;

  using SolverFactor_InvSolver_Left_Type =
      decltype(std::declval<PythonNumpy::Transpose_Type<Phi_Type>>() *
                   std::declval<Phi_Type>() +
               std::declval<Weight_U_Nc_Type>());

  using SolverFactor_InvSolver_Right_Type =
      PythonNumpy::Transpose_Type<Phi_Type>;

  using SolverFactor_InvSolver_Type =
      PythonNumpy::LinalgSolver_Type<SolverFactor_InvSolver_Left_Type,
                                     SolverFactor_InvSolver_Right_Type>;

public:
  /* Constructor */
  AdaptiveMPC_NoConstraints()
      : _kalman_filter(), _prediction_matrices(), _reference_trajectory(),
        _solver_factor(), _X_inner_model(), _U_latest(), _Y_store(),
        _solver_factor_inv_solver(), _Weight_U_Nc() {}

  template <typename EKF_Type, typename PredictionMatrices_Type,
            typename ReferenceTrajectory_Type,
            typename SolverFactor_Type_In_Constructor,
            typename Adaptive_MPC_Phi_F_Updater_Function_Object_In>
  AdaptiveMPC_NoConstraints(
      const EKF_Type &kalman_filter,
      const PredictionMatrices_Type &prediction_matrices,
      const ReferenceTrajectory_Type &reference_trajectory,
      const SolverFactor_Type_In_Constructor &solver_factor_in,
      const Weight_U_Nc_Type &Weight_U_Nc,
      Adaptive_MPC_Phi_F_Updater_Function_Object_In &phi_f_updater_function)
      : _kalman_filter(kalman_filter),
        _prediction_matrices(prediction_matrices),
        _reference_trajectory(reference_trajectory), _solver_factor(),
        _X_inner_model(), _U_latest(), _Y_store(), _solver_factor_inv_solver(),
        _Weight_U_Nc(Weight_U_Nc),
        _phi_f_updater_function(phi_f_updater_function) {

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

    this->_X_inner_model = this->_kalman_filter.get_x_hat();
  }

  /* Copy Constructor */
  AdaptiveMPC_NoConstraints(const AdaptiveMPC_NoConstraints<
                            B_Type_In, EKF_Type_In, PredictionMatrices_Type_In,
                            ReferenceTrajectory_Type_In, Parameter_Type_In,
                            SolverFactor_Type_In> &input)
      : _kalman_filter(input._kalman_filter),
        _prediction_matrices(input._prediction_matrices),
        _reference_trajectory(input._reference_trajectory),
        _solver_factor(input._solver_factor),
        _X_inner_model(input._X_inner_model), _U_latest(input._U_latest),
        _Y_store(input._Y_store),
        _solver_factor_inv_solver(input._solver_factor_inv_solver),
        _Weight_U_Nc(input._Weight_U_Nc),
        _phi_f_updater_function(input._phi_f_updater_function) {}

  AdaptiveMPC_NoConstraints<B_Type_In, EKF_Type_In, PredictionMatrices_Type_In,
                            ReferenceTrajectory_Type_In, Parameter_Type_In,
                            SolverFactor_Type_In> &
  operator=(const AdaptiveMPC_NoConstraints<
            B_Type_In, EKF_Type_In, PredictionMatrices_Type_In,
            ReferenceTrajectory_Type_In, Parameter_Type_In,
            SolverFactor_Type_In> &input) {
    if (this != &input) {
      this->_kalman_filter = input._kalman_filter;
      this->_prediction_matrices = input._prediction_matrices;
      this->_reference_trajectory = input._reference_trajectory;
      this->_solver_factor = input._solver_factor;
      this->_X_inner_model = input._X_inner_model;
      this->_U_latest = input._U_latest;
      this->_Y_store = input._Y_store;
      this->_solver_factor_inv_solver = input._solver_factor_inv_solver;
      this->_Weight_U_Nc = input._Weight_U_Nc;
      this->_phi_f_updater_function = input._phi_f_updater_function;
    }
    return *this;
  }

  /* Move Constructor */
  AdaptiveMPC_NoConstraints(AdaptiveMPC_NoConstraints<
                            B_Type_In, EKF_Type_In, PredictionMatrices_Type_In,
                            ReferenceTrajectory_Type_In, Parameter_Type_In,
                            SolverFactor_Type_In> &&input) noexcept
      : _kalman_filter(std::move(input._kalman_filter)),
        _prediction_matrices(std::move(input._prediction_matrices)),
        _reference_trajectory(std::move(input._reference_trajectory)),
        _solver_factor(std::move(input._solver_factor)),
        _X_inner_model(std::move(input._X_inner_model)),
        _U_latest(std::move(input._U_latest)),
        _Y_store(std::move(input._Y_store)),
        _solver_factor_inv_solver(std::move(input._solver_factor_inv_solver)),
        _Weight_U_Nc(std::move(input._Weight_U_Nc)),
        _phi_f_updater_function(std::move(input._phi_f_updater_function)) {}

  AdaptiveMPC_NoConstraints<B_Type_In, EKF_Type_In, PredictionMatrices_Type_In,
                            ReferenceTrajectory_Type_In, Parameter_Type_In,
                            SolverFactor_Type_In> &
  operator=(AdaptiveMPC_NoConstraints<
            B_Type_In, EKF_Type_In, PredictionMatrices_Type_In,
            ReferenceTrajectory_Type_In, Parameter_Type_In,
            SolverFactor_Type_In> &&input) noexcept {
    if (this != &input) {
      this->_kalman_filter = std::move(input._kalman_filter);
      this->_prediction_matrices = std::move(input._prediction_matrices);
      this->_reference_trajectory = std::move(input._reference_trajectory);
      this->_solver_factor = std::move(input._solver_factor);
      this->_X_inner_model = std::move(input._X_inner_model);
      this->_U_latest = std::move(input._U_latest);
      this->_Y_store = std::move(input._Y_store);
      this->_solver_factor_inv_solver =
          std::move(input._solver_factor_inv_solver);
      this->_Weight_U_Nc = std::move(input._Weight_U_Nc);
      this->_phi_f_updater_function = std::move(input._phi_f_updater_function);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Updates the solver factor based on the current Phi and weight
   * matrix for control input changes.
   *
   * This function recalculates the solver factor used in the MPC based on the
   * provided Phi matrix and weight matrix for control input changes, which is
   * essential for scaling the control inputs.
   *
   * @param Phi The current Phi matrix used in the MPC.
   * @param Weight_U_Nc The weight matrix for control input changes.
   */
  inline void update_solver_factor(const Phi_Type &Phi,
                                   const Weight_U_Nc_Type &Weight_U_Nc) {

    // So far, "np.linalg.solve(Phi.T @ Phi + Weight_U_Nc, Phi.T)" is used.
    // QR decomposition is preferred for better numerical stability, but it
    // costs much memory footprint and calculation time.

    auto Phi_T_Phi_W = PythonNumpy::ATranspose_mul_B(Phi, Phi) + Weight_U_Nc;

    PythonNumpy::substitute_matrix(
        this->_solver_factor,
        this->_solver_factor_inv_solver.solve(Phi_T_Phi_W, Phi.transpose()));
  }

  /**
   * @brief Updates the parameters of the Kalman filter used in the Adaptive
   * MPC.
   *
   * This function allows for runtime updates of the Kalman filter parameters,
   * which can affect the state estimation and control input calculations.
   *
   * @param parameters The new parameters to be set in the Kalman filter.
   */
  template <typename Parameter_Type>
  inline void update_parameters(const Parameter_Type &parameters) {

    this->_kalman_filter.parameters = parameters;
  }

  /**
   * @brief Updates the reference trajectory for the Adaptive MPC.
   *
   * This function sets a new reference trajectory based on the provided
   * reference vector, which is used to calculate control inputs.
   *
   * @tparam Ref_Type Type of the reference vector.
   * @param ref The reference vector to be set.
   */
  template <typename Ref_Type>
  inline auto update_manipulation(const Ref_Type &reference, const Y_Type &Y)
      -> U_Type {

    this->_kalman_filter.predict_and_update(this->_U_latest, Y);

    X_Type X = this->_kalman_filter.get_x_hat();

    X_Type X_compensated;
    Y_Type Y_compensated;
    this->_compensate_X_Y_delay(X, Y, X_compensated, Y_compensated);

    this->_update_Phi_F_adaptive_runtime(X_compensated, this->_U_latest,
                                         this->_kalman_filter.parameters);

    this->update_solver_factor(this->_prediction_matrices.Phi,
                               this->_Weight_U_Nc);

    auto delta_X = X_compensated - this->_X_inner_model;
    auto delta_Y = Y_compensated - this->_Y_store.get();
    auto X_augmented = PythonNumpy::concatenate_vertically(delta_X, delta_Y);

    this->_reference_trajectory.set_reference_sub_Y(reference,
                                                    this->_Y_store.get());

    auto delta_U = this->_solve(X_augmented);

    AdaptiveMPC_Operation::Integrate_U<
        U_Type, U_Horizon_Type, (INPUT_SIZE - 1)>::calculate(this->_U_latest,
                                                             delta_U);

    this->_X_inner_model = X_compensated;

    return this->_U_latest;
  }

  /**
   * @brief Returns the prediction matrices used in the Adaptive MPC.
   *
   * This function provides access to the prediction matrices, which are used
   * to compute the control inputs based on the current state and reference
   * trajectory.
   *
   * @return The prediction matrices of type PredictionMatrices_Type.
   */
  inline auto get_F(void) const -> F_Type {
    return this->_prediction_matrices.F;
  }

  /**
   * @brief Returns the Phi matrix used in the Adaptive MPC.
   *
   * This function provides access to the Phi matrix, which is part of the
   * prediction matrices and is used in the control input calculations.
   *
   * @return The Phi matrix of type Phi_Type.
   */
  inline auto get_Phi(void) const -> Phi_Type {
    return this->_prediction_matrices.Phi;
  }

  /**
   * @brief Returns the solver factor used in the Adaptive MPC.
   *
   * This function provides access to the solver factor, which is used to
   * scale the control inputs based on the reference trajectory and prediction
   * matrices.
   *
   * @return The solver factor of type SolverFactor_Type.
   */
  inline auto get_solver_factor(void) const -> SolverFactor_Type {
    return this->_solver_factor;
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

    AdaptiveMPC_Operation::compensate_X_Y_delay<NUMBER_OF_DELAY>(
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

    return this->_solver_factor *
           this->_reference_trajectory.calculate_dif(
               this->_prediction_matrices.F * X_augmented);
  }

  /**
   * @brief Updates the Phi and F matrices for adaptive runtime.
   *
   * This function allows for runtime updates of the Phi and F matrices based
   * on the current state, control input, and parameters, enabling adaptive
   * behavior in the MPC.
   *
   * @param X The current state vector.
   * @param U The current control input vector.
   * @param parameters The parameters used for updating the matrices.
   */
  inline void _update_Phi_F_adaptive_runtime(const X_Type &X, const U_Type &U,
                                             const Parameter_Type &parameters) {

    this->_phi_f_updater_function(X, U, parameters,
                                  this->_prediction_matrices.Phi,
                                  this->_prediction_matrices.F);
  }

protected:
  /* Variables */
  EKF_Type _kalman_filter;
  PredictionMatrices_Type _prediction_matrices;
  ReferenceTrajectory_Type _reference_trajectory;

  SolverFactor_Type _solver_factor;

  X_Type _X_inner_model;
  U_Type _U_latest;
  Y_Store_Type _Y_store;

  SolverFactor_InvSolver_Type _solver_factor_inv_solver;
  Weight_U_Nc_Type _Weight_U_Nc;

  _Adaptive_MPC_Phi_F_Updater_Function_Object _phi_f_updater_function;
};

/* make Adaptive MPC No Constraints */

/**
 * @brief Factory function to create an instance of AdaptiveMPC_NoConstraints.
 *
 * This function initializes the AdaptiveMPC_NoConstraints class with the
 * provided Kalman filter, prediction matrices, reference trajectory, solver
 * factor, weight matrix for control input changes, and Phi/F updater function.
 *
 * @tparam B_Type Type of the system input matrix.
 * @tparam EKF_Type Type of the Extended Kalman Filter used for state
 * estimation.
 * @tparam PredictionMatrices_Type Type of the prediction matrices used in MPC.
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory used in
 * MPC.
 * @tparam Parameter_Type Type of the parameters used for updating the MPC.
 * @tparam SolverFactor_Type_In Type of the solver factor used in MPC (default
 * is empty).
 */
template <typename B_Type, typename EKF_Type, typename PredictionMatrices_Type,
          typename ReferenceTrajectory_Type, typename Parameter_Type,
          typename SolverFactor_Type_In, typename Weight_U_Nc_Type,
          typename X_Type, typename U_Type,
          typename EmbeddedIntegratorStateSpace_Type>
inline auto make_AdaptiveMPC_NoConstraints(
    const EKF_Type &kalman_filter,
    const PredictionMatrices_Type &prediction_matrices,
    const ReferenceTrajectory_Type &reference_trajectory,
    const SolverFactor_Type_In &solver_factor_in,
    const Weight_U_Nc_Type &Weight_U_Nc,
    Adaptive_MPC_Phi_F_Updater_Function_Object<
        X_Type, U_Type, Parameter_Type,
        typename PredictionMatrices_Type::Phi_Type,
        typename PredictionMatrices_Type::F_Type,
        EmbeddedIntegratorStateSpace_Type> &phi_f_updater_function)
    -> AdaptiveMPC_NoConstraints<B_Type, EKF_Type, PredictionMatrices_Type,
                                 ReferenceTrajectory_Type, Parameter_Type,
                                 SolverFactor_Type_In> {

  return AdaptiveMPC_NoConstraints<B_Type, EKF_Type, PredictionMatrices_Type,
                                   ReferenceTrajectory_Type, Parameter_Type,
                                   SolverFactor_Type_In>(
      kalman_filter, prediction_matrices, reference_trajectory,
      solver_factor_in, Weight_U_Nc, phi_f_updater_function);
}

/* Adaptive MPC No Constraints Type */
template <typename B_Type, typename EKF_Type, typename PredictionMatrices_Type,
          typename ReferenceTrajectory_Type, typename Parameter_Type,
          typename SolverFactor_Type_In = SolverFactor_Empty>
using AdaptiveMPC_NoConstraints_Type =
    AdaptiveMPC_NoConstraints<B_Type, EKF_Type, PredictionMatrices_Type,
                              ReferenceTrajectory_Type, Parameter_Type,
                              SolverFactor_Type_In>;

/* Adaptive MPC */

/**
 * @brief Adaptive Model Predictive Control (MPC) class with constraints.
 *
 * This class extends AdaptiveMPC_NoConstraints to support constraints on
 * delta-U, U, and Y via a QP solver. The Phi/F matrices are updated at
 * runtime using the provided updater function, similar to the no-constraints
 * version, but optimization is solved with LMPC_QP_Solver.
 *
 * @tparam B_Type Type of the system input matrix B.
 * @tparam EKF_Type Type of the EKF used for state estimation.
 * @tparam PredictionMatrices_Type Type holding Phi and F.
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory.
 * @tparam Parameter_Type Type of adaptive parameters.
 * @tparam Delta_U_Min_Type Type for min delta-U constraints.
 * @tparam Delta_U_Max_Type Type for max delta-U constraints.
 * @tparam U_Min_Type Type for min U constraints.
 * @tparam U_Max_Type Type for max U constraints.
 * @tparam Y_Min_Type Type for min Y constraints.
 * @tparam Y_Max_Type Type for max Y constraints.
 * @tparam SolverFactor_Type_In Optional solver-factor type.
 */
template <typename B_Type, typename EKF_Type, typename PredictionMatrices_Type,
          typename ReferenceTrajectory_Type, typename Parameter_Type,
          typename Delta_U_Min_Type, typename Delta_U_Max_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type,
          typename SolverFactor_Type_In = SolverFactor_Empty>
class AdaptiveMPC
    : public AdaptiveMPC_NoConstraints<
          B_Type, EKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,
          Parameter_Type, SolverFactor_Type_In> {

protected:
  using _AdaptiveMPC_NoConstraints_Type =
      AdaptiveMPC_NoConstraints<B_Type, EKF_Type, PredictionMatrices_Type,
                                ReferenceTrajectory_Type, Parameter_Type,
                                SolverFactor_Type_In>;

  using _U_Horizon_Type =
      typename _AdaptiveMPC_NoConstraints_Type::U_Horizon_Type;
  using _X_Augmented_Type =
      typename _AdaptiveMPC_NoConstraints_Type::X_Augmented_Type;
  using _Weight_U_Nc_Type =
      typename _AdaptiveMPC_NoConstraints_Type::Weight_U_Nc_Type;

  using _Solver_Type = LMPC_QP_Solver_Type<
      _U_Horizon_Type::COLS, _AdaptiveMPC_NoConstraints_Type::OUTPUT_SIZE,
      typename _AdaptiveMPC_NoConstraints_Type::U_Type, _X_Augmented_Type,
      typename PredictionMatrices_Type::Phi_Type,
      typename PredictionMatrices_Type::F_Type, _Weight_U_Nc_Type,
      Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type, U_Max_Type, Y_Min_Type,
      Y_Max_Type>;

  using _Adaptive_MPC_Phi_F_Updater_Function_Object =
      typename _AdaptiveMPC_NoConstraints_Type::
          _Adaptive_MPC_Phi_F_Updater_Function_Object;

public:
  /* Constructor */
  AdaptiveMPC() : _AdaptiveMPC_NoConstraints_Type(), _solver() {}

  template <typename SolverFactor_Type>
  AdaptiveMPC(
      const EKF_Type &kalman_filter,
      const PredictionMatrices_Type &prediction_matrices,
      const ReferenceTrajectory_Type &reference_trajectory,
      const _Weight_U_Nc_Type &Weight_U_Nc,
      _Adaptive_MPC_Phi_F_Updater_Function_Object &phi_f_updater_function,
      const Delta_U_Min_Type &delta_U_min, const Delta_U_Max_Type &delta_U_max,
      const U_Min_Type &U_min, const U_Max_Type &U_max, const Y_Min_Type &Y_min,
      const Y_Max_Type &Y_max, const SolverFactor_Type &solver_factor_in)
      : _AdaptiveMPC_NoConstraints_Type(kalman_filter, prediction_matrices,
                                        reference_trajectory, solver_factor_in,
                                        Weight_U_Nc, phi_f_updater_function),
        _solver() {

    auto X_augmented = PythonNumpy::concatenate_vertically(
        this->_X_inner_model, this->_Y_store.get());

    this->_solver =
        make_LMPC_QP_Solver<_U_Horizon_Type::COLS,
                            _AdaptiveMPC_NoConstraints_Type::OUTPUT_SIZE>(
            this->_U_latest, X_augmented, this->_prediction_matrices.Phi,
            this->_prediction_matrices.F, Weight_U_Nc, delta_U_min, delta_U_max,
            U_min, U_max, Y_min, Y_max);
  }

  /* Copy Constructor */
  AdaptiveMPC(const AdaptiveMPC &other)
      : _AdaptiveMPC_NoConstraints_Type(other), _solver(other._solver) {}

  AdaptiveMPC &operator=(const AdaptiveMPC &other) {
    if (this != &other) {
      this->_AdaptiveMPC_NoConstraints_Type::operator=(other);
      this->_solver = other._solver;
    }
    return *this;
  }

  /* Move Constructor */
  AdaptiveMPC(AdaptiveMPC &&other) noexcept
      : _AdaptiveMPC_NoConstraints_Type(std::move(other)),
        _solver(std::move(other._solver)) {}

  AdaptiveMPC &operator=(AdaptiveMPC &&other) noexcept {
    if (this != &other) {
      this->_AdaptiveMPC_NoConstraints_Type::operator=(std::move(other));
      this->_solver = std::move(other._solver);
    }
    return *this;
  }

protected:
  /* Function */

  /**
   * @brief Solves the adaptive MPC optimization problem for the given augmented
   * state.
   *
   * This method updates the solver's constraints based on the latest control
   * input, the current augmented state, and the prediction matrices. It then
   * solves the optimization problem to compute the optimal change in control
   * input (delta_U) over the prediction horizon, given the reference trajectory
   * and current state.
   *
   * @param X_augmented The current augmented state vector.
   * @return _U_Horizon_Type The computed optimal change in control input over
   * the horizon.
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

/* make Adaptive MPC */

/**
 * @brief Factory function to create an instance of AdaptiveMPC.
 *
 * This function initializes the AdaptiveMPC class with the provided Kalman
 * filter, prediction matrices, reference trajectory, weight matrix for control
 * input changes, Phi/F updater function, constraints, and solver factor.
 *
 * @tparam B_Type Type of the system input matrix.
 * @tparam EKF_Type Type of the Extended Kalman Filter used for state
 * estimation.
 * @tparam PredictionMatrices_Type Type of the prediction matrices used in MPC.
 * @tparam ReferenceTrajectory_Type Type of the reference trajectory used in
 * MPC.
 * @tparam Parameter_Type Type of the parameters used for updating the MPC.
 * @tparam Delta_U_Min_Type Type for min delta-U constraints.
 * @tparam Delta_U_Max_Type Type for max delta-U constraints.
 * @tparam U_Min_Type Type for min U constraints.
 * @tparam U_Max_Type Type for max U constraints.
 * @tparam Y_Min_Type Type for min Y constraints.
 * @tparam Y_Max_Type Type for max Y constraints.
 * @tparam SolverFactor_Type_In Type of the solver factor used in MPC (default
 * is empty).
 */
template <typename B_Type, typename EKF_Type, typename PredictionMatrices_Type,
          typename ReferenceTrajectory_Type, typename Parameter_Type,
          typename Delta_U_Min_Type, typename Delta_U_Max_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type, typename SolverFactor_Type_In,
          typename Weight_U_Nc_Type, typename X_Type, typename U_Type,
          typename EmbeddedIntegratorStateSpace_Type>
inline auto make_AdaptiveMPC(
    const EKF_Type &kalman_filter,
    const PredictionMatrices_Type &prediction_matrices,
    const ReferenceTrajectory_Type &reference_trajectory,
    const Weight_U_Nc_Type &Weight_U_Nc,
    Adaptive_MPC_Phi_F_Updater_Function_Object<
        X_Type, U_Type, Parameter_Type,
        typename PredictionMatrices_Type::Phi_Type,
        typename PredictionMatrices_Type::F_Type,
        EmbeddedIntegratorStateSpace_Type> &phi_f_updater_function,
    const Delta_U_Min_Type &delta_U_min, const Delta_U_Max_Type &delta_U_max,
    const U_Min_Type &U_min, const U_Max_Type &U_max, const Y_Min_Type &Y_min,
    const Y_Max_Type &Y_max, const SolverFactor_Type_In &solver_factor_in)
    -> AdaptiveMPC<B_Type, EKF_Type, PredictionMatrices_Type,
                   ReferenceTrajectory_Type, Parameter_Type, Delta_U_Min_Type,
                   Delta_U_Max_Type, U_Min_Type, U_Max_Type, Y_Min_Type,
                   Y_Max_Type, SolverFactor_Type_In> {

  return AdaptiveMPC<B_Type, EKF_Type, PredictionMatrices_Type,
                     ReferenceTrajectory_Type, Parameter_Type, Delta_U_Min_Type,
                     Delta_U_Max_Type, U_Min_Type, U_Max_Type, Y_Min_Type,
                     Y_Max_Type, SolverFactor_Type_In>(
      kalman_filter, prediction_matrices, reference_trajectory, Weight_U_Nc,
      phi_f_updater_function, delta_U_min, delta_U_max, U_min, U_max, Y_min,
      Y_max, solver_factor_in);
}

/* Adaptive MPC Type */
template <typename B_Type, typename EKF_Type, typename PredictionMatrices_Type,
          typename ReferenceTrajectory_Type, typename Parameter_Type,
          typename Delta_U_Min_Type, typename Delta_U_Max_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type,
          typename SolverFactor_Type_In = SolverFactor_Empty>
using AdaptiveMPC_Type =
    AdaptiveMPC<B_Type, EKF_Type, PredictionMatrices_Type,
                ReferenceTrajectory_Type, Parameter_Type, Delta_U_Min_Type,
                Delta_U_Max_Type, U_Min_Type, U_Max_Type, Y_Min_Type,
                Y_Max_Type, SolverFactor_Type_In>;

} // namespace PythonMPC

#endif // __PYTHON_ADAPTIVE_MPC_HPP__
