#ifndef __PYTHON_ADAPTIVE_MPC_HPP__
#define __PYTHON_ADAPTIVE_MPC_HPP__

#include "mpc_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <functional>
#include <type_traits>

namespace PythonMPC {

/* Adaptive MPC Function Object */

template <typename X_Type, typename U_Type, typename Parameter_Type,
          typename Phi_Type, typename F_Type, typename StateSpace_Type>
using Adaptive_MPC_Phi_F_Updater_Function_Object =
    std::function<void(const X_Type &, const U_Type &, const Parameter_Type &,
                       Phi_Type &, F_Type &)>;

/* Adaptive MPC No Constraints */

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
  template <typename Parameter_Type>
  inline void update_parameters(const Parameter_Type &parameters) {

    this->_kalman_filter.parameters = parameters;
  }

  template <typename Ref_Type>
  inline auto update_manipulation(const Ref_Type &reference, const Y_Type &Y)
      -> U_Type {

    this->_kalman_filter.predict_and_update(this->_U_latest, Y);

    X_Type X = this->_kalman_filter.get_x_hat();

    X_Type X_compensated;
    Y_Type Y_compensated;
    this->_compensate_X_Y_delay(X, Y, X_compensated, Y_compensated);

    this->_prediction_matrices.update_Phi_F_adaptive_runtime(
        this->_kalman_filter.Parameters, X_compensated, this->_U_latest);

    auto delta_X = X_compensated - this->_X_inner_model;
    auto delta_Y = Y_compensated - this->_Y_store.get();
    auto X_augmented = PythonNumpy::concatenate_vertically(delta_X, delta_Y);

    this->_reference_trajectory.set_reference_sub_Y(reference,
                                                    this->_Y_store.get());

    auto delta_U = this->_solve(X_augmented);

    LMPC_Operation::Integrate_U<U_Type, U_Horizon_Type,
                                (INPUT_SIZE - 1)>::calculate(this->_U_latest,
                                                             delta_U);

    this->_X_inner_model = X_compensated;

    return this->_U_latest;
  }

  inline auto get_F(void) const -> F_Type {
    return this->_prediction_matrices.F;
  }

  inline auto get_Phi(void) const -> Phi_Type {
    return this->_prediction_matrices.Phi;
  }

  inline auto get_solver_factor(void) const -> SolverFactor_Type {
    return this->_solver_factor;
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

/* make LTV MPC No Constraints */
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

} // namespace PythonMPC

#endif // __PYTHON_ADAPTIVE_MPC_HPP__
