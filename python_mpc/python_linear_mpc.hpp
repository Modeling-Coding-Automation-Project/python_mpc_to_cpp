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
  static void calculate(U_Type &U, const U_Horizon_Type &delta_U_Horizon) {

    U.template set<Index, 0>(U.template get<Index, 0>() +
                             delta_U_Horizon.template get<Index, 0>());
    Integrate_U<U_Type, U_Horizon_Type, Index - 1>::calculate(U,
                                                              delta_U_Horizon);
  }
};

template <typename U_Type, typename U_Horizon_Type>
struct Integrate_U<U_Type, U_Horizon_Type, 0> {
  static void calculate(U_Type &U, const U_Horizon_Type &delta_U_Horizon) {

    U.template set<0, 0>(U.template get<0, 0>() +
                         delta_U_Horizon.template get<0, 0>());
  }
};

} // namespace LMPC_Operation

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
  template <typename Ref_Type>
  inline void set_reference_trajectory(const Ref_Type &ref) {

    static_assert(std::is_same<typename Ref_Type::Value_Type, _T>::value,
                  "Ref_Type::Value_Type must be equal to Value_Type");

    this->_reference_trajectory.reference_vector = ref;
  }

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
  inline void _compensate_X_Y_delay(const X_Type &X_in, const Y_Type &Y_in,
                                    X_Type &X_out, Y_Type &Y_out) {

    LMPC_Operation::compensate_X_Y_delay<NUMBER_OF_DELAY>(
        X_in, Y_in, X_out, Y_out, this->_Y_store, this->_kalman_filter);
  }

  inline auto _solve(const X_Augmented_Type &X_augmented) -> U_Horizon_Type {

    auto delta_U =
        this->_solver_factor * this->_reference_trajectory.calculate_dif(
                                   this->_prediction_matrices.F * X_augmented);

    return delta_U;
  }

  inline auto _calculate_this_U(const U_Type &delta_U) -> U_Type {

    auto U = this->_U_latest + delta_U;

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
  LTI_MPC(LTI_MPC &&other) noexcept
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

public:
  /* Function */
  inline auto solve(const ReferenceTrajectory_Type &reference_trajectory,
                    const _X_Augmented_Type &X_augmented) -> _U_Horizon_Type {

    this->_solver.update_constraints(this->_U_latest, X_augmented,
                                     this->_prediction_matrices.Phi,
                                     this->_prediction_matrices.F);

    auto delta_U = this->_solver.solve(this->_prediction_matrices.Phi,
                                       this->_prediction_matrices.F,
                                       reference_trajectory, X_augmented);

    return delta_U;
  }

protected:
  /* Variables */
  _Solver_Type _solver;
};

} // namespace PythonMPC

#endif // __PYTHON_LINEAR_MPC_HPP__
