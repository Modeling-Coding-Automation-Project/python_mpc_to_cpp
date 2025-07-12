/**
 * @file mpc_state_space_utility.hpp
 * @brief Utility classes and functions for Model Predictive Control (MPC)
 * state-space prediction and reference trajectory operations.
 *
 * This header provides a set of template classes and utility functions to
 * support the construction and manipulation of prediction matrices and
 * reference trajectories for Model Predictive Control (MPC) applications. The
 * utilities are designed to be generic and type-safe, supporting compile-time
 * checks for matrix dimensions and value types.
 */
#ifndef __MPC_STATE_SPACE_UTILITY_HPP__
#define __MPC_STATE_SPACE_UTILITY_HPP__

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <functional>
#include <type_traits>
#include <utility>

namespace PythonMPC {

namespace EmbeddedIntegratorOperation {

template <typename A_Type_In, typename B_Type_In, typename C_Type_In>
struct EmbeddedIntegratorTypes {

  using T = typename A_Type_In::Value_Type;

  // constexpr std::size_t B_COLS = B_Type_In::COLS + C_Type_In::COLS;
  // constexpr std::size_t B_ROWS = B_Type_In::ROWS;

  // constexpr std::size_t C_COLS = C_Type_In::COLS;
  // constexpr std::size_t C_ROWS = C_Type_In::ROWS + C_Type_In::COLS;

  using O_M_T_Type =
      PythonNumpy::SparseMatrixEmpty_Type<T, A_Type_In::COLS, C_Type_In::COLS>;

  using O_M_Type =
      PythonNumpy::SparseMatrixEmpty_Type<T, C_Type_In::COLS, A_Type_In::COLS>;

  using CC_Identity_Type = PythonNumpy::DiagMatrix_Type<T, C_Type_In::COLS>;

  using C_A_Type =
      decltype(std::declval<C_Type_In>() * std::declval<A_Type_In>());

  using C_B_Type =
      decltype(std::declval<C_Type_In>() * std::declval<B_Type_In>());

  using A_Type = PythonNumpy::ConcatenateBlock_Type<
      (A_Type_In::COLS + C_Type_In::COLS), (A_Type_In::COLS + C_Type_In::COLS)>(
      A_Type_In, O_M_T_Type, C_A_Type, CC_Identity_Type);

  using B_Type =
      PythonNumpy::ConcatenateBlock_Type<(B_Type_In::COLS + C_Type_In::COLS),
                                         B_Type_In::ROWS>(B_Type_In, C_B_Type);

  using C_Type = PythonNumpy::ConcatenateBlock_Type<
      C_Type_In::COLS, (C_Type_In::ROWS + C_Type_In::COLS)>(O_M_Type,
                                                            CC_Identity_Type);
};

} // namespace EmbeddedIntegratorOperation

/* MPC Prediction Matrices */

/**
 * @brief Class template for MPC Prediction Matrices.
 *
 * This class template encapsulates the prediction matrices used in Model
 * Predictive Control (MPC) for state-space systems. It includes the system
 * dynamics matrix F and the prediction matrix Phi, which are essential for
 * predicting future states and outputs based on current inputs and states.
 *
 * @tparam F_Type_In Type of the system dynamics matrix F.
 * @tparam Phi_Type_In Type of the prediction matrix Phi.
 * @tparam Np Number of prediction steps.
 * @tparam Nc Number of control steps.
 * @tparam Number_Of_Input Number of inputs to the system.
 * @tparam Number_Of_State Number of states in the system.
 * @tparam Number_Of_Output Number of outputs from the system.
 */
template <typename F_Type_In, typename Phi_Type_In, std::size_t Np,
          std::size_t Nc, std::size_t Number_Of_Input,
          std::size_t Number_Of_State, std::size_t Number_Of_Output>
class MPC_PredictionMatrices {
protected:
  /* Type */
  using _T = typename F_Type_In::Value_Type;

public:
  /* Type */
  using Value_Type = _T;

  using F_Type = F_Type_In;
  using Phi_Type = Phi_Type_In;

  static constexpr std::size_t INPUT_SIZE = Number_Of_Input;
  static constexpr std::size_t STATE_SIZE = Number_Of_State;
  static constexpr std::size_t OUTPUT_SIZE = Number_Of_Output;

  /* Check Compatibility */
  static_assert(std::is_same<typename Phi_Type::Value_Type, _T>::value,
                "F_Type and Phi_Type must have the same Value_Type");

  static_assert(F_Type::COLS == OUTPUT_SIZE * Np,
                "F_Type::COLS must be equal to OUTPUT_SIZE * Np");
  static_assert(F_Type::ROWS == STATE_SIZE,
                "F_Type::ROWS must be equal to STATE_SIZE");
  static_assert(Phi_Type::COLS == OUTPUT_SIZE * Np,
                "Phi_Type::ROWS must be equal to OUTPUT_SIZE * Np");
  static_assert(Phi_Type::ROWS == INPUT_SIZE * Nc,
                "Phi_Type::ROWS must be equal to INPUT_SIZE * Nc");

public:
  /* Constructor */
  MPC_PredictionMatrices() : F(), Phi() {}

  MPC_PredictionMatrices(const F_Type &F, const Phi_Type &Phi)
      : F(F), Phi(Phi) {}

  /* Copy Constructor */
  MPC_PredictionMatrices(
      const MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                                   Number_Of_State, Number_Of_Output> &input)
      : F(input.F), Phi(input.Phi) {}

  MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                         Number_Of_State, Number_Of_Output> &
  operator=(
      const MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                                   Number_Of_State, Number_Of_Output> &input) {
    if (this != &input) {
      this->F = input.F;
      this->Phi = input.Phi;
    }
    return *this;
  }

  /* Move Constructor */
  MPC_PredictionMatrices(
      MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                             Number_Of_State, Number_Of_Output>
          &&input) noexcept
      : F(std::move(input.F)), Phi(std::move(input.Phi)) {}

  MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                         Number_Of_State, Number_Of_Output> &
  operator=(MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                                   Number_Of_State, Number_Of_Output>
                &&input) noexcept {
    if (this != &input) {
      this->F = std::move(input.F);
      this->Phi = std::move(input.Phi);
    }
    return *this;
  }

public:
  /* Constant */
  static constexpr std::size_t NP = Np;
  static constexpr std::size_t NC = Nc;

public:
  /* Variables */
  F_Type F;
  Phi_Type Phi;
};

/* make MPC Prediction Matrices */

/**
 * @brief Factory function to create an instance of MPC_PredictionMatrices.
 *
 * This function initializes and returns an instance of MPC_PredictionMatrices
 * with the provided template parameters.
 *
 * @tparam F_Type Type of the system dynamics matrix F.
 * @tparam Phi_Type Type of the prediction matrix Phi.
 * @tparam Np Number of prediction steps.
 * @tparam Nc Number of control steps.
 * @tparam Number_Of_Input Number of inputs to the system.
 * @tparam Number_Of_State Number of states in the system.
 * @tparam Number_Of_Output Number of outputs from the system.
 * @return An instance of MPC_PredictionMatrices initialized with default
 * values.
 */
template <typename F_Type, typename Phi_Type, std::size_t Np, std::size_t Nc,
          std::size_t Number_Of_Input, std::size_t Number_Of_State,
          std::size_t Number_Of_Output>
inline auto make_MPC_PredictionMatrices(void)
    -> MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                              Number_Of_State, Number_Of_Output> {

  return MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                                Number_Of_State, Number_Of_Output>();
}

/* MPC Prediction Matrices Type */
template <typename F_Type, typename Phi_Type, std::size_t Np, std::size_t Nc,
          std::size_t Number_Of_Input, std::size_t Number_Of_State,
          std::size_t Number_Of_Output>
using MPC_PredictionMatrices_Type =
    MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                           Number_Of_State, Number_Of_Output>;

namespace MPC_ReferenceTrajectoryOperation {

/**
 * @brief Calculates the difference between a reference value and a predicted
 * value for a specific index, and stores the result in the provided difference
 * container.
 *
 * This function is enabled only when ROWS > 1, and asserts at compile time that
 * ROWS == Np. It computes the difference between the reference value at
 * position (J, I) and the predicted value at the corresponding flattened index,
 * then sets this value in the 'dif' container.
 *
 * @tparam ROWS Number of rows in the reference and prediction matrices (must be
 * > 1).
 * @tparam Np Prediction horizon length (must be equal to ROWS).
 * @tparam I Row index for the operation.
 * @tparam J Column index for the operation.
 * @tparam Ref_Type Type of the reference matrix.
 * @tparam Fx_Type Type of the predicted matrix.
 * @tparam Dif_Type Type of the difference container.
 * @param ref Reference matrix.
 * @param Fx Predicted matrix.
 * @param dif Container to store the computed difference.
 */
template <std::size_t ROWS, std::size_t Np, std::size_t I, std::size_t J,
          typename Ref_Type, typename Fx_Type, typename Dif_Type>
inline typename std::enable_if<(ROWS > 1), void>::type
calculate_each_dif(const Ref_Type &ref, const Fx_Type &Fx, Dif_Type &dif) {
  static_assert(ROWS == Np, "ROWS must be equal to Np when ROWS > 1");

  dif.template set<(I * Ref_Type::COLS) + J, 0>(
      ref.template get<J, I>() -
      Fx.template get<(I * Ref_Type::COLS) + J, 0>());
}

/**
 * @brief Calculates the difference between a reference value and a predicted
 * value for a specific index, and stores the result in the provided difference
 * container.
 *
 * This function is enabled only when ROWS == 1. It computes the difference
 * between the reference value at position (J, 0) and the predicted value at the
 * corresponding flattened index, then sets this value in the 'dif' container.
 *
 * @tparam ROWS Number of rows in the reference matrix (must be equal to 1).
 * @tparam Np Prediction horizon length (must be equal to 1).
 * @tparam I Row index for the operation.
 * @tparam J Column index for the operation.
 * @tparam Ref_Type Type of the reference matrix.
 * @tparam Fx_Type Type of the predicted matrix.
 * @tparam Dif_Type Type of the difference container.
 * @param ref Reference matrix.
 * @param Fx Predicted matrix.
 * @param dif Container to store the computed difference.
 */
template <std::size_t ROWS, std::size_t Np, std::size_t I, std::size_t J,
          typename Ref_Type, typename Fx_Type, typename Dif_Type>
inline typename std::enable_if<(ROWS == 1), void>::type
calculate_each_dif(const Ref_Type &ref, const Fx_Type &Fx, Dif_Type &dif) {
  static_assert(ROWS == 1, "ROWS must be equal to 1");

  dif.template set<(I * Ref_Type::COLS) + J, 0>(
      ref.template get<J, 0>() -
      Fx.template get<(I * Ref_Type::COLS) + J, 0>());
}

// when J_idx < N
template <typename Ref_Type, typename Fx_Type, typename Dif_Type,
          std::size_t Np, std::size_t I, std::size_t J_idx>
struct DifColumn {
  /**
   * @brief Calculates the difference for a specific column index and calls the
   * next column recursively.
   *
   * This function calculates the difference for the specified column index
   * J_idx and then recursively calls itself to calculate differences for the
   * next column index (J_idx - 1).
   *
   * @tparam Ref_Type Type of the reference matrix.
   * @tparam Fx_Type Type of the predicted matrix.
   * @tparam Dif_Type Type of the difference container.
   * @tparam Np Prediction horizon length.
   * @tparam I Row index for the operation.
   * @tparam J_idx Current column index for the operation.
   * @param ref Reference matrix.
   * @param Fx Predicted matrix.
   * @param dif Container to store the computed differences.
   */
  static void calculate(const Ref_Type &ref, const Fx_Type &Fx, Dif_Type &dif) {

    calculate_each_dif<Ref_Type::ROWS, Np, I, J_idx>(ref, Fx, dif);

    DifColumn<Ref_Type, Fx_Type, Dif_Type, Np, I, J_idx - 1>::calculate(ref, Fx,
                                                                        dif);
  }
};

// column recursion termination
template <typename Ref_Type, typename Fx_Type, typename Dif_Type,
          std::size_t Np, std::size_t I>
struct DifColumn<Ref_Type, Fx_Type, Dif_Type, Np, I, 0> {
  /**
   * @brief Calculates the difference for the first column index (0).
   *
   * This function calculates the difference for the first column index (0) and
   * does not call itself recursively.
   *
   * @tparam Ref_Type Type of the reference matrix.
   * @tparam Fx_Type Type of the predicted matrix.
   * @tparam Dif_Type Type of the difference container.
   * @tparam Np Prediction horizon length.
   * @param ref Reference matrix.
   * @param Fx Predicted matrix.
   * @param dif Container to store the computed differences.
   */
  static void calculate(const Ref_Type &ref, const Fx_Type &Fx, Dif_Type &dif) {

    calculate_each_dif<Ref_Type::ROWS, Np, I, 0>(ref, Fx, dif);
  }
};

// when I_idx < M
template <typename Ref_Type, typename Fx_Type, typename Dif_Type,
          std::size_t Np, std::size_t M, std::size_t N, std::size_t I_idx>
struct DifRow {
  /**
   * @brief Calculates the difference for a specific row index and calls the
   * next row recursively.
   *
   * This function calculates the difference for the specified row index I_idx
   * and then recursively calls itself to calculate differences for the next row
   * index (I_idx - 1).
   *
   * @tparam Ref_Type Type of the reference matrix.
   * @tparam Fx_Type Type of the predicted matrix.
   * @tparam Dif_Type Type of the difference container.
   * @tparam Np Prediction horizon length.
   * @tparam M Number of rows in the reference matrix.
   * @tparam N Number of columns in the reference matrix.
   * @tparam I_idx Current row index for the operation.
   * @param ref Reference matrix.
   * @param Fx Predicted matrix.
   * @param dif Container to store the computed differences.
   */
  static void calculate(const Ref_Type &ref, const Fx_Type &Fx, Dif_Type &dif) {
    DifColumn<Ref_Type, Fx_Type, Dif_Type, Np, I_idx, N - 1>::calculate(ref, Fx,
                                                                        dif);
    DifRow<Ref_Type, Fx_Type, Dif_Type, Np, M, N, I_idx - 1>::calculate(ref, Fx,
                                                                        dif);
  }
};

// row recursion termination
template <typename Ref_Type, typename Fx_Type, typename Dif_Type,
          std::size_t Np, std::size_t M, std::size_t N>
struct DifRow<Ref_Type, Fx_Type, Dif_Type, Np, M, N, 0> {
  /**
   * @brief Calculates the difference for the first row index (0).
   *
   * This function calculates the difference for the first row index (0) and
   * does not call itself recursively.
   *
   * @tparam Ref_Type Type of the reference matrix.
   * @tparam Fx_Type Type of the predicted matrix.
   * @tparam Dif_Type Type of the difference container.
   * @tparam Np Prediction horizon length.
   * @param ref Reference matrix.
   * @param Fx Predicted matrix.
   * @param dif Container to store the computed differences.
   */
  static void calculate(const Ref_Type &ref, const Fx_Type &Fx, Dif_Type &dif) {
    DifColumn<Ref_Type, Fx_Type, Dif_Type, Np, 0, N - 1>::calculate(ref, Fx,
                                                                    dif);
  }
};

/**
 * @brief Calculates the difference between a reference trajectory and a
 * predicted trajectory.
 *
 * This function computes the difference for each element in the reference
 * trajectory and the predicted trajectory, storing the results in the provided
 * difference container. It uses the DifRow class to handle the row-wise
 * calculations.
 *
 * @tparam Np Number of prediction steps.
 * @tparam Number_Of_Output Number of outputs in the reference trajectory.
 * @tparam Ref_Type Type of the reference matrix.
 * @tparam Fx_Type Type of the predicted matrix.
 * @tparam Dif_Type Type of the difference container.
 * @param ref Reference matrix.
 * @param Fx Predicted matrix.
 * @param dif Container to store the computed differences.
 */
template <std::size_t Np, std::size_t Number_Of_Output, typename Ref_Type,
          typename Fx_Type, typename Dif_Type>
inline void calculate_dif(const Ref_Type &ref, const Fx_Type &Fx,
                          Dif_Type &dif) {
  DifRow<Ref_Type, Fx_Type, Dif_Type, Np, Np, Number_Of_Output,
         (Np - 1)>::calculate(ref, Fx, dif);
}

} // namespace MPC_ReferenceTrajectoryOperation

/* MPC Reference Trajectory */
template <typename Ref_Type, std::size_t Np> class MPC_ReferenceTrajectory {
protected:
  /* Type */
  using _T = typename Ref_Type::Value_Type;

public:
  /* Type */
  using Value_Type = _T;

  /* Check Compatibility */
  static_assert((Ref_Type::ROWS == Np) || (Ref_Type::ROWS == 1),
                "Ref_Type::ROWS must be equal to Np, or 1");

  using Dif_Type = PythonNumpy::DenseMatrix_Type<_T, (Np * Ref_Type::COLS), 1>;

public:
  /* Constructor */
  MPC_ReferenceTrajectory() : reference() {}

  MPC_ReferenceTrajectory(const Ref_Type &reference_in)
      : reference(reference_in) {}

  /* Copy Constructor */
  MPC_ReferenceTrajectory(const MPC_ReferenceTrajectory<Ref_Type, Np> &input)
      : reference(input.reference) {}

  MPC_ReferenceTrajectory<Ref_Type, Np> &
  operator=(const MPC_ReferenceTrajectory<Ref_Type, Np> &input) {
    if (this != &input) {
      this->reference = input.reference;
    }
    return *this;
  }

  /* Move Constructor */
  MPC_ReferenceTrajectory(
      MPC_ReferenceTrajectory<Ref_Type, Np> &&input) noexcept
      : reference(std::move(input.reference)) {}

  MPC_ReferenceTrajectory<Ref_Type, Np> &
  operator=(MPC_ReferenceTrajectory<Ref_Type, Np> &&input) noexcept {
    if (this != &input) {
      this->reference = std::move(input.reference);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Calculates the difference (dif) between the reference trajectory and
   * the provided function Fx.
   *
   * This function computes the difference using the static method
   * MPC_ReferenceTrajectoryOperation::calculate_dif, which compares the current
   * reference trajectory with the given Fx object and stores the result in a
   * Dif_Type object.
   *
   * @tparam Fx_Type The type of the function or object to compare against the
   * reference trajectory. Must have a nested type Value_Type equal to _T.
   * @param Fx The function or object to compare with the reference trajectory.
   * @return Dif_Type The computed difference between the reference trajectory
   * and Fx.
   *
   * @note A static assertion ensures that Fx_Type::Value_Type matches the
   * expected Value_Type (_T).
   */
  template <typename Fx_Type>
  inline auto calculate_dif(const Fx_Type &Fx) -> Dif_Type {

    static_assert(std::is_same<typename Fx_Type::Value_Type, _T>::value,
                  "Fx_Type::Value_Type must be equal to Value_Type");

    Dif_Type dif;

    MPC_ReferenceTrajectoryOperation::calculate_dif<NP, NUMBER_OF_OUTPUT>(
        this->reference, Fx, dif);

    return dif;
  }

public:
  /* Constant */
  static constexpr std::size_t NP = Np;
  static constexpr std::size_t NUMBER_OF_OUTPUT = Ref_Type::COLS;

public:
  /* Variables */
  Ref_Type reference;
};

/* make MPC Reference Trajectory */

/**
 * @brief Factory function to create an instance of MPC_ReferenceTrajectory.
 *
 * This function initializes and returns an instance of MPC_ReferenceTrajectory
 * with the provided template parameters.
 *
 * @tparam Ref_Type Type of the reference matrix.
 * @tparam Np Number of prediction steps.
 * @return An instance of MPC_ReferenceTrajectory initialized with default
 * values.
 */
template <typename Ref_Type, std::size_t Np>
inline auto make_MPC_ReferenceTrajectory(void)
    -> MPC_ReferenceTrajectory<Ref_Type, Np> {
  return MPC_ReferenceTrajectory<Ref_Type, Np>();
}

/* MPC Reference Trajectory Type */
template <typename Ref_Type, std::size_t Np>
using MPC_ReferenceTrajectory_Type = MPC_ReferenceTrajectory<Ref_Type, Np>;

} // namespace PythonMPC

#endif // __MPC_STATE_SPACE_UTILITY_HPP__
