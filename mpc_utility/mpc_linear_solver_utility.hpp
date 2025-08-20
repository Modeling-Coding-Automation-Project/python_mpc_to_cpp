/**
 * @file mpc_linear_solver_utility.hpp
 * @brief Utility classes and functions for Model Predictive Control (MPC)
 * linear solver in C++.
 *
 * This header provides a set of template classes and meta-programming utilities
 * to support the construction and solution of quadratic programming (QP)
 * problems for linear MPC. It includes constraint management, matrix
 * operations, and QP solver integration, designed for use with Python/C++
 * hybrid MPC frameworks.
 */
#ifndef __MPC_LINEAR_SOLVER_UTILITY__
#define __MPC_LINEAR_SOLVER_UTILITY__

#include "python_numpy.hpp"
#include "python_optimization.hpp"

#include <functional>
#include <type_traits>

namespace PythonMPC {

namespace SolverUtility {

static constexpr std::size_t MAX_ITERATION_DEFAULT = 10;
static constexpr double TOL_DEFAULT = 1e-6;

} // namespace SolverUtility

namespace SolverUtilityOperation {

/**
 * @brief A utility structure to count the number of true conditions in a
 * boolean flag.
 *
 * This structure is used to count how many times a condition evaluates to
 * true in a compile-time context.
 */
template <bool Flag> struct CountTrueCondition {};

/**
 * @brief Specialization of CountTrueCondition for true.
 *
 * This specialization sets the value to 1 when the condition is true.
 */
template <> struct CountTrueCondition<true> {
  static constexpr std::size_t value = static_cast<std::size_t>(1);
};

/**
 * @brief Specialization of CountTrueCondition for false.
 *
 * This specialization sets the value to 0 when the condition is false.
 */
template <> struct CountTrueCondition<false> {
  static constexpr std::size_t value = static_cast<std::size_t>(0);
};

/**
 * @brief A utility structure to count the number of true conditions in a 2D
 * boolean flag array.
 *
 * This structure is used to count how many times a condition evaluates to
 * true in a 2D compile-time context.
 */
template <typename Flags, std::size_t Col, std::size_t Row>
struct CountTrue2D_Row {
  static constexpr std::size_t value =
      CountTrueCondition<Flags::lists[Col][Row]>::value +
      CountTrue2D_Row<Flags, Col, Row - 1>::value;
};

/**
 * @brief Specialization of CountTrue2D_Row for the last row.
 *
 * This specialization sets the value to 0 when the row index is -1, indicating
 * no more rows to count.
 */
template <typename Flags, std::size_t Col>
struct CountTrue2D_Row<Flags, Col, static_cast<std::size_t>(-1)> {
  static constexpr std::size_t value = static_cast<std::size_t>(0);
};

/**
 * @brief A utility structure to count the number of true conditions in a 2D
 * boolean flag array by columns.
 *
 * This structure is used to count how many times a condition evaluates to
 * true in a 2D compile-time context, iterating over columns.
 */
template <typename Flags, std::size_t Col, std::size_t Row>
struct CountTrue2D_Col {
  static constexpr std::size_t value =
      CountTrue2D_Row<Flags, Col, Row - 1>::value +
      CountTrue2D_Col<Flags, Col - 1, Row>::value;
};

/**
 * @brief Specialization of CountTrue2D_Col for the last column.
 *
 * This specialization sets the value to 0 when the column index is -1,
 * indicating no more columns to count.
 */
template <typename Flags, std::size_t Row>
struct CountTrue2D_Col<Flags, static_cast<std::size_t>(-1), Row> {
  static constexpr std::size_t value = static_cast<std::size_t>(0);
};

/**
 * @brief Alias template to count the number of true values in a 2D flag array
 * up to a specified column and row.
 *
 * This alias uses the CountTrue2D_Col metafunction to recursively count the
 * number of true values in a 2D compile-time flag array (Flags), considering
 * columns up to (Col - 1) and a specified row (Row).
 *
 * @tparam Flags The 2D array of boolean flags (typically a template parameter
 * pack or array).
 * @tparam Col The number of columns to consider (exclusive upper bound).
 * @tparam Row The row index to consider.
 */
template <typename Flags, std::size_t Col, std::size_t Row>
using CountTrue2D = CountTrue2D_Col<Flags, Col - 1, Row>;

} // namespace SolverUtilityOperation

/* Define delta U, U, Y limits */

/**
 * @brief A class to represent limits for delta U, U, and Y in a Model
 * Predictive Control (MPC) context.
 *
 * This class encapsulates the limits for delta U (change in control input),
 * U (control input), and Y (output) with their respective minimum and maximum
 * values. It provides methods to check the activity of each limit and to
 * retrieve the number of constraints.
 */
template <typename Delta_U_Min_Type, typename Delta_U_Max_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type>
class DU_U_Y_Limits {
public:
  /* Type */
  using Value_Type = typename Delta_U_Min_Type::Value_Type;

  using Delta_U_Min_Flags = typename Delta_U_Min_Type::SparseAvailable_Type;
  using Delta_U_Max_Flags = typename Delta_U_Max_Type::SparseAvailable_Type;
  using U_Min_Flags = typename U_Min_Type::SparseAvailable_Type;
  using U_Max_Flags = typename U_Max_Type::SparseAvailable_Type;
  using Y_Min_Flags = typename Y_Min_Type::SparseAvailable_Type;
  using Y_Max_Flags = typename Y_Max_Type::SparseAvailable_Type;

  /* Check Compatibility */
  static_assert(
      std::is_same<typename Delta_U_Max_Type::Value_Type, Value_Type>::value,
      "Delta_U_Min_Type::Value_Type must be equal to Value_Type");
  static_assert(
      std::is_same<typename U_Min_Type::Value_Type, Value_Type>::value,
      "Delta_U_Max_Type::Value_Type must be equal to Value_Type");
  static_assert(
      std::is_same<typename U_Max_Type::Value_Type, Value_Type>::value,
      "U_Min_Type::Value_Type must be equal to Value_Type");
  static_assert(
      std::is_same<typename Y_Min_Type::Value_Type, Value_Type>::value,
      "U_Max_Type::Value_Type must be equal to Value_Type");
  static_assert(
      std::is_same<typename Y_Max_Type::Value_Type, Value_Type>::value,
      "Y_Min_Type::Value_Type must be equal to Value_Type");

  static_assert(Delta_U_Min_Type::COLS == Delta_U_Max_Type::COLS &&
                    Delta_U_Min_Type::ROWS == Delta_U_Max_Type::ROWS,
                "Delta_U_Min_Type size must be equal to Delta_U_Max_Type");

  static_assert(Delta_U_Min_Type::COLS == U_Min_Type::COLS &&
                    Delta_U_Min_Type::ROWS == U_Min_Type::ROWS,
                "Delta_U_Min_Type size must be equal to U_Min_Type");

  static_assert(U_Min_Type::COLS == U_Max_Type::COLS &&
                    U_Min_Type::ROWS == U_Max_Type::ROWS,
                "U_Min_Type size must be equal to U_Max_Type");

  static_assert(Y_Min_Type::COLS == Y_Max_Type::COLS &&
                    Y_Min_Type::ROWS == Y_Max_Type::ROWS,
                "Y_Min_Type size must be equal to Y_Max_Type");

protected:
  /* Type */
  using _T = Value_Type;

public:
  /* Constant */
  static constexpr std::size_t DELTA_U_MIN_SIZE = Delta_U_Min_Type::COLS;
  static constexpr std::size_t DELTA_U_MAX_SIZE = Delta_U_Max_Type::COLS;
  static constexpr std::size_t U_MIN_SIZE = U_Min_Type::COLS;
  static constexpr std::size_t U_MAX_SIZE = U_Max_Type::COLS;
  static constexpr std::size_t Y_MIN_SIZE = Y_Min_Type::COLS;
  static constexpr std::size_t Y_MAX_SIZE = Y_Max_Type::COLS;

public:
  /* Constructor */
  DU_U_Y_Limits()
      : delta_U_min(), delta_U_max(), U_min(), U_max(), Y_min(), Y_max() {}

  DU_U_Y_Limits(const Delta_U_Min_Type &delta_U_min_in,
                const Delta_U_Max_Type &delta_U_max_in,
                const U_Min_Type &U_min_in, const U_Max_Type &U_max_in,
                const Y_Min_Type &Y_min_in, const Y_Max_Type &Y_max_in)
      : delta_U_min(delta_U_min_in), delta_U_max(delta_U_max_in),
        U_min(U_min_in), U_max(U_max_in), Y_min(Y_min_in), Y_max(Y_max_in) {}

  /* Copy Constructor */
  DU_U_Y_Limits(
      const DU_U_Y_Limits<Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
                          U_Max_Type, Y_Min_Type, Y_Max_Type> &input)
      : delta_U_min(input.delta_U_min), delta_U_max(input.delta_U_max),
        U_min(input.U_min), U_max(input.U_max), Y_min(input.Y_min),
        Y_max(input.Y_max) {}

  DU_U_Y_Limits<Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type, U_Max_Type,
                Y_Min_Type, Y_Max_Type> &
  operator=(const DU_U_Y_Limits<Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
                                U_Max_Type, Y_Min_Type, Y_Max_Type> &input) {
    if (this != &input) {
      this->delta_U_min = input.delta_U_min;
      this->delta_U_max = input.delta_U_max;
      this->U_min = input.U_min;
      this->U_max = input.U_max;
      this->Y_min = input.Y_min;
      this->Y_max = input.Y_max;
    }
    return *this;
  }

  /* Move Constructor */
  DU_U_Y_Limits(
      DU_U_Y_Limits<Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type, U_Max_Type,
                    Y_Min_Type, Y_Max_Type> &&input) noexcept
      : delta_U_min(std::move(input.delta_U_min)),
        delta_U_max(std::move(input.delta_U_max)),
        U_min(std::move(input.U_min)), U_max(std::move(input.U_max)),
        Y_min(std::move(input.Y_min)), Y_max(std::move(input.Y_max)) {}

  DU_U_Y_Limits<Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type, U_Max_Type,
                Y_Min_Type, Y_Max_Type> &
  operator=(
      DU_U_Y_Limits<Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type, U_Max_Type,
                    Y_Min_Type, Y_Max_Type> &&input) noexcept {
    if (this != &input) {
      this->delta_U_min = std::move(input.delta_U_min);
      this->delta_U_max = std::move(input.delta_U_max);
      this->U_min = std::move(input.U_min);
      this->U_max = std::move(input.U_max);
      this->Y_min = std::move(input.Y_min);
      this->Y_max = std::move(input.Y_max);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Check if a delta U min constraint is active at a given index.
   *
   * This function checks if the delta U min constraint at the specified index
   * is active. It wraps the index to ensure it is within the valid range.
   *
   * @param index The index of the delta U min constraint to check.
   * @return True if the constraint is active, false otherwise.
   */
  inline auto is_delta_U_min_active(const std::size_t &index) const -> bool {

    std::size_t index_wrapped;

    if (index < 0) {
      index_wrapped = 0;
    } else if (index >= DELTA_U_MIN_SIZE) {
      index_wrapped = DELTA_U_MIN_SIZE - 1;
    } else {
      index_wrapped = index;
    }

    return Delta_U_Min_Flags::lists[index_wrapped][0];
  }

  /**
   * @brief Check if a delta U max constraint is active at a given index.
   *
   * This function checks if the delta U max constraint at the specified index
   * is active. It wraps the index to ensure it is within the valid range.
   *
   * @param index The index of the delta U max constraint to check.
   * @return True if the constraint is active, false otherwise.
   */
  inline auto is_delta_U_max_active(const std::size_t &index) const -> bool {

    std::size_t index_wrapped;

    if (index < 0) {
      index_wrapped = 0;
    } else if (index >= DELTA_U_MAX_SIZE) {
      index_wrapped = DELTA_U_MAX_SIZE - 1;
    } else {
      index_wrapped = index;
    }

    return Delta_U_Max_Flags::lists[index_wrapped][0];
  }

  /**
   * @brief Check if a U min constraint is active at a given index.
   *
   * This function checks if the U min constraint at the specified index is
   * active. It wraps the index to ensure it is within the valid range.
   *
   * @param index The index of the U min constraint to check.
   * @return True if the constraint is active, false otherwise.
   */
  inline auto is_U_min_active(const std::size_t &index) const -> bool {

    std::size_t index_wrapped;

    if (index < 0) {
      index_wrapped = 0;
    } else if (index >= U_MIN_SIZE) {
      index_wrapped = U_MIN_SIZE - 1;
    } else {
      index_wrapped = index;
    }

    return U_Min_Flags::lists[index_wrapped][0];
  }

  /**
   * @brief Check if a U max constraint is active at a given index.
   *
   * This function checks if the U max constraint at the specified index is
   * active. It wraps the index to ensure it is within the valid range.
   *
   * @param index The index of the U max constraint to check.
   * @return True if the constraint is active, false otherwise.
   */
  inline auto is_U_max_active(const std::size_t &index) const -> bool {

    std::size_t index_wrapped;

    if (index < 0) {
      index_wrapped = 0;
    } else if (index >= U_MAX_SIZE) {
      index_wrapped = U_MAX_SIZE - 1;
    } else {
      index_wrapped = index;
    }

    return U_Max_Flags::lists[index_wrapped][0];
  }

  /**
   * @brief Check if a Y min constraint is active at a given index.
   *
   * This function checks if the Y min constraint at the specified index is
   * active. It wraps the index to ensure it is within the valid range.
   *
   * @param index The index of the Y min constraint to check.
   * @return True if the constraint is active, false otherwise.
   */
  inline auto is_Y_min_active(const std::size_t &index) const -> bool {

    std::size_t index_wrapped;

    if (index < 0) {
      index_wrapped = 0;
    } else if (index >= Y_MIN_SIZE) {
      index_wrapped = Y_MIN_SIZE - 1;
    } else {
      index_wrapped = index;
    }

    return Y_Min_Flags::lists[index_wrapped][0];
  }

  /**
   * @brief Check if a Y max constraint is active at a given index.
   *
   * This function checks if the Y max constraint at the specified index is
   * active. It wraps the index to ensure it is within the valid range.
   *
   * @param index The index of the Y max constraint to check.
   * @return True if the constraint is active, false otherwise.
   */
  inline auto is_Y_max_active(const std::size_t &index) const -> bool {

    std::size_t index_wrapped;

    if (index < 0) {
      index_wrapped = 0;
    } else if (index >= Y_MAX_SIZE) {
      index_wrapped = Y_MAX_SIZE - 1;
    } else {
      index_wrapped = index;
    }

    return Y_Max_Flags::lists[index_wrapped][0];
  }

  /**
   * @brief Get the total number of all constraints.
   *
   * This function returns the total number of constraints across delta U,
   * U, and Y limits.
   *
   * @return The total number of constraints.
   */
  inline auto get_number_of_all_constraints(void) const -> std::size_t {

    return this->_number_of_delta_U_constraints +
           this->_number_of_U_constraints + this->_number_of_Y_constraints;
  }

  /**
   * @brief Get the number of delta U constraints.
   *
   * This function returns the total number of delta U constraints.
   *
   * @return The number of delta U constraints.
   */
  inline auto get_number_of_delta_U_constraints(void) const -> std::size_t {
    return this->_number_of_delta_U_constraints;
  }

  /**
   * @brief Get the number of U constraints.
   *
   * This function returns the total number of U constraints.
   *
   * @return The number of U constraints.
   */
  inline auto get_number_of_U_constraints(void) const -> std::size_t {
    return this->_number_of_U_constraints;
  }

  /**
   * @brief Get the number of Y constraints.
   *
   * This function returns the total number of Y constraints.
   *
   * @return The number of Y constraints.
   */
  inline auto get_number_of_Y_constraints(void) const -> std::size_t {
    return this->_number_of_Y_constraints;
  }

public:
  /* Constant */
  static constexpr std::size_t DELTA_U_MIN_CONSTRAINTS =
      SolverUtilityOperation::CountTrue2D<Delta_U_Min_Flags,
                                          Delta_U_Min_Type::COLS,
                                          Delta_U_Min_Type::ROWS>::value;
  static constexpr std::size_t DELTA_U_MAX_CONSTRAINTS =
      SolverUtilityOperation::CountTrue2D<Delta_U_Max_Flags,
                                          Delta_U_Max_Type::COLS,
                                          Delta_U_Max_Type::ROWS>::value;

  static constexpr std::size_t NUMBER_OF_DELTA_U_CONSTRAINTS =
      DELTA_U_MIN_CONSTRAINTS + DELTA_U_MAX_CONSTRAINTS;

  static constexpr std::size_t U_MIN_CONSTRAINTS =
      SolverUtilityOperation::CountTrue2D<U_Min_Flags, U_Min_Type::COLS,
                                          U_Min_Type::ROWS>::value;
  static constexpr std::size_t U_MAX_CONSTRAINTS =
      SolverUtilityOperation::CountTrue2D<U_Max_Flags, U_Max_Type::COLS,
                                          U_Max_Type::ROWS>::value;

  static constexpr std::size_t NUMBER_OF_U_CONSTRAINTS =
      U_MIN_CONSTRAINTS + U_MAX_CONSTRAINTS;

  static constexpr std::size_t Y_MIN_CONSTRAINTS =
      SolverUtilityOperation::CountTrue2D<Y_Min_Flags, Y_Min_Type::COLS,
                                          Y_Min_Type::ROWS>::value;
  static constexpr std::size_t Y_MAX_CONSTRAINTS =
      SolverUtilityOperation::CountTrue2D<Y_Max_Flags, Y_Max_Type::COLS,
                                          Y_Max_Type::ROWS>::value;

  static constexpr std::size_t NUMBER_OF_Y_CONSTRAINTS =
      Y_MIN_CONSTRAINTS + Y_MAX_CONSTRAINTS;

  static constexpr std::size_t NUMBER_OF_ALL_CONSTRAINTS =
      NUMBER_OF_DELTA_U_CONSTRAINTS + NUMBER_OF_U_CONSTRAINTS +
      NUMBER_OF_Y_CONSTRAINTS;

public:
  /* Variables */
  Delta_U_Min_Type delta_U_min;
  Delta_U_Max_Type delta_U_max;
  U_Min_Type U_min;
  U_Max_Type U_max;
  Y_Min_Type Y_min;
  Y_Max_Type Y_max;
};

/* make DU_U_Y_Limits */

/**
 * @brief Factory function to create a DU_U_Y_Limits object.
 *
 * This function constructs a DU_U_Y_Limits object with the provided limits for
 * delta U, U, and Y.
 *
 * @tparam Delta_U_Min_Type Type of the delta U minimum limits.
 * @tparam Delta_U_Max_Type Type of the delta U maximum limits.
 * @tparam U_Min_Type Type of the U minimum limits.
 * @tparam U_Max_Type Type of the U maximum limits.
 * @tparam Y_Min_Type Type of the Y minimum limits.
 * @tparam Y_Max_Type Type of the Y maximum limits.
 * @param delta_U_min_in Input for delta U minimum limits.
 * @param delta_U_max_in Input for delta U maximum limits.
 * @param U_min_in Input for U minimum limits.
 * @param U_max_in Input for U maximum limits.
 * @param Y_min_in Input for Y minimum limits.
 * @param Y_max_in Input for Y maximum limits.
 * @return A DU_U_Y_Limits object initialized with the provided limits.
 */
template <typename Delta_U_Min_Type, typename Delta_U_Max_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type>
inline auto
make_DU_U_Y_Limits(const Delta_U_Min_Type &delta_U_min_in,
                   const Delta_U_Max_Type &delta_U_max_in,
                   const U_Min_Type &U_min_in, const U_Max_Type &U_max_in,
                   const Y_Min_Type &Y_min_in, const Y_Max_Type &Y_max_in)
    -> DU_U_Y_Limits<Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type, U_Max_Type,
                     Y_Min_Type, Y_Max_Type> {

  return DU_U_Y_Limits<Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
                       U_Max_Type, Y_Min_Type, Y_Max_Type>(
      delta_U_min_in, delta_U_max_in, U_min_in, U_max_in, Y_min_in, Y_max_in);
}

/* DU_U_Y_Limits Type */
template <typename Delta_U_Min_Type, typename Delta_U_Max_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type>
using DU_U_Y_Limits_Type =
    DU_U_Y_Limits<Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type, U_Max_Type,
                  Y_Min_Type, Y_Max_Type>;

namespace LMPC_QP_SolverOperation {

/* calculate M gamma for delta U min  */
template <typename M_Type, typename Gamma_Type,
          typename Delta_U_min_Matrix_Type, std::size_t I, bool Flag>
struct Calculate_M_Gamma_Delta_U_Min_Condition {};

template <typename M_Type, typename Gamma_Type,
          typename Delta_U_min_Matrix_Type, std::size_t I>
struct Calculate_M_Gamma_Delta_U_Min_Condition<
    M_Type, Gamma_Type, Delta_U_min_Matrix_Type, I, true> {
  /**
   * @brief Applies constraints to the given matrices for a specific index.
   *
   * This static function modifies the matrix M and vector gamma by applying a
   * constraint at the position specified by initial_position and the template
   * parameter I. It sets the corresponding entry in M to -1.0 and updates gamma
   * using the value from delta_U_matrix at position <I, 0>. The set_count is
   * incremented to reflect the addition of a new constraint.
   *
   * @tparam I Index at which the constraint is applied (template parameter).
   * @tparam M_Type Type of the matrix M.
   * @tparam Gamma_Type Type of the vector gamma.
   * @tparam Delta_U_min_Matrix_Type Type of the delta_U_matrix.
   * @param M Reference to the matrix to be modified.
   * @param gamma Reference to the vector to be modified.
   * @param delta_U_matrix Matrix providing the constraint value.
   * @param set_count Reference to the counter tracking the number of
   * constraints set.
   * @param initial_position The starting position in M and gamma for applying
   * the constraint.
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Delta_U_min_Matrix_Type &delta_U_matrix,
                    std::size_t &set_count,
                    const std::size_t &initial_position) {

    M.access(initial_position + I, I) =
        static_cast<typename M_Type::Value_Type>(-1.0);
    gamma.access(initial_position + I, 0) =
        -delta_U_matrix.template get<I, 0>();
    set_count += 1;
  }
};

template <typename M_Type, typename Gamma_Type,
          typename Delta_U_min_Matrix_Type, std::size_t I>
struct Calculate_M_Gamma_Delta_U_Min_Condition<
    M_Type, Gamma_Type, Delta_U_min_Matrix_Type, I, false> {
  /**
   * @brief A no-op function for the case when the condition is false.
   * This function does nothing and is used to maintain the structure of the
   * template specialization.
   * @param M Reference to the matrix (unused).
   * @param gamma Reference to the vector (unused).
   * @param delta_U_matrix Matrix providing the constraint value (unused).
   * @param set_count Reference to the counter tracking the number of
   * constraints set (unused).
   * @param initial_position The starting position in M and gamma for applying
   * the constraint (unused).
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Delta_U_min_Matrix_Type &delta_U_matrix,
                    std::size_t &set_count,
                    const std::size_t &initial_position) {

    // Do Nothing.
    static_cast<void>(M);
    static_cast<void>(gamma);
    static_cast<void>(delta_U_matrix);
    static_cast<void>(set_count);
    static_cast<void>(initial_position);
  }
};

template <typename Delta_U_min_Matrix_Type, typename M_Type,
          typename Gamma_Type, std::size_t I, std::size_t I_Dif>
struct Calculate_M_Gamma_Delta_U_Min_Loop {
  /**
   * @brief Applies constraints to the given matrices for a range of indices.
   *
   * This static function iterates over the indices from I to I_Dif, applying
   * constraints to the matrix M and vector gamma based on the delta_U_matrix.
   * It updates the total_index with the number of constraints set and calls
   * itself recursively for the next index.
   *
   * @tparam Delta_U_min_Matrix_Type Type of the delta_U_min_matrix.
   * @tparam M_Type Type of the matrix M.
   * @tparam Gamma_Type Type of the vector gamma.
   * @tparam I Current index in the iteration (template parameter).
   * @tparam I_Dif The difference in indices for recursion (template parameter).
   * @param M Reference to the matrix to be modified.
   * @param gamma Reference to the vector to be modified.
   * @param delta_U_matrix Matrix providing the constraint value.
   * @param total_index Reference to the counter tracking the total number of
   * constraints set.
   * @param initial_position The starting position in M and gamma for applying
   * the constraints.
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Delta_U_min_Matrix_Type &delta_U_matrix,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {

    std::size_t set_count = static_cast<std::size_t>(0);

    Calculate_M_Gamma_Delta_U_Min_Condition<
        M_Type, Gamma_Type, Delta_U_min_Matrix_Type, I,
        Delta_U_min_Matrix_Type::SparseAvailable_Type::lists[I][0]>::
        apply(M, gamma, delta_U_matrix, set_count, initial_position);

    total_index += set_count;
    Calculate_M_Gamma_Delta_U_Min_Loop<Delta_U_min_Matrix_Type, M_Type,
                                       Gamma_Type, (I + 1),
                                       (I_Dif - 1)>::apply(M, gamma,
                                                           delta_U_matrix,
                                                           total_index,
                                                           initial_position);
  }
};

template <typename Delta_U_min_Matrix_Type, typename M_Type,
          typename Gamma_Type, std::size_t I>
struct Calculate_M_Gamma_Delta_U_Min_Loop<Delta_U_min_Matrix_Type, M_Type,
                                          Gamma_Type, I, 0> {
  /**
   * @brief A no-op function for the case when the loop has no more indices to
   * process.
   *
   * This static function does nothing and is used to terminate the recursion
   * in the template specialization.
   *
   * @tparam Delta_U_min_Matrix_Type Type of the delta_U_min_matrix (unused).
   * @tparam M_Type Type of the matrix M (unused).
   * @tparam Gamma_Type Type of the vector gamma (unused).
   * @param M Reference to the matrix (unused).
   * @param gamma Reference to the vector (unused).
   * @param delta_U_matrix Matrix providing the constraint value (unused).
   * @param total_index Reference to the counter tracking the total number of
   * constraints set (unused).
   * @param initial_position The starting position in M and gamma for applying
   * the constraints (unused).
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Delta_U_min_Matrix_Type &delta_U_matrix,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {

    // Do Nothing.
    static_cast<void>(M);
    static_cast<void>(gamma);
    static_cast<void>(delta_U_matrix);
    static_cast<void>(total_index);
    static_cast<void>(initial_position);
  }
};

/**
 * @brief Alias template to calculate M gamma for delta U min constraints.
 *
 * This alias uses the Calculate_M_Gamma_Delta_U_Min_Loop metafunction to
 * recursively apply constraints for delta U min, starting from index 0 and
 * iterating through the specified Delta_U_Size.
 *
 * @tparam M_Type Type of the matrix M.
 * @tparam Gamma_Type Type of the vector gamma.
 * @tparam Delta_U_min_Matrix_Type Type of the delta U minimum matrix.
 * @tparam Delta_U_Size The size of the delta U constraints.
 */
template <typename M_Type, typename Gamma_Type,
          typename Delta_U_min_Matrix_Type, std::size_t Delta_U_Size>
using Calculate_M_Gamma_Delta_U_Min =
    Calculate_M_Gamma_Delta_U_Min_Loop<Delta_U_min_Matrix_Type, M_Type,
                                       Gamma_Type, 0, Delta_U_Size>;

/* calculate M gamma for delta U max  */

/**
 * @brief A utility structure to calculate M gamma for delta U max constraints.
 *
 * This structure defines a loop to apply constraints for delta U max, iterating
 * through the indices and applying conditions based on the flags defined in the
 * Delta_U_max_Matrix_Type.
 */
template <typename M_Type, typename Gamma_Type,
          typename Delta_U_max_Matrix_Type, std::size_t I, bool Flag>
struct Calculate_M_Gamma_Delta_U_Max_Condition {};

template <typename M_Type, typename Gamma_Type,
          typename Delta_U_max_Matrix_Type, std::size_t I>
struct Calculate_M_Gamma_Delta_U_Max_Condition<
    M_Type, Gamma_Type, Delta_U_max_Matrix_Type, I, true> {
  /**
   * @brief Applies constraints to the given matrices for a specific index.
   *
   * This static function modifies the matrix M and vector gamma by applying a
   * constraint at the position specified by initial_position and the template
   * parameter I. It sets the corresponding entry in M to 1.0 and updates gamma
   * using the value from delta_U_matrix at position <I, 0>. The set_count is
   * incremented to reflect the addition of a new constraint.
   *
   * @tparam I Index at which the constraint is applied (template parameter).
   * @tparam M_Type Type of the matrix M.
   * @tparam Gamma_Type Type of the vector gamma.
   * @tparam Delta_U_max_Matrix_Type Type of the delta_U_matrix.
   * @param M Reference to the matrix to be modified.
   * @param gamma Reference to the vector to be modified.
   * @param delta_U_matrix Matrix providing the constraint value.
   * @param set_count Reference to the counter tracking the number of
   * constraints set.
   * @param initial_position The starting position in M and gamma for applying
   * the constraint.
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Delta_U_max_Matrix_Type &delta_U_matrix,
                    std::size_t &set_count,
                    const std::size_t &initial_position) {

    M.access(initial_position + I, I) =
        static_cast<typename M_Type::Value_Type>(1.0);
    gamma.access(initial_position + I, 0) = delta_U_matrix.template get<I, 0>();
    set_count += 1;
  }
};

template <typename M_Type, typename Gamma_Type,
          typename Delta_U_max_Matrix_Type, std::size_t I>
struct Calculate_M_Gamma_Delta_U_Max_Condition<
    M_Type, Gamma_Type, Delta_U_max_Matrix_Type, I, false> {
  /**
   * @brief A no-op function for the case when the condition is false.
   * This function does nothing and is used to maintain the structure of the
   * template specialization.
   * @param M Reference to the matrix (unused).
   * @param gamma Reference to the vector (unused).
   * @param delta_U_matrix Matrix providing the constraint value (unused).
   * @param set_count Reference to the counter tracking the number of
   * constraints set (unused).
   * @param initial_position The starting position in M and gamma for applying
   * the constraint (unused).
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Delta_U_max_Matrix_Type &delta_U_matrix,
                    std::size_t &set_count,
                    const std::size_t &initial_position) {

    // Do Nothing.
    static_cast<void>(M);
    static_cast<void>(gamma);
    static_cast<void>(delta_U_matrix);
    static_cast<void>(set_count);
    static_cast<void>(initial_position);
  }
};

template <typename M_Type, typename Gamma_Type,
          typename Delta_U_max_Matrix_Type, std::size_t I, std::size_t I_Dif>
struct Calculate_M_Gamma_Delta_U_Max_Loop {
  /**
   * @brief Applies constraints to the given matrices for a range of indices.
   *
   * This static function iterates over the indices from I to I_Dif, applying
   * constraints to the matrix M and vector gamma based on the delta_U_matrix.
   * It updates the total_index with the number of constraints set and calls
   * itself recursively for the next index.
   *
   * @tparam M_Type Type of the matrix M.
   * @tparam Gamma_Type Type of the vector gamma.
   * @tparam Delta_U_max_Matrix_Type Type of the delta U maximum matrix.
   * @tparam I Current index in the iteration (template parameter).
   * @tparam I_Dif The difference in indices for recursion (template parameter).
   * @param M Reference to the matrix to be modified.
   * @param gamma Reference to the vector to be modified.
   * @param delta_U_matrix Matrix providing the constraint value.
   * @param total_index Reference to the counter tracking the total number of
   * constraints set.
   * @param initial_position The starting position in M and gamma for applying
   * the constraints.
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Delta_U_max_Matrix_Type &delta_U_matrix,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {

    std::size_t set_count = static_cast<std::size_t>(0);

    Calculate_M_Gamma_Delta_U_Max_Condition<
        M_Type, Gamma_Type, Delta_U_max_Matrix_Type, I,
        Delta_U_max_Matrix_Type::SparseAvailable_Type::lists[I][0]>::
        apply(M, gamma, delta_U_matrix, set_count, initial_position);

    total_index += set_count;
    Calculate_M_Gamma_Delta_U_Max_Loop<M_Type, Gamma_Type,
                                       Delta_U_max_Matrix_Type, (I + 1),
                                       (I_Dif - 1)>::apply(M, gamma,
                                                           delta_U_matrix,
                                                           total_index,
                                                           initial_position);
  }
};

template <typename M_Type, typename Gamma_Type,
          typename Delta_U_max_Matrix_Type, std::size_t I>
struct Calculate_M_Gamma_Delta_U_Max_Loop<M_Type, Gamma_Type,
                                          Delta_U_max_Matrix_Type, I, 0> {
  /**
   * @brief A no-op function for the case when the loop has no more indices to
   * process.
   *
   * This static function does nothing and is used to terminate the recursion
   * in the template specialization.
   *
   * @tparam M_Type Type of the matrix M (unused).
   * @tparam Gamma_Type Type of the vector gamma (unused).
   * @tparam Delta_U_max_Matrix_Type Type of the delta U maximum matrix
   * (unused).
   * @param M Reference to the matrix (unused).
   * @param gamma Reference to the vector (unused).
   * @param delta_U_matrix Matrix providing the constraint value (unused).
   * @param total_index Reference to the counter tracking the total number of
   * constraints set (unused).
   * @param initial_position The starting position in M and gamma for applying
   * the constraints (unused).
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Delta_U_max_Matrix_Type &delta_U_matrix,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {

    // Do Nothing.
    static_cast<void>(M);
    static_cast<void>(gamma);
    static_cast<void>(delta_U_matrix);
    static_cast<void>(total_index);
    static_cast<void>(initial_position);
  }
};

/**
 * @brief Alias template to calculate M gamma for delta U max constraints.
 *
 * This alias uses the Calculate_M_Gamma_Delta_U_Max_Loop metafunction to
 * recursively apply constraints for delta U max, starting from index 0 and
 * iterating through the specified Delta_U_Size.
 *
 * @tparam M_Type Type of the matrix M.
 * @tparam Gamma_Type Type of the vector gamma.
 * @tparam Delta_U_max_Matrix_Type Type of the delta U maximum matrix.
 * @tparam Delta_U_Size The size of the delta U constraints.
 */
template <typename M_Type, typename Gamma_Type,
          typename Delta_U_max_Matrix_Type, std::size_t Delta_U_Size>
using Calculate_M_Gamma_Delta_U_Max = Calculate_M_Gamma_Delta_U_Max_Loop<
    M_Type, Gamma_Type, Delta_U_max_Matrix_Type, 0, Delta_U_Size>;

/* calculate M gamma for U min */

/**
 * @brief A utility structure to calculate M gamma for U min constraints.
 *
 * This structure defines a loop to apply constraints for U min, iterating
 * through the indices and applying conditions based on the flags defined in the
 * U_min_Matrix_Type.
 */
template <typename M_Type, typename Gamma_Type, typename U_min_Matrix_Type,
          typename U_Type, std::size_t I, bool Flag>
struct Calculate_M_Gamma_U_Min_Condition {};

template <typename M_Type, typename Gamma_Type, typename U_min_Matrix_Type,
          typename U_Type, std::size_t I>
struct Calculate_M_Gamma_U_Min_Condition<M_Type, Gamma_Type, U_min_Matrix_Type,
                                         U_Type, I, true> {
  /**
   * @brief Applies constraints to the given matrices for a specific index.
   *
   * This static function modifies the matrix M and vector gamma by applying a
   * constraint at the position specified by initial_position and the template
   * parameter I. It sets the corresponding entry in M to -1.0 and updates gamma
   * using the value from U_matrix at position <0, I>. The set_count is
   * incremented to reflect the addition of a new constraint.
   *
   * @tparam I Index at which the constraint is applied (template parameter).
   * @tparam M_Type Type of the matrix M.
   * @tparam Gamma_Type Type of the vector gamma.
   * @tparam U_min_Matrix_Type Type of the U_min_matrix.
   * @tparam U_Type Type of the U matrix.
   * @param M Reference to the matrix to be modified.
   * @param gamma Reference to the vector to be modified.
   * @param U_matrix Matrix providing the constraint value.
   * @param U Matrix providing additional values for gamma.
   * @param set_count Reference to the counter tracking the number of
   * constraints set.
   * @param initial_position The starting position in M and gamma for applying
   * the constraint.
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const U_min_Matrix_Type &U_matrix, const U_Type &U,
                    std::size_t &set_count,
                    const std::size_t &initial_position) {

    M.access(initial_position + I, I) =
        static_cast<typename M_Type::Value_Type>(-1.0);
    gamma.access(initial_position + I, 0) =
        -U_matrix.template get<I, 0>() + U.template get<I, 0>();
    set_count += 1;
  }
};

template <typename M_Type, typename Gamma_Type, typename U_min_Matrix_Type,
          typename U_Type, std::size_t I>
struct Calculate_M_Gamma_U_Min_Condition<M_Type, Gamma_Type, U_min_Matrix_Type,
                                         U_Type, I, false> {
  /**
   * @brief A no-op function for the case when the condition is false.
   * This function does nothing and is used to maintain the structure of the
   * template specialization.
   * @param M Reference to the matrix (unused).
   * @param gamma Reference to the vector (unused).
   * @param U_matrix Matrix providing the constraint value (unused).
   * @param U Matrix providing additional values for gamma (unused).
   * @param set_count Reference to the counter tracking the number of
   * constraints set (unused).
   * @param initial_position The starting position in M and gamma for applying
   * the constraint (unused).
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const U_min_Matrix_Type &U_matrix, const U_Type &U,
                    std::size_t &set_count,
                    const std::size_t &initial_position) {
    // Do Nothing.
    static_cast<void>(M);
    static_cast<void>(gamma);
    static_cast<void>(U_matrix);
    static_cast<void>(U);
    static_cast<void>(set_count);
    static_cast<void>(initial_position);
  }
};

template <typename M_Type, typename Gamma_Type, typename U_min_Matrix_Type,
          typename U_Type, std::size_t I, std::size_t I_Dif>
struct Calculate_M_Gamma_U_Min_Loop {
  /**
   * @brief Applies constraints to the given matrices for a range of indices.
   *
   * This static function iterates over the indices from I to I_Dif, applying
   * constraints to the matrix M and vector gamma based on the U_matrix. It
   * updates the total_index with the number of constraints set and calls itself
   * recursively for the next index.
   *
   * @tparam M_Type Type of the matrix M.
   * @tparam Gamma_Type Type of the vector gamma.
   * @tparam U_min_Matrix_Type Type of the U minimum matrix.
   * @tparam U_Type Type of the U matrix.
   * @tparam I Current index in the iteration (template parameter).
   * @tparam I_Dif The difference in indices for recursion (template parameter).
   * @param M Reference to the matrix to be modified.
   * @param gamma Reference to the vector to be modified.
   * @param U_matrix Matrix providing the constraint value.
   * @param U Matrix providing additional values for gamma.
   * @param total_index Reference to the counter tracking the total number of
   * constraints set.
   * @param initial_position The starting position in M and gamma for applying
   * the constraints.
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const U_min_Matrix_Type &U_matrix, const U_Type &U,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {

    std::size_t set_count = static_cast<std::size_t>(0);

    Calculate_M_Gamma_U_Min_Condition<
        M_Type, Gamma_Type, U_min_Matrix_Type, U_Type, I,
        U_min_Matrix_Type::SparseAvailable_Type::lists[I][0]>::
        apply(M, gamma, U_matrix, U, set_count, initial_position);

    total_index += set_count;
    Calculate_M_Gamma_U_Min_Loop<M_Type, Gamma_Type, U_min_Matrix_Type, U_Type,
                                 (I + 1), (I_Dif - 1)>::apply(M, gamma,
                                                              U_matrix, U,
                                                              total_index,
                                                              initial_position);
  }
};

template <typename M_Type, typename Gamma_Type, typename U_min_Matrix_Type,
          typename U_Type, std::size_t I>
struct Calculate_M_Gamma_U_Min_Loop<M_Type, Gamma_Type, U_min_Matrix_Type,
                                    U_Type, I, 0> {
  /**
   * @brief A no-op function for the case when the loop has no more indices to
   * process.
   *
   * This static function does nothing and is used to terminate the recursion
   * in the template specialization.
   *
   * @tparam M_Type Type of the matrix M (unused).
   * @tparam Gamma_Type Type of the vector gamma (unused).
   * @tparam U_min_Matrix_Type Type of the U minimum matrix (unused).
   * @tparam U_Type Type of the U matrix (unused).
   * @param M Reference to the matrix (unused).
   * @param gamma Reference to the vector (unused).
   * @param U_matrix Matrix providing the constraint value (unused).
   * @param U Matrix providing additional values for gamma (unused).
   * @param total_index Reference to the counter tracking the total number of
   * constraints set (unused).
   * @param initial_position The starting position in M and gamma for applying
   * the constraints (unused).
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const U_min_Matrix_Type &U_matrix, const U_Type &U,
                    const std::size_t &total_index,
                    const std::size_t &initial_position) {
    // Do Nothing.
    static_cast<void>(M);
    static_cast<void>(gamma);
    static_cast<void>(U_matrix);
    static_cast<void>(U);
    static_cast<void>(total_index);
    static_cast<void>(initial_position);
  }
};

/**
 * @brief Alias template to calculate M gamma for U min constraints.
 *
 * This alias uses the Calculate_M_Gamma_U_Min_Loop metafunction to recursively
 * apply constraints for U min, starting from index 0 and iterating through the
 * specified U_Min_Size.
 *
 * @tparam M_Type Type of the matrix M.
 * @tparam Gamma_Type Type of the vector gamma.
 * @tparam U_min_Matrix_Type Type of the U minimum matrix.
 * @tparam U_Type Type of the U matrix.
 * @tparam U_Min_Size The size of the U constraints.
 */
template <typename M_Type, typename Gamma_Type, typename U_min_Matrix_Type,
          typename U_Type, std::size_t U_Min_Size>
using Calculate_M_Gamma_U_Min =
    Calculate_M_Gamma_U_Min_Loop<M_Type, Gamma_Type, U_min_Matrix_Type, U_Type,
                                 0, U_Min_Size>;

/* calculate M gamma for U max */

/**
 * @brief A utility structure to calculate M gamma for U max constraints.
 *
 * This structure defines a loop to apply constraints for U max, iterating
 * through the indices and applying conditions based on the flags defined in the
 * U_max_Matrix_Type.
 */
template <typename M_Type, typename Gamma_Type, typename U_max_Matrix_Type,
          typename U_Type, std::size_t I, bool Flag>
struct Calculate_M_Gamma_U_Max_Condition {};

template <typename M_Type, typename Gamma_Type, typename U_max_Matrix_Type,
          typename U_Type, std::size_t I>
struct Calculate_M_Gamma_U_Max_Condition<M_Type, Gamma_Type, U_max_Matrix_Type,
                                         U_Type, I, true> {
  /**
   * @brief Applies constraints to the given matrices for a specific index.
   *
   * This static function modifies the matrix M and vector gamma by applying a
   * constraint at the position specified by initial_position and the template
   * parameter I. It sets the corresponding entry in M to 1.0 and updates gamma
   * using the value from U_matrix at position <I, 0>. The set_count is
   * incremented to reflect the addition of a new constraint.
   *
   * @tparam I Index at which the constraint is applied (template parameter).
   * @tparam M_Type Type of the matrix M.
   * @tparam Gamma_Type Type of the vector gamma.
   * @tparam U_max_Matrix_Type Type of the U_max_matrix.
   * @tparam U_Type Type of the U matrix.
   * @param M Reference to the matrix to be modified.
   * @param gamma Reference to the vector to be modified.
   * @param U_matrix Matrix providing the constraint value.
   * @param U Matrix providing additional values for gamma.
   * @param set_count Reference to the counter tracking the number of
   * constraints set.
   * @param initial_position The starting position in M and gamma for applying
   * the constraint.
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const U_max_Matrix_Type &U_matrix, const U_Type &U,
                    std::size_t &set_count,
                    const std::size_t &initial_position) {

    M.access(initial_position + I, I) =
        static_cast<typename M_Type::Value_Type>(1.0);
    gamma.access(initial_position + I, 0) =
        U_matrix.template get<I, 0>() - U.template get<I, 0>();
    set_count += 1;
  }
};

template <typename M_Type, typename Gamma_Type, typename U_max_Matrix_Type,
          typename U_Type, std::size_t I>
struct Calculate_M_Gamma_U_Max_Condition<M_Type, Gamma_Type, U_max_Matrix_Type,
                                         U_Type, I, false> {
  /**
   * @brief A no-op function for the case when the condition is false.
   * This function does nothing and is used to maintain the structure of the
   * template specialization.
   * @param M Reference to the matrix (unused).
   * @param gamma Reference to the vector (unused).
   * @param U_matrix Matrix providing the constraint value (unused).
   * @param U Matrix providing additional values for gamma (unused).
   * @param set_count Reference to the counter tracking the number of
   * constraints set (unused).
   * @param initial_position The starting position in M and gamma for applying
   * the constraint (unused).
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const U_max_Matrix_Type &U_matrix, const U_Type &U,
                    std::size_t &set_count,
                    const std::size_t &initial_position) {
    // Do Nothing.
    static_cast<void>(M);
    static_cast<void>(gamma);
    static_cast<void>(U_matrix);
    static_cast<void>(U);
    static_cast<void>(set_count);
    static_cast<void>(initial_position);
  }
};

template <typename M_Type, typename Gamma_Type, typename U_max_Matrix_Type,
          typename U_Type, std::size_t I, std::size_t I_Dif>
struct Calculate_M_Gamma_U_Max_Loop {
  /**
   * @brief Applies constraints to the given matrices for a range of indices.
   *
   * This static function iterates over the indices from I to I_Dif, applying
   * constraints to the matrix M and vector gamma based on the U_matrix. It
   * updates the total_index with the number of constraints set and calls itself
   * recursively for the next index.
   *
   * @tparam M_Type Type of the matrix M.
   * @tparam Gamma_Type Type of the vector gamma.
   * @tparam U_max_Matrix_Type Type of the U maximum matrix.
   * @tparam U_Type Type of the U matrix.
   * @tparam I Current index in the iteration (template parameter).
   * @tparam I_Dif The difference in indices for recursion (template parameter).
   * @param M Reference to the matrix to be modified.
   * @param gamma Reference to the vector to be modified.
   * @param U_matrix Matrix providing the constraint value.
   * @param U Matrix providing additional values for gamma.
   * @param total_index Reference to the counter tracking the total number of
   * constraints set.
   * @param initial_position The starting position in M and gamma for applying
   * the constraints.
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const U_max_Matrix_Type &U_matrix, const U_Type &U,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {

    std::size_t set_count = static_cast<std::size_t>(0);

    Calculate_M_Gamma_U_Max_Condition<
        M_Type, Gamma_Type, U_max_Matrix_Type, U_Type, I,
        U_max_Matrix_Type::SparseAvailable_Type::lists[I][0]>::
        apply(M, gamma, U_matrix, U, set_count, initial_position);

    total_index += set_count;
    Calculate_M_Gamma_U_Max_Loop<M_Type, Gamma_Type, U_max_Matrix_Type, U_Type,
                                 (I + 1), (I_Dif - 1)>::apply(M, gamma,
                                                              U_matrix, U,
                                                              total_index,
                                                              initial_position);
  }
};

template <typename M_Type, typename Gamma_Type, typename U_max_Matrix_Type,
          typename U_Type, std::size_t I>
struct Calculate_M_Gamma_U_Max_Loop<M_Type, Gamma_Type, U_max_Matrix_Type,
                                    U_Type, I, 0> {
  /**
   * @brief A no-op function for the case when the loop has no more indices to
   * process.
   *
   * This static function does nothing and is used to terminate the recursion
   * in the template specialization.
   *
   * @tparam M_Type Type of the matrix M (unused).
   * @tparam Gamma_Type Type of the vector gamma (unused).
   * @tparam U_max_Matrix_Type Type of the U maximum matrix (unused).
   * @tparam U_Type Type of the U matrix (unused).
   * @param M Reference to the matrix (unused).
   * @param gamma Reference to the vector (unused).
   * @param U_matrix Matrix providing the constraint value (unused).
   * @param U Matrix providing additional values for gamma (unused).
   * @param total_index Reference to the counter tracking the total number of
   * constraints set (unused).
   * @param initial_position The starting position in M and gamma for applying
   * the constraints (unused).
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const U_max_Matrix_Type &U_matrix, const U_Type &U,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {
    // Do Nothing.
    static_cast<void>(M);
    static_cast<void>(gamma);
    static_cast<void>(U_matrix);
    static_cast<void>(U);
    static_cast<void>(total_index);
    static_cast<void>(initial_position);
  }
};

/**
 * @brief Alias template to calculate M gamma for U max constraints.
 *
 * This alias uses the Calculate_M_Gamma_U_Max_Loop metafunction to recursively
 * apply constraints for U max, starting from index 0 and iterating through the
 * specified U_Max_Size.
 *
 * @tparam M_Type Type of the matrix M.
 * @tparam Gamma_Type Type of the vector gamma.
 * @tparam U_max_Matrix_Type Type of the U maximum matrix.
 * @tparam U_Type Type of the U matrix.
 * @tparam U_Max_Size The size of the U constraints.
 */
template <typename M_Type, typename Gamma_Type, typename U_max_Matrix_Type,
          typename U_Type, std::size_t U_Max_Size>
using Calculate_M_Gamma_U_Max =
    Calculate_M_Gamma_U_Max_Loop<M_Type, Gamma_Type, U_max_Matrix_Type, U_Type,
                                 0, U_Max_Size>;

/* calculate M gamma for Y min */
template <typename M_Type, typename Phi_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I,
          std::size_t J, std::size_t J_Dif>
struct Set_M_Y_Min_Cols {
  /**
   * @brief Applies constraints to the matrix M based on the Phi matrix.
   *
   * This static function modifies the matrix M by setting specific entries
   * based on the values from the Phi matrix. It iterates through the columns
   * of the Phi matrix, applying constraints for each column until J_Dif is
   * reached.
   *
   * @tparam M_Type Type of the matrix M.
   * @tparam Phi_Type Type of the Phi matrix.
   * @tparam Y_Constraints_Prediction_Offset Offset for Y constraints in Phi.
   * @tparam I Current row index in M and Phi.
   * @tparam J Current column index in M and Phi.
   * @tparam J_Dif The difference in columns for recursion (template parameter).
   * @param M Reference to the matrix to be modified.
   * @param Phi Reference to the Phi matrix providing constraint values.
   * @param initial_position The starting position in M for applying the
   * constraints.
   */
  static void apply(M_Type &M, const Phi_Type &Phi,
                    std::size_t initial_position) {

    M.access(initial_position + I, J) =
        -Phi.template get<Y_Constraints_Prediction_Offset + I, J>();
    Set_M_Y_Min_Cols<M_Type, Phi_Type, Y_Constraints_Prediction_Offset, I,
                     J + 1, J_Dif - 1>::apply(M, Phi, initial_position);
  }
};

template <typename M_Type, typename Phi_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I,
          std::size_t J>
struct Set_M_Y_Min_Cols<M_Type, Phi_Type, Y_Constraints_Prediction_Offset, I, J,
                        0> {
  /**
   * @brief A no-op function for the case when there are no more columns to
   * process.
   *
   * This static function does nothing and is used to terminate the recursion
   * in the template specialization.
   *
   * @tparam M_Type Type of the matrix M (unused).
   * @tparam Phi_Type Type of the Phi matrix (unused).
   * @tparam Y_Constraints_Prediction_Offset Offset for Y constraints in Phi
   * (unused).
   * @tparam I Current row index in M and Phi (unused).
   * @tparam J Current column index in M and Phi (unused).
   * @param M Reference to the matrix (unused).
   * @param Phi Reference to the Phi matrix (unused).
   * @param initial_position The starting position in M for applying the
   * constraints (unused).
   */
  static void apply(M_Type &M, const Phi_Type &Phi,
                    std::size_t initial_position) {

    // Do Nothing.
    static_cast<void>(M);
    static_cast<void>(Phi);
    static_cast<void>(initial_position);
  }
};

template <typename M_Type, typename Gamma_Type, typename Y_min_Type,
          typename Phi_Type, typename F_X_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I, bool Flag>
struct Calculate_M_Gamma_Y_Min_Condition {};

template <typename M_Type, typename Gamma_Type, typename Y_min_Type,
          typename Phi_Type, typename F_X_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I>
struct Calculate_M_Gamma_Y_Min_Condition<
    M_Type, Gamma_Type, Y_min_Type, Phi_Type, F_X_Type,
    Y_Constraints_Prediction_Offset, I, true> {
  /**
   * @brief Applies constraints to the given matrices for a specific index.
   *
   * This static function modifies the matrix M and vector gamma by applying a
   * constraint at the position specified by initial_position and the template
   * parameter I. It sets the corresponding entry in M based on the Phi matrix
   * and updates gamma using the value from Y_min_matrix at position <I, 0>.
   * The set_count is incremented to reflect the addition of a new constraint.
   *
   * @tparam I Index at which the constraint is applied (template parameter).
   * @tparam M_Type Type of the matrix M.
   * @tparam Gamma_Type Type of the vector gamma.
   * @tparam Y_min_Type Type of the Y minimum matrix.
   * @tparam Phi_Type Type of the Phi matrix.
   * @tparam F_X_Type Type of the F_X matrix.
   * @param M Reference to the matrix to be modified.
   * @param gamma Reference to the vector to be modified.
   * @param Y_min_matrix Matrix providing the minimum Y values.
   * @param Phi Reference to the Phi matrix providing constraint values.
   * @param F_X Reference to the F_X matrix providing additional values for
   * gamma.
   * @param set_count Reference to the counter tracking the number of
   * constraints set.
   * @param initial_position The starting position in M and gamma for applying
   * the constraint.
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Y_min_Type &Y_min_matrix, const Phi_Type &Phi,
                    const F_X_Type &F_X, std::size_t &set_count,
                    std::size_t initial_position) {

    Set_M_Y_Min_Cols<M_Type, Phi_Type, Y_Constraints_Prediction_Offset, I, 0,
                     (Phi_Type::ROWS - 1)>::apply(M, Phi, initial_position);

    gamma.access(initial_position + I, 0) =
        -Y_min_matrix.template get<I, 0>() +
        F_X.template get<Y_Constraints_Prediction_Offset + I, 0>();
    set_count += 1;
  }
};

template <typename M_Type, typename Gamma_Type, typename Y_min_Type,
          typename Phi_Type, typename F_X_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I>
struct Calculate_M_Gamma_Y_Min_Condition<
    M_Type, Gamma_Type, Y_min_Type, Phi_Type, F_X_Type,
    Y_Constraints_Prediction_Offset, I, false> {
  /**
   * @brief A no-op function for the case when the condition is false.
   * This function does nothing and is used to maintain the structure of the
   * template specialization.
   * @param M Reference to the matrix (unused).
   * @param gamma Reference to the vector (unused).
   * @param Y_min_matrix Matrix providing the minimum Y values (unused).
   * @param Phi Reference to the Phi matrix providing constraint values
   * (unused).
   * @param F_X Reference to the F_X matrix providing additional values for
   * gamma (unused).
   * @param set_count Reference to the counter tracking the number of
   * constraints set (unused).
   * @param initial_position The starting position in M and gamma for applying
   * the constraint (unused).
   */
  static void apply(M_Type &M, Gamma_Type &gamma, const Y_min_Type Y_min_matrix,
                    const Phi_Type &Phi, const F_X_Type &F_X,
                    std::size_t &set_count, std::size_t initial_position) {

    // Do nothing
    static_cast<void>(M);
    static_cast<void>(gamma);
    static_cast<void>(Y_min_matrix);
    static_cast<void>(Phi);
    static_cast<void>(F_X);
    static_cast<void>(set_count);
    static_cast<void>(initial_position);
  }
};

template <typename M_Type, typename Gamma_Type, typename Y_min_Type,
          typename Phi_Type, typename F_X_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I,
          std::size_t I_Dif>
struct Calculate_M_Gamma_Y_Min_Loop {
  /**
   * @brief Applies constraints to the given matrices for a range of indices.
   *
   * This static function iterates over the indices from I to I_Dif, applying
   * constraints to the matrix M and vector gamma based on the Y_min_matrix and
   * Phi. It updates the total_index with the number of constraints set and
   * calls itself recursively for the next index.
   *
   * @tparam M_Type Type of the matrix M.
   * @tparam Gamma_Type Type of the vector gamma.
   * @tparam Y_min_Type Type of the Y minimum matrix.
   * @tparam Phi_Type Type of the Phi matrix.
   * @tparam F_X_Type Type of the F_X matrix.
   * @tparam Y_Constraints_Prediction_Offset Offset for Y constraints in Phi.
   * @tparam I Current index in the iteration (template parameter).
   * @tparam I_Dif The difference in indices for recursion (template parameter).
   * @param M Reference to the matrix to be modified.
   * @param gamma Reference to the vector to be modified.
   * @param Y_min_matrix Matrix providing the minimum Y values.
   * @param Phi Reference to the Phi matrix providing constraint values.
   * @param F_X Reference to the F_X matrix providing additional values for
   * gamma.
   * @param total_index Reference to the counter tracking the total number of
   * constraints set.
   * @param initial_position The starting position in M and gamma for applying
   * the constraints.
   */
  static void apply(M_Type &M, Gamma_Type &gamma, const Y_min_Type Y_min_matrix,
                    const Phi_Type &Phi, const F_X_Type &F_X,
                    std::size_t &total_index, std::size_t initial_position) {

    std::size_t set_count = 0;
    Calculate_M_Gamma_Y_Min_Condition<
        M_Type, Gamma_Type, Y_min_Type, Phi_Type, F_X_Type,
        Y_Constraints_Prediction_Offset, I,
        Y_min_Type::SparseAvailable_Type::lists[I][0]>::apply(M, gamma,
                                                              Y_min_matrix, Phi,
                                                              F_X, set_count,
                                                              initial_position);
    total_index += set_count;
    Calculate_M_Gamma_Y_Min_Loop<M_Type, Gamma_Type, Y_min_Type, Phi_Type,
                                 F_X_Type, Y_Constraints_Prediction_Offset,
                                 I + 1, I_Dif - 1>::apply(M, gamma,
                                                          Y_min_matrix, Phi,
                                                          F_X, total_index,
                                                          initial_position);
  }
};

template <typename M_Type, typename Gamma_Type, typename Y_min_Type,
          typename Phi_Type, typename F_X_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I>
struct Calculate_M_Gamma_Y_Min_Loop<M_Type, Gamma_Type, Y_min_Type, Phi_Type,
                                    F_X_Type, Y_Constraints_Prediction_Offset,
                                    I, 0> {
  /**
   * @brief A no-op function for the case when the loop has no more indices to
   * process.
   *
   * This static function does nothing and is used to terminate the recursion
   * in the template specialization.
   *
   * @tparam M_Type Type of the matrix M (unused).
   * @tparam Gamma_Type Type of the vector gamma (unused).
   * @tparam Y_min_Type Type of the Y minimum matrix (unused).
   * @tparam Phi_Type Type of the Phi matrix (unused).
   * @tparam F_X_Type Type of the F_X matrix (unused).
   * @param M Reference to the matrix (unused).
   * @param gamma Reference to the vector (unused).
   * @param Y_min_matrix Matrix providing the minimum Y values (unused).
   * @param Phi Reference to the Phi matrix providing constraint values
   * (unused).
   * @param F_X Reference to the F_X matrix providing additional values for
   * gamma (unused).
   * @param total_index Reference to the counter tracking the total number of
   * constraints set (unused).
   * @param initial_position The starting position in M and gamma for applying
   * the constraints (unused).
   */
  static void apply(M_Type &M, Gamma_Type &gamma, const Y_min_Type Y_min_matrix,
                    const Phi_Type &Phi, const F_X_Type &F_X,
                    std::size_t &total_index, std::size_t initial_position) {

    // Do nothing
    static_cast<void>(M);
    static_cast<void>(gamma);
    static_cast<void>(Y_min_matrix);
    static_cast<void>(Phi);
    static_cast<void>(F_X);
    static_cast<void>(total_index);
    static_cast<void>(initial_position);
  }
};

/**
 * @brief Alias template to calculate M gamma for Y min constraints.
 *
 * This alias uses the Calculate_M_Gamma_Y_Min_Loop metafunction to recursively
 * apply constraints for Y min, starting from index 0 and iterating through the
 * specified Y_Min_Size.
 *
 * @tparam M_Type Type of the matrix M.
 * @tparam Gamma_Type Type of the vector gamma.
 * @tparam Y_min_Type Type of the Y minimum matrix.
 * @tparam Phi_Type Type of the Phi matrix.
 * @tparam F_X_Type Type of the F_X matrix.
 * @tparam Y_Constraints_Prediction_Offset Offset for Y constraints in Phi.
 * @tparam Y_Min_Size The size of the Y constraints.
 */
template <typename M_Type, typename Gamma_Type, typename Y_min_Type,
          typename Phi_Type, typename F_X_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t Y_Min_Size>
using Calculate_M_Gamma_Y_Min =
    Calculate_M_Gamma_Y_Min_Loop<M_Type, Gamma_Type, Y_min_Type, Phi_Type,
                                 F_X_Type, Y_Constraints_Prediction_Offset, 0,
                                 Y_Min_Size>;

/* calculate M gamma for Y max */
template <typename M_Type, typename Phi_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I,
          std::size_t J, std::size_t J_Dif>
struct Set_M_Y_Max_Cols {
  /**
   * @brief Applies constraints to the matrix M based on the Phi matrix.
   *
   * This static function modifies the matrix M by setting specific entries
   * based on the values from the Phi matrix. It iterates through the columns
   * of the Phi matrix, applying constraints for each column until J_Dif is
   * reached.
   *
   * @tparam M_Type Type of the matrix M.
   * @tparam Phi_Type Type of the Phi matrix.
   * @tparam Y_Constraints_Prediction_Offset Offset for Y constraints in Phi.
   * @tparam I Current row index in M and Phi.
   * @tparam J Current column index in M and Phi.
   * @tparam J_Dif The difference in columns for recursion (template parameter).
   * @param M Reference to the matrix to be modified.
   * @param Phi Reference to the Phi matrix providing constraint values.
   * @param initial_position The starting position in M for applying the
   * constraints.
   */
  static void apply(M_Type &M, const Phi_Type &Phi,
                    std::size_t initial_position) {

    M.access(initial_position + I, J) =
        Phi.template get<Y_Constraints_Prediction_Offset + I, J>();
    Set_M_Y_Max_Cols<M_Type, Phi_Type, Y_Constraints_Prediction_Offset, I,
                     J + 1, J_Dif - 1>::apply(M, Phi, initial_position);
  }
};

template <typename M_Type, typename Phi_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I,
          std::size_t J>
struct Set_M_Y_Max_Cols<M_Type, Phi_Type, Y_Constraints_Prediction_Offset, I, J,
                        0> {
  /**
   * @brief A no-op function for the case when there are no more columns to
   * process.
   *
   * This static function does nothing and is used to terminate the recursion
   * in the template specialization.
   *
   * @tparam M_Type Type of the matrix M (unused).
   * @tparam Phi_Type Type of the Phi matrix (unused).
   * @tparam Y_Constraints_Prediction_Offset Offset for Y constraints in Phi
   * (unused).
   * @tparam I Current row index in M and Phi (unused).
   * @tparam J Current column index in M and Phi (unused).
   * @param M Reference to the matrix (unused).
   * @param Phi Reference to the Phi matrix (unused).
   * @param initial_position The starting position in M for applying the
   * constraints (unused).
   */
  static void apply(M_Type &M, const Phi_Type &Phi,
                    std::size_t initial_position) {

    // Do Nothing.
    static_cast<void>(M);
    static_cast<void>(Phi);
    static_cast<void>(initial_position);
  }
};

template <typename M_Type, typename Gamma_Type, typename Y_max_Type,
          typename Phi_Type, typename F_X_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I, bool Flag>
struct Calculate_M_Gamma_Y_Max_Condition {};

template <typename M_Type, typename Gamma_Type, typename Y_max_Type,
          typename Phi_Type, typename F_X_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I>
struct Calculate_M_Gamma_Y_Max_Condition<
    M_Type, Gamma_Type, Y_max_Type, Phi_Type, F_X_Type,
    Y_Constraints_Prediction_Offset, I, true> {
  /**
   * @brief Applies constraints to the given matrices for a specific index.
   *
   * This static function modifies the matrix M and vector gamma by applying a
   * constraint at the position specified by initial_position and the template
   * parameter I. It sets the corresponding entry in M based on the Phi matrix
   * and updates gamma using the value from Y_max_matrix at position <I, 0>.
   * The set_count is incremented to reflect the addition of a new constraint.
   *
   * @tparam I Index at which the constraint is applied (template parameter).
   * @tparam M_Type Type of the matrix M.
   * @tparam Gamma_Type Type of the vector gamma.
   * @tparam Y_max_Type Type of the Y maximum matrix.
   * @tparam Phi_Type Type of the Phi matrix.
   * @tparam F_X_Type Type of the F_X matrix.
   * @param M Reference to the matrix to be modified.
   * @param gamma Reference to the vector to be modified.
   * @param Y_max_matrix Matrix providing the maximum Y values.
   * @param Phi Reference to the Phi matrix providing constraint values.
   * @param F_X Reference to the F_X matrix providing additional values for
   * gamma.
   * @param set_count Reference to the counter tracking the number of
   * constraints set.
   * @param initial_position The starting position in M and gamma for applying
   * the constraint.
   */
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Y_max_Type &Y_max_matrix, const Phi_Type &Phi,
                    const F_X_Type &F_X, std::size_t &set_count,
                    std::size_t initial_position) {

    Set_M_Y_Max_Cols<M_Type, Phi_Type, Y_Constraints_Prediction_Offset, I, 0,
                     (Phi_Type::ROWS - 1)>::apply(M, Phi, initial_position);

    gamma.access(initial_position + I, 0) =
        Y_max_matrix.template get<I, 0>() -
        F_X.template get<Y_Constraints_Prediction_Offset + I, 0>();
    set_count += 1;
  }
};

template <typename M_Type, typename Gamma_Type, typename Y_max_Type,
          typename Phi_Type, typename F_X_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I>
struct Calculate_M_Gamma_Y_Max_Condition<
    M_Type, Gamma_Type, Y_max_Type, Phi_Type, F_X_Type,
    Y_Constraints_Prediction_Offset, I, false> {
  /**
   * @brief A no-op function for the case when the condition is false.
   * This function does nothing and is used to maintain the structure of the
   * template specialization.
   * @param M Reference to the matrix (unused).
   * @param gamma Reference to the vector (unused).
   * @param Y_max_matrix Matrix providing the maximum Y values (unused).
   * @param Phi Reference to the Phi matrix providing constraint values
   * (unused).
   * @param F_X Reference to the F_X matrix providing additional values for
   * gamma (unused).
   * @param set_count Reference to the counter tracking the number of
   * constraints set (unused).
   * @param initial_position The starting position in M and gamma for applying
   * the constraint (unused).
   */
  static void apply(M_Type &M, Gamma_Type &gamma, const Y_max_Type Y_max_matrix,
                    const Phi_Type &Phi, const F_X_Type &F_X,
                    std::size_t &set_count, std::size_t initial_position) {

    // Do nothing
    static_cast<void>(M);
    static_cast<void>(gamma);
    static_cast<void>(Y_max_matrix);
    static_cast<void>(Phi);
    static_cast<void>(F_X);
    static_cast<void>(set_count);
    static_cast<void>(initial_position);
  }
};

template <typename M_Type, typename Gamma_Type, typename Y_max_Type,
          typename Phi_Type, typename F_X_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I,
          std::size_t I_Dif>
struct Calculate_M_Gamma_Y_Max_Loop {
  /**
   * @brief Applies constraints to the given matrices for a range of indices.
   *
   * This static function iterates over the indices from I to I_Dif, applying
   * constraints to the matrix M and vector gamma based on the Y_max_matrix and
   * Phi. It updates the total_index with the number of constraints set and
   * calls itself recursively for the next index.
   *
   * @tparam M_Type Type of the matrix M.
   * @tparam Gamma_Type Type of the vector gamma.
   * @tparam Y_max_Type Type of the Y maximum matrix.
   * @tparam Phi_Type Type of the Phi matrix.
   * @tparam F_X_Type Type of the F_X matrix.
   * @tparam Y_Constraints_Prediction_Offset Offset for Y constraints in Phi.
   * @tparam I Current index in the iteration (template parameter).
   * @tparam I_Dif The difference in indices for recursion (template parameter).
   * @param M Reference to the matrix to be modified.
   * @param gamma Reference to the vector to be modified.
   * @param Y_max_matrix Matrix providing the maximum Y values.
   * @param Phi Reference to the Phi matrix providing constraint values.
   * @param F_X Reference to the F_X matrix providing additional values for
   * gamma.
   * @param total_index Reference to the counter tracking the total number of
   * constraints set.
   * @param initial_position The starting position in M and gamma for applying
   * the constraints.
   */
  static void apply(M_Type &M, Gamma_Type &gamma, const Y_max_Type Y_max_matrix,
                    const Phi_Type &Phi, const F_X_Type &F_X,
                    std::size_t &total_index, std::size_t initial_position) {

    std::size_t set_count = 0;
    Calculate_M_Gamma_Y_Max_Condition<
        M_Type, Gamma_Type, Y_max_Type, Phi_Type, F_X_Type,
        Y_Constraints_Prediction_Offset, I,
        Y_max_Type::SparseAvailable_Type::lists[I][0]>::apply(M, gamma,
                                                              Y_max_matrix, Phi,
                                                              F_X, set_count,
                                                              initial_position);
    total_index += set_count;
    Calculate_M_Gamma_Y_Max_Loop<M_Type, Gamma_Type, Y_max_Type, Phi_Type,
                                 F_X_Type, Y_Constraints_Prediction_Offset,
                                 I + 1, I_Dif - 1>::apply(M, gamma,
                                                          Y_max_matrix, Phi,
                                                          F_X, total_index,
                                                          initial_position);
  }
};

template <typename M_Type, typename Gamma_Type, typename Y_max_Type,
          typename Phi_Type, typename F_X_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t I>
struct Calculate_M_Gamma_Y_Max_Loop<M_Type, Gamma_Type, Y_max_Type, Phi_Type,
                                    F_X_Type, Y_Constraints_Prediction_Offset,
                                    I, 0> {
  /**
   * @brief A no-op function for the case when the loop has no more indices to
   * process.
   *
   * This static function does nothing and is used to terminate the recursion
   * in the template specialization.
   *
   * @tparam M_Type Type of the matrix M (unused).
   * @tparam Gamma_Type Type of the vector gamma (unused).
   * @tparam Y_max_Type Type of the Y maximum matrix (unused).
   * @tparam Phi_Type Type of the Phi matrix (unused).
   * @tparam F_X_Type Type of the F_X matrix (unused).
   * @param M Reference to the matrix (unused).
   * @param gamma Reference to the vector (unused).
   * @param Y_max_matrix Matrix providing the maximum Y values (unused).
   * @param Phi Reference to the Phi matrix providing constraint values
   * (unused).
   * @param F_X Reference to the F_X matrix providing additional values for
   * gamma (unused).
   * @param total_index Reference to the counter tracking the total number of
   * constraints set (unused).
   * @param initial_position The starting position in M and gamma for applying
   * the constraints (unused).
   */
  static void apply(M_Type &M, Gamma_Type &gamma, const Y_max_Type Y_max_matrix,
                    const Phi_Type &Phi, const F_X_Type &F_X,
                    std::size_t &total_index, std::size_t initial_position) {

    // Do nothing
    static_cast<void>(M);
    static_cast<void>(gamma);
    static_cast<void>(Y_max_matrix);
    static_cast<void>(Phi);
    static_cast<void>(F_X);
    static_cast<void>(total_index);
    static_cast<void>(initial_position);
  }
};

/**
 * @brief Alias template to calculate M gamma for Y max constraints.
 *
 * This alias uses the Calculate_M_Gamma_Y_Max_Loop metafunction to recursively
 * apply constraints for Y max, starting from index 0 and iterating through the
 * specified Y_Max_Size.
 *
 * @tparam M_Type Type of the matrix M.
 * @tparam Gamma_Type Type of the vector gamma.
 * @tparam Y_max_Type Type of the Y maximum matrix.
 * @tparam Phi_Type Type of the Phi matrix.
 * @tparam F_X_Type Type of the F_X matrix.
 * @tparam Y_Constraints_Prediction_Offset Offset for Y constraints in Phi.
 * @tparam Y_Max_Size The size of the Y constraints.
 */
template <typename M_Type, typename Gamma_Type, typename Y_max_Type,
          typename Phi_Type, typename F_X_Type,
          std::size_t Y_Constraints_Prediction_Offset, std::size_t Y_Max_Size>
using Calculate_M_Gamma_Y_Max =
    Calculate_M_Gamma_Y_Max_Loop<M_Type, Gamma_Type, Y_max_Type, Phi_Type,
                                 F_X_Type, Y_Constraints_Prediction_Offset, 0,
                                 Y_Max_Size>;

} // namespace LMPC_QP_SolverOperation

/* LTI MPC QP solver */

/**
 * @brief Class template for LTI MPC QP Solver.
 *
 * This class template implements a solver for Linear Time-Invariant Model
 * Predictive Control (LTI MPC) using Quadratic Programming (QP). It handles
 * constraints on control inputs and outputs, and provides methods to update
 * constraints and solve the QP problem.
 *
 * @tparam Number_Of_Variables Number of control variables.
 * @tparam Output_Size Size of the output vector.
 * @tparam U_Type Type representing the control input matrix.
 * @tparam X_augmented_Type Type representing the augmented state vector.
 * @tparam Phi_Type Type representing the prediction matrix.
 * @tparam F_Type Type representing the system dynamics matrix.
 * @tparam Weight_U_Nc_Type Type representing the weight for control input
 * changes.
 * @tparam Delta_U_Min_Type Type representing the minimum change in control
 * input.
 * @tparam Delta_U_Max_Type Type representing the maximum change in control
 * input.
 * @tparam U_Min_Type Type representing the minimum control input limits.
 * @tparam U_Max_Type Type representing the maximum control input limits.
 * @tparam Y_Min_Type Type representing the minimum output limits.
 * @tparam Y_Max_Type Type representing the maximum output limits.
 * @tparam Y_Constraints_Prediction_Offset Offset for Y constraints in
 * prediction matrix (default is 0).
 */
template <std::size_t Number_Of_Variables, std::size_t Output_Size,
          typename U_Type, typename X_augmented_Type, typename Phi_Type,
          typename F_Type, typename Weight_U_Nc_Type, typename Delta_U_Min_Type,
          typename Delta_U_Max_Type, typename U_Min_Type, typename U_Max_Type,
          typename Y_Min_Type, typename Y_Max_Type,
          std::size_t Y_Constraints_Prediction_Offset = 0>
class LMPC_QP_Solver {
public:
  /* Type */
  using Value_Type = typename U_Type::Value_Type;

  using Limits_Type =
      DU_U_Y_Limits_Type<Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
                         U_Max_Type, Y_Min_Type, Y_Max_Type>;

  static constexpr std::size_t NUMBER_OF_ALL_CONSTRAINTS =
      Limits_Type::NUMBER_OF_ALL_CONSTRAINTS;

  using M_Type =
      PythonNumpy::DenseMatrix_Type<Value_Type, NUMBER_OF_ALL_CONSTRAINTS,
                                    Number_Of_Variables>;

  using Gamma_Type =
      PythonNumpy::DenseMatrix_Type<Value_Type, NUMBER_OF_ALL_CONSTRAINTS, 1>;

  using U_Nc_Type =
      PythonNumpy::DenseMatrix_Type<Value_Type, Number_Of_Variables, 1>;

  static constexpr std::size_t Y_CONSTRAINTS_PREDICTION_OFFSET =
      Y_Constraints_Prediction_Offset;

  /* Check Compatibility */
  static_assert(
      std::is_same<typename X_augmented_Type::Value_Type, Value_Type>::value,
      "X_augmented_Type::Value_Type must be equal to Value_Type");
  static_assert(std::is_same<typename Phi_Type::Value_Type, Value_Type>::value,
                "Phi_Type::Value_Type must be equal to Value_Type");
  static_assert(std::is_same<typename F_Type::Value_Type, Value_Type>::value,
                "F_Type::Value_Type must be equal to Value_Type");

  static_assert(
      std::is_same<typename Delta_U_Min_Type::Value_Type, Value_Type>::value,
      "Delta_U_Min_Type::Value_Type must be equal to Value_Type");
  static_assert(
      std::is_same<typename Delta_U_Max_Type::Value_Type, Value_Type>::value,
      "Delta_U_Max_Type::Value_Type must be equal to Value_Type");
  static_assert(
      std::is_same<typename U_Min_Type::Value_Type, Value_Type>::value,
      "U_Min_Type::Value_Type must be equal to Value_Type");
  static_assert(
      std::is_same<typename U_Max_Type::Value_Type, Value_Type>::value,
      "U_Max_Type::Value_Type must be equal to Value_Type");
  static_assert(
      std::is_same<typename Y_Min_Type::Value_Type, Value_Type>::value,
      "Y_Min_Type::Value_Type must be equal to Value_Type");
  static_assert(
      std::is_same<typename Y_Max_Type::Value_Type, Value_Type>::value,
      "Y_Max_Type::Value_Type must be equal to Value_Type");

  static_assert(U_Type::ROWS == static_cast<std::size_t>(1),
                "U_Type must be a row vector");

  static_assert(X_augmented_Type::COLS == F_Type::ROWS &&
                    X_augmented_Type::ROWS == static_cast<std::size_t>(1),
                "X_augmented_Type size doesn't match F_Type size");

  static_assert(Phi_Type::COLS == F_Type::COLS &&
                    Phi_Type::ROWS == Number_Of_Variables,
                "Phi_Type size doesn't match F_Type size");

protected:
  /* Type */
  using _T = Value_Type;

  using _E_Empty_Type =
      PythonNumpy::SparseMatrixEmpty_Type<_T, Number_Of_Variables,
                                          Number_Of_Variables>;

  using _Solver_Type = PythonOptimization::QP_ActiveSetSolver_Type<
      Value_Type, Number_Of_Variables, NUMBER_OF_ALL_CONSTRAINTS>;

public:
  /* Constructor */
  LMPC_QP_Solver() : limits(), M(), gamma(), _solver() {}

  LMPC_QP_Solver(const U_Type &U_in, const X_augmented_Type &X_augmented_in,
                 const Phi_Type &Phi_in, const F_Type &F_in,
                 const Weight_U_Nc_Type &weight_U_Nc_in,
                 const Delta_U_Min_Type &delta_U_Min_in,
                 const Delta_U_Max_Type &delta_U_Max_in,
                 const U_Min_Type &U_min_in, const U_Max_Type &U_max_in,
                 const Y_Min_Type &Y_min_in, const Y_Max_Type &Y_max_in)
      : limits(), M(), gamma(), _solver() {

    this->limits = make_DU_U_Y_Limits(delta_U_Min_in, delta_U_Max_in, U_min_in,
                                      U_max_in, Y_min_in, Y_max_in);

    this->update_constraints(U_in, X_augmented_in, Phi_in, F_in);

    this->_solver.set_max_iteration(
        PythonMPC::SolverUtility::MAX_ITERATION_DEFAULT);
    this->_solver.set_tol(
        static_cast<_T>(PythonMPC::SolverUtility::TOL_DEFAULT));

    // this->_solver.set_kkt_inv_solver_division_min(static_cast<_T>(1.0e-5));

    this->update_E(Phi_in, weight_U_Nc_in);
  }

  /* Copy Constructor */
  LMPC_QP_Solver(const LMPC_QP_Solver &other)
      : limits(other.limits), M(other.M), gamma(other.gamma),
        _solver(other._solver) {}

  LMPC_QP_Solver<Number_Of_Variables, Output_Size, U_Type, X_augmented_Type,
                 Phi_Type, F_Type, Weight_U_Nc_Type, Delta_U_Min_Type,
                 Delta_U_Max_Type, U_Min_Type, U_Max_Type, Y_Min_Type,
                 Y_Max_Type, Y_Constraints_Prediction_Offset> &
  operator=(const LMPC_QP_Solver &other) {
    if (this != &other) {
      this->limits = other.limits;
      this->M = other.M;
      this->gamma = other.gamma;
      this->_solver = other._solver;
    }
    return *this;
  }

  /* Move Constructor */
  LMPC_QP_Solver(LMPC_QP_Solver &&other) noexcept
      : limits(std::move(other.limits)), M(std::move(other.M)),
        gamma(std::move(other.gamma)), _solver(std::move(other._solver)) {}

  LMPC_QP_Solver<Number_Of_Variables, Output_Size, U_Type, X_augmented_Type,
                 Phi_Type, F_Type, Weight_U_Nc_Type, Delta_U_Min_Type,
                 Delta_U_Max_Type, U_Min_Type, U_Max_Type, Y_Min_Type,
                 Y_Max_Type, Y_Constraints_Prediction_Offset> &
  operator=(LMPC_QP_Solver &&other) noexcept {
    if (this != &other) {
      this->limits = std::move(other.limits);
      this->M = std::move(other.M);
      this->gamma = std::move(other.gamma);
      this->_solver = std::move(other._solver);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Updates the constraints based on the provided inputs.
   *
   * This function calculates and updates the constraints for the MPC QP solver
   * based on the input control vector U, augmented state vector X_augmented,
   * prediction matrix Phi, and system dynamics matrix F.
   *
   * @param U_in Input control vector.
   * @param X_augmented_in Augmented state vector.
   * @param Phi_in Prediction matrix.
   * @param F_in System dynamics matrix.
   */
  template <std::size_t N = NUMBER_OF_ALL_CONSTRAINTS>
  inline typename std::enable_if<(N > 0), void>::type
  update_constraints(const U_Type &U_in, const X_augmented_Type &X_augmented_in,
                     const Phi_Type &Phi_in, const F_Type &F_in) {

    std::size_t total_index = static_cast<std::size_t>(0);

    this->_calculate_M_gamma_delta_U(total_index);

    this->_calculate_M_gamma_U(total_index, U_in);

    this->_calculate_M_gamma_Y(total_index, X_augmented_in, Phi_in, F_in);
  }

  /**
   * @brief A no-op function for the case when there are no constraints to
   * update.
   *
   * This function does nothing and is used to maintain the structure of the
   * template specialization when there are no constraints to update.
   *
   * @param U_in Input control vector (unused).
   * @param X_augmented_in Augmented state vector (unused).
   * @param Phi_in Prediction matrix (unused).
   * @param F_in System dynamics matrix (unused).
   */
  template <std::size_t N = NUMBER_OF_ALL_CONSTRAINTS>
  inline typename std::enable_if<(N == 0), void>::type
  update_constraints(const U_Type &U_in, const X_augmented_Type &X_augmented_in,
                     const Phi_Type &Phi_in, const F_Type &F_in) {

    // Do Nothing.
    static_cast<void>(U_in);
    static_cast<void>(X_augmented_in);
    static_cast<void>(Phi_in);
    static_cast<void>(F_in);
  }

  /**
   * @brief Updates the E matrix used in the QP solver.
   *
   * This function updates the E matrix by calculating the product of the
   * transpose of Phi and Phi, and adding the provided weight matrix for control
   * input changes.
   *
   * @param Phi The prediction matrix.
   * @param Weight_U_Nc The weight matrix for control input changes.
   */
  inline void update_E(const Phi_Type &Phi,
                       const Weight_U_Nc_Type &Weight_U_Nc) {

    this->_solver.update_E(PythonNumpy::ATranspose_mul_B(Phi, Phi) +
                           Weight_U_Nc);
  }

  /**
   * @brief Returns the number of constraints for delta U.
   *
   * This function returns the number of constraints related to the change in
   * control input (delta U).
   *
   * @return The number of delta U constraints.
   */
  inline auto get_number_of_Y_constraints_prediction_offset(void) const
      -> std::size_t {
    return this->_Y_constraints_prediction_offset;
  }

  /**
   * @brief Returns the number of constraints for delta U.
   *
   * This function returns the number of constraints related to the change in
   * control input (delta U).
   *
   * @return The number of delta U constraints.
   */
  template <typename ReferenceTrajectoryType>
  inline auto solve(const Phi_Type &Phi, const F_Type &F,
                    ReferenceTrajectoryType &reference_trajectory,
                    const X_augmented_Type &X_augmented) -> U_Nc_Type {

    auto L = PythonNumpy::ATranspose_mul_B(
        Phi, reference_trajectory.calculate_dif(F * X_augmented));

    auto x_opt = this->_solver.solve(_E_Empty_Type{}, L, this->M, this->gamma);

    return x_opt;
  }

  /**
   * @brief Solves the QP problem with the given parameters.
   *
   * This function solves the QP problem using the provided prediction matrix
   * Phi, system dynamics matrix F, reference trajectory, augmented state
   * vector, and weight for control input changes.
   *
   * @param Phi The prediction matrix.
   * @param F The system dynamics matrix.
   * @param reference_trajectory The reference trajectory object.
   * @param X_augmented The augmented state vector.
   * @param Weight_U_Nc The weight matrix for control input changes.
   * @return The optimal control input vector U_Nc.
   */
  template <typename ReferenceTrajectoryType>
  inline auto solve(const Phi_Type &Phi, const F_Type &F,
                    const ReferenceTrajectoryType &reference_trajectory,
                    const X_augmented_Type &X_augmented,
                    const Weight_U_Nc_Type &Weight_U_Nc) -> U_Nc_Type {

    auto L = PythonNumpy::ATranspose_mul_B(
        Phi, reference_trajectory.calculate_dif(F * X_augmented));

    this->update_E(Phi, Weight_U_Nc);

    auto x_opt = this->_solver.solve(_E_Empty_Type{}, L, this->M, this->gamma);

    return x_opt;
  }

protected:
  /* Function */

  /**
   * @brief Calculates the M and gamma matrices for delta U and U constraints.
   *
   * This function calculates the M and gamma matrices based on the provided
   * limits for delta U and U, updating the total index for constraints.
   *
   * @param total_index Reference to the counter tracking the total number of
   * constraints set.
   */
  inline void _calculate_M_gamma_delta_U(std::size_t &total_index) {
    std::size_t initial_position = total_index;

    // delta_U_min constraints
    LMPC_QP_SolverOperation::Calculate_M_Gamma_Delta_U_Min<
        M_Type, Gamma_Type, decltype(this->limits.delta_U_min),
        Limits_Type::DELTA_U_MIN_SIZE>::apply(this->M, this->gamma,
                                              this->limits.delta_U_min,
                                              total_index, initial_position);

    initial_position = total_index;

    // delta_U_max constraints
    LMPC_QP_SolverOperation::Calculate_M_Gamma_Delta_U_Max<
        M_Type, Gamma_Type, decltype(this->limits.delta_U_max),
        Limits_Type::DELTA_U_MAX_SIZE>::apply(this->M, this->gamma,
                                              this->limits.delta_U_max,
                                              total_index, initial_position);
  }

  /**
   * @brief Calculates the M and gamma matrices for U constraints.
   *
   * This function calculates the M and gamma matrices based on the provided
   * limits for U, updating the total index for constraints.
   *
   * @param total_index Reference to the counter tracking the total number of
   * constraints set.
   * @param U The control input vector.
   */
  inline void _calculate_M_gamma_U(std::size_t &total_index, const U_Type &U) {
    std::size_t initial_position = total_index;

    // U_min constraints
    LMPC_QP_SolverOperation::Calculate_M_Gamma_U_Min<
        M_Type, Gamma_Type, decltype(this->limits.U_min), U_Type,
        Limits_Type::U_MIN_SIZE>::apply(this->M, this->gamma,
                                        this->limits.U_min, U, total_index,
                                        initial_position);

    initial_position = total_index;

    // U_max constraints
    LMPC_QP_SolverOperation::Calculate_M_Gamma_U_Max<
        M_Type, Gamma_Type, decltype(this->limits.U_max), U_Type,
        Limits_Type::U_MAX_SIZE>::apply(this->M, this->gamma,
                                        this->limits.U_max, U, total_index,
                                        initial_position);
  }

  /**
   * @brief Calculates the M and gamma matrices for Y constraints.
   *
   * This function calculates the M and gamma matrices based on the provided
   * augmented state vector, prediction matrix, and system dynamics matrix,
   * updating the total index for constraints.
   *
   * @param total_index Reference to the counter tracking the total number of
   * constraints set.
   * @param X_augmented The augmented state vector.
   * @param Phi The prediction matrix.
   * @param F The system dynamics matrix.
   */
  inline void _calculate_M_gamma_Y(std::size_t &total_index,
                                   const X_augmented_Type &X_augmented,
                                   const Phi_Type &Phi, const F_Type &F) {

    std::size_t initial_position = total_index;
    auto F_X = F * X_augmented;

    // Y_min constraints
    LMPC_QP_SolverOperation::Calculate_M_Gamma_Y_Min<
        M_Type, Gamma_Type, decltype(this->limits.Y_min), Phi_Type,
        decltype(F_X), Y_CONSTRAINTS_PREDICTION_OFFSET,
        Limits_Type::Y_MIN_SIZE>::apply(this->M, this->gamma,
                                        this->limits.Y_min, Phi, F_X,
                                        total_index, initial_position);

    initial_position = total_index;

    // Y_max constraints
    LMPC_QP_SolverOperation::Calculate_M_Gamma_Y_Max<
        M_Type, Gamma_Type, decltype(this->limits.Y_max), Phi_Type,
        decltype(F_X), Y_CONSTRAINTS_PREDICTION_OFFSET,
        Limits_Type::Y_MAX_SIZE>::apply(this->M, this->gamma,
                                        this->limits.Y_max, Phi, F_X,
                                        total_index, initial_position);
  }

public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_VARIABLES = Number_Of_Variables;
  static constexpr std::size_t U_SIZE = U_Type::COLS;
  static constexpr std::size_t Y_SIZE = Output_Size;

public:
  /* Variable */
  Limits_Type limits;

  M_Type M;
  Gamma_Type gamma;

protected:
  /* Variable */
  _Solver_Type _solver;
};

/* make LMPC_QP_Solver */

/**
 * @brief Factory function to create an instance of LMPC_QP_Solver.
 *
 * This function initializes and returns an instance of LMPC_QP_Solver with
 * the provided parameters.
 *
 * @tparam Number_Of_Variables Number of control variables.
 * @tparam Output_Size Size of the output vector.
 * @tparam U_Type Type representing the control input matrix.
 * @tparam X_augmented_Type Type representing the augmented state vector.
 * @tparam Phi_Type Type representing the prediction matrix.
 * @tparam F_Type Type representing the system dynamics matrix.
 * @tparam Weight_U_Nc_Type Type representing the weight for control input
 * changes.
 * @tparam Delta_U_Min_Type Type representing the minimum change in control
 * input.
 * @tparam Delta_U_Max_Type Type representing the maximum change in control
 * input.
 * @tparam U_Min_Type Type representing the minimum control input limits.
 * @tparam U_Max_Type Type representing the maximum control input limits.
 * @tparam Y_Min_Type Type representing the minimum output limits.
 * @tparam Y_Max_Type Type representing the maximum output limits.
 * @return An instance of LMPC_QP_Solver initialized with the provided
 * parameters.
 */
template <std::size_t Number_Of_Variables, std::size_t Output_Size,
          typename U_Type, typename X_augmented_Type, typename Phi_Type,
          typename F_Type, typename Weight_U_Nc_Type, typename Delta_U_Min_Type,
          typename Delta_U_Max_Type, typename U_Min_Type, typename U_Max_Type,
          typename Y_Min_Type, typename Y_Max_Type,
          std::size_t Y_Constraints_Prediction_Offset = 0>
inline auto
make_LMPC_QP_Solver(const U_Type &U_in, const X_augmented_Type &X_augmented_in,
                    const Phi_Type &Phi_in, const F_Type &F_in,
                    const Weight_U_Nc_Type &weight_U_Nc_in,
                    const Delta_U_Min_Type &delta_U_Min_in,
                    const Delta_U_Max_Type &delta_U_Max_in,
                    const U_Min_Type &U_min_in, const U_Max_Type &U_max_in,
                    const Y_Min_Type &Y_min_in, const Y_Max_Type &Y_max_in)
    -> LMPC_QP_Solver<Number_Of_Variables, Output_Size, U_Type,
                      X_augmented_Type, Phi_Type, F_Type, Weight_U_Nc_Type,
                      Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
                      U_Max_Type, Y_Min_Type, Y_Max_Type,
                      Y_Constraints_Prediction_Offset> {

  return LMPC_QP_Solver<
      Number_Of_Variables, Output_Size, U_Type, X_augmented_Type, Phi_Type,
      F_Type, Weight_U_Nc_Type, Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
      U_Max_Type, Y_Min_Type, Y_Max_Type, Y_Constraints_Prediction_Offset>(
      U_in, X_augmented_in, Phi_in, F_in, weight_U_Nc_in, delta_U_Min_in,
      delta_U_Max_in, U_min_in, U_max_in, Y_min_in, Y_max_in);
}

/* LMPC_QP_Solver Type */
template <std::size_t Number_Of_Variables, std::size_t Output_Size,
          typename U_Type, typename X_augmented_Type, typename Phi_Type,
          typename F_Type, typename Weight_U_Nc_Type, typename Delta_U_Min_Type,
          typename Delta_U_Max_Type, typename U_Min_Type, typename U_Max_Type,
          typename Y_Min_Type, typename Y_Max_Type,
          std::size_t Y_Constraints_Prediction_Offset = 0>
using LMPC_QP_Solver_Type =
    LMPC_QP_Solver<Number_Of_Variables, Output_Size, U_Type, X_augmented_Type,
                   Phi_Type, F_Type, Weight_U_Nc_Type, Delta_U_Min_Type,
                   Delta_U_Max_Type, U_Min_Type, U_Max_Type, Y_Min_Type,
                   Y_Max_Type, Y_Constraints_Prediction_Offset>;

} // namespace PythonMPC

#endif // __MPC_LINEAR_SOLVER_UTILITY__
