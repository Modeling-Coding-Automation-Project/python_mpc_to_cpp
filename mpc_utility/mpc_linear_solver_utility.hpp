#ifndef __MPC_LINEAR_SOLVER_UTILITY__
#define __MPC_LINEAR_SOLVER_UTILITY__

#include "python_numpy.hpp"
#include "python_optimization.hpp"

#include <functional>
#include <type_traits>

namespace PythonMPC {

namespace SolverUtility {

static constexpr std::size_t MAX_ITERATION_DEFAULT = 10;
static constexpr double TOL_DEFAULT = 1e-8;

} // namespace SolverUtility

namespace SolverUtilityOperation {

template <bool Flag> struct CountTrueCondition {};

template <> struct CountTrueCondition<true> {
  static constexpr std::size_t value = static_cast<std::size_t>(1);
};

template <> struct CountTrueCondition<false> {
  static constexpr std::size_t value = static_cast<std::size_t>(0);
};

template <typename Flags, std::size_t Col, std::size_t Row>
struct CountTrue2D_Row {
  static constexpr std::size_t value =
      CountTrueCondition<Flags::lists[Col][Row]>::value +
      CountTrue2D_Row<Flags, Col, Row - 1>::value;
};

template <typename Flags, std::size_t Col>
struct CountTrue2D_Row<Flags, Col, static_cast<std::size_t>(-1)> {
  static constexpr std::size_t value = static_cast<std::size_t>(0);
};

template <typename Flags, std::size_t Col, std::size_t Row>
struct CountTrue2D_Col {
  static constexpr std::size_t value =
      CountTrue2D_Row<Flags, Col, Row - 1>::value +
      CountTrue2D_Col<Flags, Col - 1, Row>::value;
};

template <typename Flags, std::size_t Row>
struct CountTrue2D_Col<Flags, static_cast<std::size_t>(-1), Row> {
  static constexpr std::size_t value = static_cast<std::size_t>(0);
};

template <typename Flags, std::size_t Col, std::size_t Row>
using CountTrue2D = CountTrue2D_Col<Flags, Col - 1, Row>;

} // namespace SolverUtilityOperation

/* Define delta U, U, Y limits */
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

  inline auto get_number_of_all_constraints(void) const -> std::size_t {

    return this->_number_of_delta_U_constraints +
           this->_number_of_U_constraints + this->_number_of_Y_constraints;
  }

  inline auto get_number_of_delta_U_constraints(void) const -> std::size_t {
    return this->_number_of_delta_U_constraints;
  }

  inline auto get_number_of_U_constraints(void) const -> std::size_t {
    return this->_number_of_U_constraints;
  }

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

namespace LTI_MPC_QP_SolverOperation {

/* calculate M gamma for delta U min  */
template <typename M_Type, typename Gamma_Type,
          typename Delta_U_min_Matrix_Type, std::size_t I, bool Flag>
struct Calculate_M_Gamma_Delta_U_Min_Condition {};

template <typename M_Type, typename Gamma_Type,
          typename Delta_U_min_Matrix_Type, std::size_t I>
struct Calculate_M_Gamma_Delta_U_Min_Condition<
    M_Type, Gamma_Type, Delta_U_min_Matrix_Type, I, true> {
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

template <typename Flags_Type, typename M_Type, typename Gamma_Type,
          typename Delta_U_min_Matrix_Type, std::size_t I, std::size_t I_Dif>
struct Calculate_M_Gamma_Delta_U_Min_Loop {
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Delta_U_min_Matrix_Type &delta_U_matrix,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {

    std::size_t set_count = static_cast<std::size_t>(0);

    Calculate_M_Gamma_Delta_U_Min_Condition<
        M_Type, Gamma_Type, Delta_U_min_Matrix_Type, I,
        Flags_Type::lists[I][0]>::apply(M, gamma, delta_U_matrix, set_count,
                                        initial_position);

    total_index += set_count;
    Calculate_M_Gamma_Delta_U_Min_Loop<Flags_Type, M_Type, Gamma_Type,
                                       Delta_U_min_Matrix_Type, (I + 1),
                                       (I_Dif - 1)>::apply(M, gamma,
                                                           delta_U_matrix,
                                                           total_index,
                                                           initial_position);
  }
};

template <typename Flags_Type, typename M_Type, typename Gamma_Type,
          typename Delta_U_min_Matrix_Type, std::size_t I>
struct Calculate_M_Gamma_Delta_U_Min_Loop<Flags_Type, M_Type, Gamma_Type,
                                          Delta_U_min_Matrix_Type, I, 0> {
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Delta_U_min_Matrix_Type &delta_U_matrix,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {

    // Do Nothing.
    static_cast<void>(delta_U_matrix);
    static_cast<void>(total_index);
    static_cast<void>(initial_position);
  }
};

template <typename Flags_Type, typename M_Type, typename Gamma_Type,
          typename Delta_U_min_Matrix_Type, std::size_t Delta_U_Size>
using Calculate_M_Gamma_Delta_U_Min = Calculate_M_Gamma_Delta_U_Min_Loop<
    Flags_Type, M_Type, Gamma_Type, Delta_U_min_Matrix_Type, 0, Delta_U_Size>;

/* calculate M gamma for delta U max  */
template <typename M_Type, typename Gamma_Type,
          typename Delta_U_max_Matrix_Type, std::size_t I, bool Flag>
struct Calculate_M_Gamma_Delta_U_Max_Condition {};

template <typename M_Type, typename Gamma_Type,
          typename Delta_U_max_Matrix_Type, std::size_t I>
struct Calculate_M_Gamma_Delta_U_Max_Condition<
    M_Type, Gamma_Type, Delta_U_max_Matrix_Type, I, true> {
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

template <typename Flags_Type, typename M_Type, typename Gamma_Type,
          typename Delta_U_max_Matrix_Type, std::size_t I, std::size_t I_Dif>
struct Calculate_M_Gamma_Delta_U_Max_Loop {
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Delta_U_max_Matrix_Type &delta_U_matrix,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {

    std::size_t set_count = static_cast<std::size_t>(0);

    Calculate_M_Gamma_Delta_U_Max_Condition<
        M_Type, Gamma_Type, Delta_U_max_Matrix_Type, I,
        Flags_Type::lists[I][0]>::apply(M, gamma, delta_U_matrix, set_count,
                                        initial_position);

    total_index += set_count;
    Calculate_M_Gamma_Delta_U_Max_Loop<Flags_Type, M_Type, Gamma_Type,
                                       Delta_U_max_Matrix_Type, (I + 1),
                                       (I_Dif - 1)>::apply(M, gamma,
                                                           delta_U_matrix,
                                                           total_index,
                                                           initial_position);
  }
};

template <typename Flags_Type, typename M_Type, typename Gamma_Type,
          typename Delta_U_max_Matrix_Type, std::size_t I>
struct Calculate_M_Gamma_Delta_U_Max_Loop<Flags_Type, M_Type, Gamma_Type,
                                          Delta_U_max_Matrix_Type, I, 0> {
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const Delta_U_max_Matrix_Type &delta_U_matrix,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {

    // Do Nothing.
    static_cast<void>(delta_U_matrix);
    static_cast<void>(total_index);
    static_cast<void>(initial_position);
  }
};

template <typename Flags_Type, typename M_Type, typename Gamma_Type,
          typename Delta_U_max_Matrix_Type, std::size_t Delta_U_Size>
using Calculate_M_Gamma_Delta_U_Max = Calculate_M_Gamma_Delta_U_Max_Loop<
    Flags_Type, M_Type, Gamma_Type, Delta_U_max_Matrix_Type, 0, Delta_U_Size>;

/* calculate M gamma for U min */
template <typename M_Type, typename Gamma_Type, typename U_min_Matrix_Type,
          typename U_Type, std::size_t I, bool Flag>
struct Calculate_M_Gamma_U_Min_Condition {};

template <typename M_Type, typename Gamma_Type, typename U_min_Matrix_Type,
          typename U_Type, std::size_t I>
struct Calculate_M_Gamma_U_Min_Condition<M_Type, Gamma_Type, U_min_Matrix_Type,
                                         U_Type, I, true> {
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const U_min_Matrix_Type &U_matrix, const U_Type &U,
                    std::size_t &set_count,
                    const std::size_t &initial_position) {

    M.access(initial_position + I, I) =
        static_cast<typename M_Type::Value_Type>(-1.0);
    gamma.access(initial_position + I, 0) =
        -U_matrix.template get<I, 0>() + U.template get<0, I>();
    set_count += 1;
  }
};

template <typename M_Type, typename Gamma_Type, typename U_min_Matrix_Type,
          typename U_Type, std::size_t I>
struct Calculate_M_Gamma_U_Min_Condition<M_Type, Gamma_Type, U_min_Matrix_Type,
                                         U_Type, I, false> {
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

template <typename Flags_Type, typename M_Type, typename Gamma_Type,
          typename U_min_Matrix_Type, typename U_Type, std::size_t I,
          std::size_t I_Dif>
struct Calculate_M_Gamma_U_Min_Loop {
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const U_min_Matrix_Type &U_matrix, const U_Type &U,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {

    std::size_t set_count = static_cast<std::size_t>(0);

    Calculate_M_Gamma_U_Min_Condition<
        M_Type, Gamma_Type, U_min_Matrix_Type, U_Type, I,
        Flags_Type::lists[I][0]>::apply(M, gamma, U_matrix, U, set_count,
                                        initial_position);

    total_index += set_count;
    Calculate_M_Gamma_U_Min_Loop<Flags_Type, M_Type, Gamma_Type,
                                 U_min_Matrix_Type, U_Type, (I + 1),
                                 (I_Dif - 1)>::apply(M, gamma, U_matrix, U,
                                                     total_index,
                                                     initial_position);
  }
};

template <typename Flags_Type, typename M_Type, typename Gamma_Type,
          typename U_min_Matrix_Type, typename U_Type, std::size_t I>
struct Calculate_M_Gamma_U_Min_Loop<Flags_Type, M_Type, Gamma_Type,
                                    U_min_Matrix_Type, U_Type, I, 0> {
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

template <typename Flags_Type, typename M_Type, typename Gamma_Type,
          typename U_min_Matrix_Type, typename U_Type, std::size_t U_Min_Size>
using Calculate_M_Gamma_U_Min =
    Calculate_M_Gamma_U_Min_Loop<Flags_Type, M_Type, Gamma_Type,
                                 U_min_Matrix_Type, U_Type, 0, U_Min_Size>;

/* calculate M gamma for U max */
template <typename M_Type, typename Gamma_Type, typename U_max_Matrix_Type,
          typename U_Type, std::size_t I, bool Flag>
struct Calculate_M_Gamma_U_Max_Condition {};

template <typename M_Type, typename Gamma_Type, typename U_max_Matrix_Type,
          typename U_Type, std::size_t I>
struct Calculate_M_Gamma_U_Max_Condition<M_Type, Gamma_Type, U_max_Matrix_Type,
                                         U_Type, I, true> {
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const U_max_Matrix_Type &U_matrix, const U_Type &U,
                    std::size_t &set_count,
                    const std::size_t &initial_position) {

    M.access(initial_position + I, I) =
        static_cast<typename M_Type::Value_Type>(1.0);
    gamma.access(initial_position + I, 0) =
        U_matrix.template get<I, 0>() - U.template get<0, I>();
    set_count += 1;
  }
};

template <typename M_Type, typename Gamma_Type, typename U_max_Matrix_Type,
          typename U_Type, std::size_t I>
struct Calculate_M_Gamma_U_Max_Condition<M_Type, Gamma_Type, U_max_Matrix_Type,
                                         U_Type, I, false> {
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

template <typename Flags_Type, typename M_Type, typename Gamma_Type,
          typename U_max_Matrix_Type, typename U_Type, std::size_t I,
          std::size_t I_Dif>
struct Calculate_M_Gamma_U_Max_Loop {
  static void apply(M_Type &M, Gamma_Type &gamma,
                    const U_max_Matrix_Type &U_matrix, const U_Type &U,
                    std::size_t &total_index,
                    const std::size_t &initial_position) {

    std::size_t set_count = static_cast<std::size_t>(0);

    Calculate_M_Gamma_U_Max_Condition<
        M_Type, Gamma_Type, U_max_Matrix_Type, U_Type, I,
        Flags_Type::lists[I][0]>::apply(M, gamma, U_matrix, U, set_count,
                                        initial_position);

    total_index += set_count;
    Calculate_M_Gamma_U_Max_Loop<Flags_Type, M_Type, Gamma_Type,
                                 U_max_Matrix_Type, U_Type, (I + 1),
                                 (I_Dif - 1)>::apply(M, gamma, U_matrix, U,
                                                     total_index,
                                                     initial_position);
  }
};

template <typename Flags_Type, typename M_Type, typename Gamma_Type,
          typename U_max_Matrix_Type, typename U_Type, std::size_t I>
struct Calculate_M_Gamma_U_Max_Loop<Flags_Type, M_Type, Gamma_Type,
                                    U_max_Matrix_Type, U_Type, I, 0> {
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

template <typename Flags_Type, typename M_Type, typename Gamma_Type,
          typename U_max_Matrix_Type, typename U_Type, std::size_t U_Max_Size>
using Calculate_M_Gamma_U_Max =
    Calculate_M_Gamma_U_Max_Loop<Flags_Type, M_Type, Gamma_Type,
                                 U_max_Matrix_Type, U_Type, 0, U_Max_Size>;

} // namespace LTI_MPC_QP_SolverOperation

/* LTI MPC QP solver */
template <std::size_t Number_Of_Variables, std::size_t Output_Size,
          typename U_Type, typename X_augmented_Type, typename Phi_Type,
          typename F_Type, typename Weight_U_Nc_Type, typename Delta_U_Min_Type,
          typename Delta_U_Max_Type, typename U_Min_Type, typename U_Max_Type,
          typename Y_Min_Type, typename Y_Max_Type,
          std::size_t Y_Constraints_Prediction_Offset = 0>
class LTI_MPC_QP_Solver {
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

  using _Solver_Type = PythonOptimization::QP_ActiveSetSolver_Type<
      Value_Type, Number_Of_Variables, NUMBER_OF_ALL_CONSTRAINTS>;

public:
  /* Constructor */
  LTI_MPC_QP_Solver() : limits(), M(), gamma(), _solver() {}

  LTI_MPC_QP_Solver(const U_Type &U_in, const X_augmented_Type &X_augmented_in,
                    Phi_Type &Phi_in, const F_Type &F_in,
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
  }

  /* Copy Constructor */

  /* Move Constructor */

public:
  /* Function */
  template <std::size_t N = NUMBER_OF_ALL_CONSTRAINTS>
  inline typename std::enable_if<(N > 0), void>::type
  update_constraints(const U_Type &U_in, const X_augmented_Type &X_augmented_in,
                     Phi_Type &Phi_in, const F_Type &F_in) {

    std::size_t total_index = static_cast<std::size_t>(0);

    this->_calculate_M_gamma_delta_U(total_index);

    this->_calculate_M_gamma_U(total_index, U_in);

    this->_calculate_M_gamma_Y(total_index, X_augmented_in, Phi_in, F_in);
  }

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

  inline auto get_number_of_Y_constraints_prediction_offset(void) const
      -> std::size_t {
    return this->_Y_constraints_prediction_offset;
  }

protected:
  /* Function */
  inline void _calculate_M_gamma_delta_U(std::size_t &total_index) {
    std::size_t initial_position = total_index;

    // delta_U_min constraints
    LTI_MPC_QP_SolverOperation::Calculate_M_Gamma_Delta_U_Min<
        Limits_Type::Delta_U_Min_Flags, M_Type, Gamma_Type,
        decltype(this->limits.delta_U_min),
        Limits_Type::DELTA_U_MIN_SIZE>::apply(this->M, this->gamma,
                                              this->limits.delta_U_min,
                                              total_index, initial_position);

    initial_position = total_index;

    // delta_U_max constraints
    LTI_MPC_QP_SolverOperation::Calculate_M_Gamma_Delta_U_Max<
        Limits_Type::Delta_U_Max_Flags, M_Type, Gamma_Type,
        decltype(this->limits.delta_U_max),
        Limits_Type::DELTA_U_MAX_SIZE>::apply(this->M, this->gamma,
                                              this->limits.delta_U_max,
                                              total_index, initial_position);
  }

  inline void _calculate_M_gamma_U(std::size_t &total_index, const U_Type &U) {
    std::size_t initial_position = total_index;

    // U_min constraints
    LTI_MPC_QP_SolverOperation::Calculate_M_Gamma_U_Min<
        Limits_Type::U_Min_Flags, M_Type, Gamma_Type,
        decltype(this->limits.U_min), U_Type,
        Limits_Type::U_MIN_SIZE>::apply(this->M, this->gamma,
                                        this->limits.U_min, U, total_index,
                                        initial_position);

    initial_position = total_index;

    // U_max constraints
    LTI_MPC_QP_SolverOperation::Calculate_M_Gamma_U_Max<
        Limits_Type::U_Max_Flags, M_Type, Gamma_Type,
        decltype(this->limits.U_max), U_Type,
        Limits_Type::U_MAX_SIZE>::apply(this->M, this->gamma,
                                        this->limits.U_max, U, total_index,
                                        initial_position);
  }

  inline void _calculate_M_gamma_Y(std::size_t &total_index,
                                   const X_augmented_Type &X_augmented,
                                   Phi_Type &Phi, const F_Type &F) {

    std::size_t initial_position = total_index;
    auto F_X = F * X_augmented;

    // Y_min constraints
    for (std::size_t i = 0; i < Limits_Type::Y_MIN_SIZE; ++i) {
      std::size_t set_count = static_cast<std::size_t>(0);
      if (this->limits.is_Y_min_active(i)) {

        for (std::size_t j = 0; j < Number_Of_Variables; ++j) {
          this->M.access(initial_position + i, j) =
              -Phi.access(Y_CONSTRAINTS_PREDICTION_OFFSET + i, j);
        }

        this->gamma.access(initial_position + i, 0) =
            -this->limits.Y_min.access(i, 0) +
            F_X.access(Y_CONSTRAINTS_PREDICTION_OFFSET + i, 0);
        set_count += 1;
      }
      total_index += set_count;
    }

    initial_position = total_index;

    // Y_max constraints
    for (std::size_t i = 0; i < Limits_Type::Y_MAX_SIZE; ++i) {
      std::size_t set_count = static_cast<std::size_t>(0);
      if (this->limits.is_Y_max_active(i)) {

        for (std::size_t j = 0; j < Number_Of_Variables; ++j) {
          this->M.access(initial_position + i, j) =
              Phi.access(Y_CONSTRAINTS_PREDICTION_OFFSET + i, j);
        }

        this->gamma.access(initial_position + i, 0) =
            this->limits.Y_max.access(i, 0) -
            F_X.access(Y_CONSTRAINTS_PREDICTION_OFFSET + i, 0);
        set_count += 1;
      }
      total_index += set_count;
    }
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

/* make LTI_MPC_QP_Solver */
template <std::size_t Number_Of_Variables, std::size_t Output_Size,
          typename U_Type, typename X_augmented_Type, typename Phi_Type,
          typename F_Type, typename Weight_U_Nc_Type, typename Delta_U_Min_Type,
          typename Delta_U_Max_Type, typename U_Min_Type, typename U_Max_Type,
          typename Y_Min_Type, typename Y_Max_Type,
          std::size_t Y_Constraints_Prediction_Offset = 0>
inline auto
make_LTI_MPC_QP_Solver(const U_Type &U_in,
                       const X_augmented_Type &X_augmented_in, Phi_Type &Phi_in,
                       const F_Type &F_in,
                       const Weight_U_Nc_Type &weight_U_Nc_in,
                       const Delta_U_Min_Type &delta_U_Min_in,
                       const Delta_U_Max_Type &delta_U_Max_in,
                       const U_Min_Type &U_min_in, const U_Max_Type &U_max_in,
                       const Y_Min_Type &Y_min_in, const Y_Max_Type &Y_max_in)
    -> LTI_MPC_QP_Solver<Number_Of_Variables, Output_Size, U_Type,
                         X_augmented_Type, Phi_Type, F_Type, Weight_U_Nc_Type,
                         Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
                         U_Max_Type, Y_Min_Type, Y_Max_Type,
                         Y_Constraints_Prediction_Offset> {

  return LTI_MPC_QP_Solver<
      Number_Of_Variables, Output_Size, U_Type, X_augmented_Type, Phi_Type,
      F_Type, Weight_U_Nc_Type, Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
      U_Max_Type, Y_Min_Type, Y_Max_Type, Y_Constraints_Prediction_Offset>(
      U_in, X_augmented_in, Phi_in, F_in, weight_U_Nc_in, delta_U_Min_in,
      delta_U_Max_in, U_min_in, U_max_in, Y_min_in, Y_max_in);
}

/* LTI_MPC_QP_Solver Type */
template <std::size_t Number_Of_Variables, std::size_t Output_Size,
          typename U_Type, typename X_augmented_Type, typename Phi_Type,
          typename F_Type, typename Weight_U_Nc_Type, typename Delta_U_Min_Type,
          typename Delta_U_Max_Type, typename U_Min_Type, typename U_Max_Type,
          typename Y_Min_Type, typename Y_Max_Type,
          std::size_t Y_Constraints_Prediction_Offset = 0>
using LTI_MPC_QP_Solver_Type = LTI_MPC_QP_Solver<
    Number_Of_Variables, Output_Size, U_Type, X_augmented_Type, Phi_Type,
    F_Type, Weight_U_Nc_Type, Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
    U_Max_Type, Y_Min_Type, Y_Max_Type, Y_Constraints_Prediction_Offset>;

} // namespace PythonMPC

#endif // __MPC_LINEAR_SOLVER_UTILITY__
