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

protected:
  /* Function */

public:
  /* Constant */
  static constexpr std::size_t DELTA_U_MIN_SIZE = Delta_U_Min_Type::COLS;
  static constexpr std::size_t DELTA_U_MAX_SIZE = Delta_U_Max_Type::COLS;
  static constexpr std::size_t U_MIN_SIZE = U_Min_Type::COLS;
  static constexpr std::size_t U_MAX_SIZE = U_Max_Type::COLS;
  static constexpr std::size_t Y_MIN_SIZE = Y_Min_Type::COLS;
  static constexpr std::size_t Y_MAX_SIZE = Y_Max_Type::COLS;

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

/* LTI MPC QP solver */
template <std::size_t Number_Of_Variables, std::size_t Output_Size,
          typename U_Type, typename X_augmented_Type, typename Phi_Type,
          typename F_Type, typename Delta_U_Min_Type, typename Delta_U_Max_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type>
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

  /* Check Compatibility */

protected:
  /* Type */
  using _T = Value_Type;

  using _Solver_Type = PythonOptimization::QP_ActiveSetSolver_Type<
      Value_Type, Number_Of_Variables, NUMBER_OF_ALL_CONSTRAINTS>;

public:
  /* Constructor */
  LTI_MPC_QP_Solver()
      : max_iteration(SolverUtility::MAX_ITERATION_DEFAULT),
        tol(static_cast<_T>(SolverUtility::TOL_DEFAULT)), M(), gamma(),
        _solver(),
        _Y_constraints_prediction_offset(static_cast<std::size_t>(0)) {}

public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_VARIABLES = Number_Of_Variables;
  static constexpr std::size_t U_SIZE = U_Type::COLS;
  static constexpr std::size_t Y_SIZE = Output_Size;

public:
  /* Variable */
  std::size_t max_iteration;
  Value_Type tol;

  M_Type M;
  Gamma_Type gamma;

protected:
  /* Variable */
  _Solver_Type _solver;
  std::size_t _Y_constraints_prediction_offset;
};

} // namespace PythonMPC

#endif // __MPC_LINEAR_SOLVER_UTILITY__
