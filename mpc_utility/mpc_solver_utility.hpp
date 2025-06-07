#ifndef __MPC_SOLVER_UTILITY_HPP__
#define __MPC_SOLVER_UTILITY_HPP__

#include "python_numpy.hpp"

#include <functional>
#include <type_traits>

namespace PythonMPC {

template <typename Delta_U_Min_Type, typename Delta_U_Max_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type>
class DU_U_Y_Limits {
public:
  /* Type */
  using Value_Type = typename Delta_U_Min_Type::Value_Type;

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
        U_min(U_min_in), U_max(U_max_in), Y_min(Y_min_in), Y_max(Y_max_in) {

    this->count_constraints();
  }

protected:
  /* Function */
  inline void count_constraints(void) {

    using Delta_U_Min_Flags = typename Delta_U_Min_Type::SparseAvailable_Type;
    using Delta_U_Max_Flags = typename Delta_U_Max_Type::SparseAvailable_Type;
    using U_Min_Flags = typename U_Min_Type::SparseAvailable_Type;
    using U_Max_Flags = typename U_Max_Type::SparseAvailable_Type;
    using Y_Min_Flags = typename Y_Min_Type::SparseAvailable_Type;
    using Y_Max_Flags = typename Y_Max_Type::SparseAvailable_Type;

    this->_number_of_delta_U_constraints = static_cast<std::size_t>(0);
    this->_number_of_U_constraints = static_cast<std::size_t>(0);
    this->_number_of_Y_constraints = static_cast<std::size_t>(0);

    for (std::size_t i = 0; i < Delta_U_Min_Flags::COLS; ++i) {
      for (std::size_t j = 0; j < Delta_U_Min_Flags::ROWS; ++j) {
        if (Delta_U_Min_Flags::lists[i][j]) {
          this->_number_of_delta_U_constraints++;
        }
      }
    }

    for (std::size_t i = 0; i < Delta_U_Max_Flags::COLS; ++i) {
      for (std::size_t j = 0; j < Delta_U_Max_Flags::ROWS; ++j) {
        if (Delta_U_Max_Flags::lists[i][j]) {
          this->_number_of_delta_U_constraints++;
        }
      }
    }

    for (std::size_t i = 0; i < U_Min_Flags::COLS; ++i) {
      for (std::size_t j = 0; j < U_Min_Flags::ROWS; ++j) {
        if (U_Min_Flags::lists[i][j]) {
          this->_number_of_U_constraints++;
        }
      }
    }

    for (std::size_t i = 0; i < U_Max_Flags::COLS; ++i) {
      for (std::size_t j = 0; j < U_Max_Flags::ROWS; ++j) {
        if (U_Max_Flags::lists[i][j]) {
          this->_number_of_U_constraints++;
        }
      }
    }

    for (std::size_t i = 0; i < Y_Min_Flags::COLS; ++i) {
      for (std::size_t j = 0; j < Y_Min_Flags::ROWS; ++j) {
        if (Y_Min_Flags::lists[i][j]) {
          this->_number_of_Y_constraints++;
        }
      }
    }

    for (std::size_t i = 0; i < Y_Max_Flags::COLS; ++i) {
      for (std::size_t j = 0; j < Y_Max_Flags::ROWS; ++j) {
        if (Y_Max_Flags::lists[i][j]) {
          this->_number_of_Y_constraints++;
        }
      }
    }
  }

public:
  /* Variables */
  Delta_U_Min_Type delta_U_min;
  Delta_U_Max_Type delta_U_max;
  U_Min_Type U_min;
  U_Max_Type U_max;
  Y_Min_Type Y_min;
  Y_Max_Type Y_max;

protected:
  /* Variables */
  std::size_t _number_of_delta_U_constraints;
  std::size_t _number_of_U_constraints;
  std::size_t _number_of_Y_constraints;
};

} // namespace PythonMPC

#endif // __MPC_SOLVER_UTILITY_HPP__
