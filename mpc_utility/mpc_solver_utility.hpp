#ifndef __MPC_SOLVER_UTILITY_HPP__
#define __MPC_SOLVER_UTILITY_HPP__

#include "python_control.hpp"
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
  using Value_Type = Delta_U_Min_Type::Value_Type;

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
  /* Variables */
  Delta_U_Min_Type delta_u_min;
  Delta_U_Max_Type delta_u_max;
  U_Min_Type u_min;
  U_Max_Type u_max;
  Y_Min_Type y_min;
  Y_Max_Type y_max;
};

} // namespace PythonMPC

#endif // __MPC_SOLVER_UTILITY_HPP__
