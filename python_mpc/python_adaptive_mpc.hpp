#ifndef __PYTHON_ADAPTIVE_MPC_HPP__
#define __PYTHON_ADAPTIVE_MPC_HPP__

#include "mpc_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <functional>
#include <type_traits>

namespace PythonMPC {

/* Adaptive MPC Function Object */

template <typename X_Type, typename Y_Type, typename Parameter_Type,
          typename Phi_Type, typename F_Type, typename StateSpace_Type>
using Adaptive_MPC_Phi_F_Updater_Function_Object =
    std::function<void(const X_Type &, const Y_Type &, const Parameter_Type &,
                       Phi_Type &, F_Type &)>;

/* Adaptive MPC No Constraints */

template <typename EKF_Type_In, typename PredictionMatrices_Type_In,
          typename ReferenceTrajectory_Type_In, typename Parameter_Type_In,
          typename SolverFactor_Type_In = SolverFactor_Empty>
class AdaptiveMPC_NoConstraints {};

/* Adaptive MPC No Constraints Type */
template <typename EKF_Type, typename PredictionMatrices_Type,
          typename ReferenceTrajectory_Type, typename Parameter_Type,
          typename SolverFactor_Type_In = SolverFactor_Empty>
using AdaptiveMPC_NoConstraints_Type =
    AdaptiveMPC_NoConstraints<EKF_Type, PredictionMatrices_Type,
                              ReferenceTrajectory_Type, Parameter_Type,
                              SolverFactor_Type_In>;

} // namespace PythonMPC

#endif // __PYTHON_ADAPTIVE_MPC_HPP__
