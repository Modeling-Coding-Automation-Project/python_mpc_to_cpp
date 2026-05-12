#ifndef ADAPTIVE_MPC_FUNCTION_DECLARATIONS_HPP_
#define ADAPTIVE_MPC_FUNCTION_DECLARATIONS_HPP_

#include "python_mpc.hpp"
#include "test_mpc_two_wheel_vehicle_model_data.hpp"

using namespace PythonNumpy;
using namespace PythonControl;
using namespace PythonMPC;

template <typename T>
auto get_adaptive_mpc_phi_f_updater_function() ->
Adaptive_MPC_Phi_F_Updater_Function_Object<
    StateSpaceState_Type<T, PythonMPC_TwoWheelVehicleModelData::STATE_SIZE>,
    StateSpaceInput_Type<T, PythonMPC_TwoWheelVehicleModelData::INPUT_SIZE>,
    PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_ekf_parameter::Parameter_Type<T>,
    PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_Phi::type<T>,
    PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_F::type<T>,
    typename EmbeddedIntegratorTypes<
    PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ekf_A::type<T>,
    PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_B::type<T>,
    PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ekf_C::type<T>
    >::StateSpace_Type>;

#endif // ADAPTIVE_MPC_FUNCTION_DECLARATIONS_HPP_
