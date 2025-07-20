#include "test_Adaptive_MPC_Phi_F_Updater_Function.hpp"

template <typename T>
auto get_adaptive_mpc_phi_f_updater_function()
    -> Adaptive_MPC_Phi_F_Updater_Function_Object<
        StateSpaceState_Type<T, PythonMPC_TwoWheelVehicleModelData::STATE_SIZE>,
        StateSpaceInput_Type<T, PythonMPC_TwoWheelVehicleModelData::INPUT_SIZE>,
        PythonMPC_TwoWheelVehicleModelData::
            two_wheel_vehicle_model_ada_mpc_ekf_parameter::Parameter_Type<T>,
        PythonMPC_TwoWheelVehicleModelData::
            two_wheel_vehicle_model_ada_mpc_Phi::type<T>,
        PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_F::
            type<T>,
        typename EmbeddedIntegratorTypes<
            PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ekf_A::
                type<T>,
            PythonMPC_TwoWheelVehicleModelData::
                two_wheel_vehicle_model_ada_mpc_B::type<T>,
            PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ekf_C::
                type<T>>::StateSpace_Type> {

  using X_Type = StateSpaceState_Type<T, PythonMPC_TwoWheelVehicleModelData::STATE_SIZE>;
  using U_Type = StateSpaceInput_Type<T, PythonMPC_TwoWheelVehicleModelData::INPUT_SIZE>;
  using Parameter_Type = PythonMPC_TwoWheelVehicleModelData::
      two_wheel_vehicle_model_ada_mpc_ekf_parameter::Parameter_Type<T>;
  using Phi_Type = PythonMPC_TwoWheelVehicleModelData::
      two_wheel_vehicle_model_ada_mpc_Phi::type<T>;
  using F_Type = PythonMPC_TwoWheelVehicleModelData::
      two_wheel_vehicle_model_ada_mpc_F::type<T>;
  using EmbeddedIntegratorStateSpace_Type = typename EmbeddedIntegratorTypes<PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ekf_A::type<T>,
      PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_B::type<T>,
      PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ekf_C::type<T>
  >::StateSpace_Type;

  using ReturnType = Adaptive_MPC_Phi_F_Updater_Function_Object<
      X_Type, U_Type, Parameter_Type, Phi_Type, F_Type, EmbeddedIntegratorStateSpace_Type>;

  return static_cast<ReturnType>(& PythonMPC_TwoWheelVehicleModelData::
      two_wheel_vehicle_model_adaptive_mpc_phi_f_updater::
          Adaptive_MPC_Phi_F_Updater<T>::template update<
              X_Type, U_Type, Parameter_Type, Phi_Type, F_Type,
              EmbeddedIntegratorStateSpace_Type>);
}

// Explicit instantiation for float type
template auto get_adaptive_mpc_phi_f_updater_function<double>()
    -> Adaptive_MPC_Phi_F_Updater_Function_Object<
        StateSpaceState_Type<double,
                             PythonMPC_TwoWheelVehicleModelData::STATE_SIZE>,
        StateSpaceInput_Type<double,
                             PythonMPC_TwoWheelVehicleModelData::INPUT_SIZE>,
        PythonMPC_TwoWheelVehicleModelData::
            two_wheel_vehicle_model_ada_mpc_ekf_parameter::Parameter_Type<
                double>,
        PythonMPC_TwoWheelVehicleModelData::
            two_wheel_vehicle_model_ada_mpc_Phi::type<double>,
        PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_F::
            type<double>,
        typename EmbeddedIntegratorTypes<
            PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ekf_A::
                type<double>,
            PythonMPC_TwoWheelVehicleModelData::
                two_wheel_vehicle_model_ada_mpc_B::type<double>,
            PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ekf_C::
                type<double>>::StateSpace_Type>;

template auto get_adaptive_mpc_phi_f_updater_function<float>()
    -> Adaptive_MPC_Phi_F_Updater_Function_Object<
        StateSpaceState_Type<float,
                             PythonMPC_TwoWheelVehicleModelData::STATE_SIZE>,
        StateSpaceInput_Type<float,
                             PythonMPC_TwoWheelVehicleModelData::INPUT_SIZE>,
        PythonMPC_TwoWheelVehicleModelData::
            two_wheel_vehicle_model_ada_mpc_ekf_parameter::Parameter_Type<
                float>,
        PythonMPC_TwoWheelVehicleModelData::
            two_wheel_vehicle_model_ada_mpc_Phi::type<float>,
        PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_F::
            type<float>,
        typename EmbeddedIntegratorTypes<
            PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ekf_A::
                type<float>,
            PythonMPC_TwoWheelVehicleModelData::
                two_wheel_vehicle_model_ada_mpc_B::type<float>,
            PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ekf_C::
                type<float>>::StateSpace_Type>;
