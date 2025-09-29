#ifndef __TEST_MPC_KINEMATIC_BICYCLE_MODEL_DATA_HPP__
#define __TEST_MPC_KINEMATIC_BICYCLE_MODEL_DATA_HPP__

#include "python_mpc.hpp"

namespace PythonMPC_KinematicBicycleModelData {

using namespace PythonMath;
using namespace PythonNumpy;
using namespace PythonControl;

namespace kinematic_bicycle_model_nmpc_ekf_parameter {

template <typename T>
class Parameter {
public:
    T wheel_base = static_cast<T>(2.8);
    T delta_time = static_cast<T>(0.1);
};

template <typename T>
using Parameter_Type = Parameter<T>;

} // namespace kinematic_bicycle_model_nmpc_ekf_parameter

template <typename T>
using Parameter_Type = kinematic_bicycle_model_nmpc_ekf_parameter::Parameter_Type<T>;

namespace kinematic_bicycle_model_ekf_A {

using SparseAvailable_ekf_A = SparseAvailable<
    ColumnAvailable<true, false, true, false>,
    ColumnAvailable<false, true, true, true>,
    ColumnAvailable<false, false, true, true>,
    ColumnAvailable<false, false, true, true>
>;

template <typename T>
using type = SparseMatrix_Type<T, SparseAvailable_ekf_A>;

template <typename T>
inline auto make(void) -> type<T> {

    return make_SparseMatrix<SparseAvailable_ekf_A>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

}

} // namespace kinematic_bicycle_model_ekf_A

namespace kinematic_bicycle_model_ekf_C {

template <typename T>
using type = DiagMatrix_Type<T, 4>;

template <typename T>
inline auto make(void) -> type<T> {

    return make_DiagMatrix<4>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

}

} // namespace kinematic_bicycle_model_ekf_C

namespace kinematic_bicycle_model_nmpc_ekf_state_function {

template <typename T>
using A_Type = kinematic_bicycle_model_ekf_A::type<T>;

template <typename T>
using X_Type = StateSpaceState_Type<T, A_Type<T>::COLS>;

template <typename T>
using U_Type = StateSpaceInput_Type<T, 2>;

template <typename T>
inline auto sympy_function(const T delta, const T wheel_base, const T v, const T q3, const T px, const T delta_time, const T q0, const T py) -> X_Type<T> {

    X_Type result;

    T x0 = delta_time * v;

    T x1 = x0 * tan(delta) / (static_cast<T>(2) * wheel_base);

    T x2 = cos(x1);

    T x3 = sin(x1);

    result.template set<0, 0>(static_cast<T>(px + x0 * (static_cast<T>(2) * (q0 * q0) - static_cast<T>(1))));
    result.template set<1, 0>(static_cast<T>(py + static_cast<T>(2) * q0 * q3 * x0));
    result.template set<2, 0>(static_cast<T>(q0 * x2 - q3 * x3));
    result.template set<3, 0>(static_cast<T>(q0 * x3 + q3 * x2));

    return result;
}

template <typename T>
inline auto function(const X_Type<T> X, const U_Type<T> U, const Parameter_Type<T> Parameters) -> X_Type<T> {

    T px = X.template get<0, 0>();

    T py = X.template get<1, 0>();

    T q0 = X.template get<2, 0>();

    T q3 = X.template get<3, 0>();

    T v = U.template get<0, 0>();

    T delta = U.template get<1, 0>();

    T wheel_base = Parameters.wheel_base;

    T delta_time = Parameters.delta_time;

    return sympy_function(delta, wheel_base, v, q3, px, delta_time, q0, py);
}

} // namespace kinematic_bicycle_model_nmpc_ekf_state_function

namespace kinematic_bicycle_model_nmpc_ekf_state_function_jacobian {

template <typename T>
using A_Type = kinematic_bicycle_model_ekf_A::type<T>;

template <typename T>
using X_Type = StateSpaceState_Type<T, A_Type<T>::COLS>;

template <typename T>
using U_Type = StateSpaceInput_Type<T, 2>;

template <typename T>
inline auto sympy_function(const T delta, const T wheel_base, const T v, const T q3, const T delta_time, const T q0) -> A_Type<T> {

    A_Type<T> result;

    T x0 = delta_time * v;

    T x1 = q0 * x0;

    T x2 = x0 * tan(delta) / (static_cast<T>(2) * wheel_base);

    T x3 = cos(x2);

    T x4 = sin(x2);

    result.template set<0, 0>(static_cast<T>(1));
    result.template set<0, 1>(static_cast<T>(0));
    result.template set<0, 2>(static_cast<T>(4 * x1));
    result.template set<0, 3>(static_cast<T>(0));
    result.template set<1, 0>(static_cast<T>(0));
    result.template set<1, 1>(static_cast<T>(1));
    result.template set<1, 2>(static_cast<T>(2 * q3 * x0));
    result.template set<1, 3>(static_cast<T>(2 * x1));
    result.template set<2, 0>(static_cast<T>(0));
    result.template set<2, 1>(static_cast<T>(0));
    result.template set<2, 2>(static_cast<T>(x3));
    result.template set<2, 3>(static_cast<T>(-x4));
    result.template set<3, 0>(static_cast<T>(0));
    result.template set<3, 1>(static_cast<T>(0));
    result.template set<3, 2>(static_cast<T>(x4));
    result.template set<3, 3>(static_cast<T>(x3));

    return result;
}

template <typename T>
inline auto function(const X_Type<T> X, const U_Type<T> U, const Parameter_Type<T> Parameters) -> A_Type<T> {

    T q0 = X.template get<2, 0>();

    T q3 = X.template get<3, 0>();

    T v = U.template get<0, 0>();

    T delta = U.template get<1, 0>();

    T wheel_base = Parameters.wheel_base;

    T delta_time = Parameters.delta_time;

    return sympy_function(delta, wheel_base, v, q3, delta_time, q0);
}


} // namespace kinematic_bicycle_model_nmpc_ekf_state_function_jacobian

namespace kinematic_bicycle_model_nmpc_ekf_measurement_function {

template <typename T>
using A_Type = kinematic_bicycle_model_ekf_A::type<T>;

template <typename T>
using C_Type = kinematic_bicycle_model_ekf_C::type<T>;

template <typename T>
using X_Type = StateSpaceState_Type<T, A_Type<T>::COLS>;

template <typename T>
using Y_Type = StateSpaceOutput_Type<T, C_Type<T>::COLS>;

template <typename T>
inline auto sympy_function(const T q3, const T q0, const T py, const T px) -> Y_Type<T> {

    Y_Type<T> result;

    result.template set<0, 0>(static_cast<T>(px));
    result.template set<1, 0>(static_cast<T>(py));
    result.template set<2, 0>(static_cast<T>(q0));
    result.template set<3, 0>(static_cast<T>(q3));

    return result;
}

template <typename T>
inline auto function(const X_Type<T> X, const Parameter_Type<T> Parameters) -> Y_Type<T> {

    T px = X.template get<0, 0>();

    T py = X.template get<1, 0>();

    T q0 = X.template get<2, 0>();

    T q3 = X.template get<3, 0>();

    return sympy_function(q3, q0, py, px);
}


} // namespace kinematic_bicycle_model_nmpc_ekf_measurement_function

namespace kinematic_bicycle_model_nmpc_ekf_measurement_function_jacobian {

template <typename T>
using A_Type = kinematic_bicycle_model_ekf_A::type<T>;

template <typename T>
using C_Type = kinematic_bicycle_model_ekf_C::type<T>;

template <typename T>
using X_Type = StateSpaceState_Type<T, A_Type<T>::COLS>;

template <typename T>
using Y_Type = StateSpaceOutput_Type<T, C_Type<T>::COLS>;

template <typename T>
inline auto sympy_function() -> C_Type<T> {

    C_Type<T> result;

    result.template set<0, 0>(static_cast<T>(1));
    result.template set<0, 1>(static_cast<T>(0));
    result.template set<0, 2>(static_cast<T>(0));
    result.template set<0, 3>(static_cast<T>(0));
    result.template set<1, 0>(static_cast<T>(0));
    result.template set<1, 1>(static_cast<T>(1));
    result.template set<1, 2>(static_cast<T>(0));
    result.template set<1, 3>(static_cast<T>(0));
    result.template set<2, 0>(static_cast<T>(0));
    result.template set<2, 1>(static_cast<T>(0));
    result.template set<2, 2>(static_cast<T>(1));
    result.template set<2, 3>(static_cast<T>(0));
    result.template set<3, 0>(static_cast<T>(0));
    result.template set<3, 1>(static_cast<T>(0));
    result.template set<3, 2>(static_cast<T>(0));
    result.template set<3, 3>(static_cast<T>(1));

    return result;
}

template <typename T>
inline auto function(const X_Type<T> X, const Parameter_Type<T> Parameters) -> C_Type<T> {

    return sympy_function<T>();
}


} // namespace kinematic_bicycle_model_nmpc_ekf_measurement_function_jacobian

namespace kinematic_bicycle_model_nmpc_ekf {

constexpr std::size_t NUMBER_OF_DELAY = 0;

template <typename T>
using A_Type = kinematic_bicycle_model_ekf_A::type<T>;

template <typename T>
using C_Type = kinematic_bicycle_model_ekf_C::type<T>;

constexpr std::size_t STATE_SIZE = 4;
constexpr std::size_t INPUT_SIZE = 2;
constexpr std::size_t OUTPUT_SIZE = 2;

template <typename T>
using X_Type = StateSpaceState_Type<T, STATE_SIZE>;

template <typename T>
using U_Type = StateSpaceInput_Type<T, INPUT_SIZE>;

template <typename T>
using Y_Type = StateSpaceOutput_Type<T, OUTPUT_SIZE>;

template <typename T>
using Q_Type = KalmanFilter_Q_Type<T, STATE_SIZE>;

template <typename T>
using R_Type = KalmanFilter_R_Type<T, OUTPUT_SIZE>;

template <typename T>
using Parameter_Type = kinematic_bicycle_model_nmpc_ekf_parameter::Parameter_Type<T>;

template <typename T>
using type = ExtendedKalmanFilter_Type<
    A_Type<T>, C_Type<T>, U_Type<T>, Q_Type<T>, R_Type<T>, Parameter_Type<T>, NUMBER_OF_DELAY>;

template <typename T>
inline auto make() -> type<T> {

    auto Q = make_KalmanFilter_Q<STATE_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

    auto R = make_KalmanFilter_R<OUTPUT_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

    Parameter_Type<T> parameters;

    StateFunction_Object<X_Type<T>, U_Type<T>, Parameter_Type<T>> state_function_object =
        [](const X_Type<T>& X, const U_Type<T>& U, const Parameter_Type<T>& Parameters) {
        return kinematic_bicycle_model_nmpc_ekf_state_function::function(X, U, Parameters);
        };

    StateFunctionJacobian_Object<A_Type<T>, X_Type<T>, U_Type<T>, Parameter_Type<T>> state_function_jacobian_object =
        [](const X_Type<T>& X, const U_Type<T>& U, const Parameter_Type<T>& Parameters) {
        return kinematic_bicycle_model_nmpc_ekf_state_function_jacobian::function(X, U, Parameters);
        };

    MeasurementFunction_Object<Y_Type<T>, X_Type<T>, Parameter_Type<T>> measurement_function_object =
        [](const X_Type<T>& X, const Parameter_Type<T>& Parameters) {
        return kinematic_bicycle_model_nmpc_ekf_measurement_function::function(X, Parameters);
        };

    MeasurementFunctionJacobian_Object<C_Type<T>, X_Type<T>, Parameter_Type<T>> measurement_function_jacobian_object =
        [](const X_Type<T>& X, const Parameter_Type<T>& Parameters) {
        return kinematic_bicycle_model_nmpc_ekf_measurement_function_jacobian::function(X, Parameters);
        };

    return ExtendedKalmanFilter_Type<
        A_Type<T>, C_Type<T>, U_Type<T>, Q_Type<T>, R_Type<T>, Parameter_Type<T>, NUMBER_OF_DELAY>(
            Q, R, state_function_object, state_function_jacobian_object,
            measurement_function_object, measurement_function_jacobian_object,
            parameters);

}

} // namespace kinematic_bicycle_model_nmpc_ekf


} // namespace PythonMPC_KinematicBicycleModelData

#endif // __TEST_MPC_KINEMATIC_BICYCLE_MODEL_DATA_HPP__
