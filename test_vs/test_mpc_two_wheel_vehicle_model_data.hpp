#ifndef __TEST_MPC_TWO_WHEEL_VEHICLE_MODEL_DATA_HPP__
#define __TEST_MPC_TWO_WHEEL_VEHICLE_MODEL_DATA_HPP__

#include "python_mpc.hpp"

namespace PythonMPC_TwoWheelVehicleModelData {

using namespace PythonNumpy;
using namespace PythonControl;

constexpr std::size_t NP = 4;
constexpr std::size_t NC = 1;

constexpr std::size_t INPUT_SIZE = 2;
constexpr std::size_t STATE_SIZE = 6;
constexpr std::size_t OUTPUT_SIZE = 5;

constexpr std::size_t AUGMENTED_STATE_SIZE = STATE_SIZE + OUTPUT_SIZE;

constexpr std::size_t NUMBER_OF_DELAY = 0;

namespace two_wheel_vehicle_model_ekf_A {

using namespace PythonNumpy;

using SparseAvailable_ekf_A = SparseAvailable<
    ColumnAvailable<true, false, true, false, false, true>,
    ColumnAvailable<false, true, true, false, false, true>,
    ColumnAvailable<false, false, true, true, false, false>,
    ColumnAvailable<false, false, false, true, true, true>,
    ColumnAvailable<false, false, false, true, true, true>,
    ColumnAvailable<false, false, false, false, false, true>
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
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

}

} // namespace two_wheel_vehicle_model_ekf_A

namespace two_wheel_vehicle_model_ekf_C {

using namespace PythonNumpy;

using SparseAvailable_ekf_C = SparseAvailable<
    ColumnAvailable<true, false, false, false, false, false>,
    ColumnAvailable<false, true, false, false, false, false>,
    ColumnAvailable<false, false, true, false, false, false>,
    ColumnAvailable<false, false, false, true, false, false>,
    ColumnAvailable<false, false, false, false, false, true>
>;

template <typename T>
using type = SparseMatrix_Type<T, SparseAvailable_ekf_C>;

template <typename T>
inline auto make(void) -> type<T> {

    return make_SparseMatrix<SparseAvailable_ekf_C>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

}

} // namespace two_wheel_vehicle_model_ekf_C

namespace two_wheel_vehicle_model_ada_mpc_ekf_parameter {

template <typename T>
class Parameter {
public:
    T m = static_cast<T>(2000);
    T l_f = static_cast<T>(1.4);
    T l_r = static_cast<T>(1.6);
    T I = static_cast<T>(4000);
    T K_f = static_cast<T>(12000.0);
    T K_r = static_cast<T>(11000.0);
};

template <typename T>
using Parameter_Type = Parameter<T>;

} // namespace two_wheel_vehicle_model_ada_mpc_ekf_parameter

template <typename T>
using Parameter_Type = two_wheel_vehicle_model_ada_mpc_ekf_parameter::Parameter_Type<T>;

namespace two_wheel_vehicle_model_ada_mpc_ekf_state_function {

template <typename T>
using A_Type = two_wheel_vehicle_model_ekf_A::type<T>;
template <typename T>
using X_Type = StateSpaceState_Type<T, A_Type<T>::COLS>;
template <typename T>
using U_Type = StateSpaceInput_Type<T, 2>;

template <typename T>
inline auto sympy_function(const T py, const T m, const T beta, const T r, const T l_r, const T l_f, const T K_r, const T K_f, const T px, const T theta, const T delta, const T I, const T accel, const T V) -> X_Type<T> {

    X_Type<T> result;

    T x0 = static_cast<T>(0.01) * V;

    T x1 = K_f * V;

    T x2 = K_f * r;

    T x3 = V * V;

    result.template set<0, 0>(static_cast<T>(px + x0 * cos(theta)));
    result.template set<1, 0>(static_cast<T>(py + x0 * sin(theta)));
    result.template set<2, 0>(static_cast<T>(0.01 * r + theta));
    result.template set<3, 0>(static_cast<T>(r + 0.02 * (K_f * V * delta * l_f + K_r * V * beta * l_r - K_r * (l_r * l_r) * r - beta * l_f * x1 - l_f * l_f * x2) / (I * V)));
    result.template set<4, 0>(static_cast<T>(beta + 0.01 * (2 * K_f * V * delta - 2 * K_r * V * beta + 2 * K_r * l_r * r - 2 * beta * x1 - 2 * l_f * x2 - m * r * x3) / (m * x3)));
    result.template set<5, 0>(static_cast<T>(V + 0.01 * accel));

    return result;
}

template <typename T>
inline auto function(const X_Type<T> X, const U_Type<T> U, const Parameter_Type<T> Parameters) -> X_Type<T> {

    T px = X.template get<0, 0>();

    T py = X.template get<1, 0>();

    T theta = X.template get<2, 0>();

    T r = X.template get<3, 0>();

    T beta = X.template get<4, 0>();

    T V = X.template get<5, 0>();

    T delta = U.template get<0, 0>();

    T accel = U.template get<1, 0>();

    T m = Parameters.m;

    T l_r = Parameters.l_r;

    T l_f = Parameters.l_f;

    T K_r = Parameters.K_r;

    T K_f = Parameters.K_f;

    T I = Parameters.I;

    return sympy_function<T>(py, m, beta, r, l_r, l_f, K_r, K_f, px, theta, delta, I,
        accel, V);
}

} // namespace two_wheel_vehicle_model_ada_mpc_ekf_state_function

namespace two_wheel_vehicle_model_ada_mpc_ekf_state_function_jacobian {

template <typename T>
using A_Type = two_wheel_vehicle_model_ekf_A::type<T>;

template <typename T>
using X_Type = StateSpaceState_Type<T, A_Type<T>::COLS>;

template <typename T>
using U_Type = StateSpaceInput_Type<T, 2>;

template <typename T>
inline auto sympy_function(const T m, const T beta, const T r, const T l_r, const T l_f, const T K_r, const T K_f, const T theta, const T delta, const T I, const T V) -> A_Type<T> {

    A_Type<T> result;

    T x0 = static_cast<T>(0.01) * sin(theta);

    T x1 = static_cast<T>(0.01) * cos(theta);

    T x2 = K_f * (l_f * l_f);

    T x3 = K_r * (l_r * l_r);

    T x4 = static_cast<T>(0.02) / I;

    T x5 = x4 / V;

    T x6 = K_f * l_f;

    T x7 = V * x6;

    T x8 = V * V;

    T x9 = 1 / x8;

    T x10 = 2 * x6;

    T x11 = m * x8;

    T x12 = 1 / m;

    T x13 = static_cast<T>(0.01) * x12 * x9;

    T x14 = 2 * V;

    T x15 = K_f * x14;

    T x16 = K_r * x14;

    T x17 = 2 * beta;

    result.template set<0, 0>(static_cast<T>(1));
    result.template set<0, 1>(static_cast<T>(0));
    result.template set<0, 2>(static_cast<T>(-V * x0));
    result.template set<0, 3>(static_cast<T>(0));
    result.template set<0, 4>(static_cast<T>(0));
    result.template set<0, 5>(static_cast<T>(x1));
    result.template set<1, 0>(static_cast<T>(0));
    result.template set<1, 1>(static_cast<T>(1));
    result.template set<1, 2>(static_cast<T>(V * x1));
    result.template set<1, 3>(static_cast<T>(0));
    result.template set<1, 4>(static_cast<T>(0));
    result.template set<1, 5>(static_cast<T>(x0));
    result.template set<2, 0>(static_cast<T>(0));
    result.template set<2, 1>(static_cast<T>(0));
    result.template set<2, 2>(static_cast<T>(1));
    result.template set<2, 3>(static_cast<T>(0.01));
    result.template set<2, 4>(static_cast<T>(0));
    result.template set<2, 5>(static_cast<T>(0));
    result.template set<3, 0>(static_cast<T>(0));
    result.template set<3, 1>(static_cast<T>(0));
    result.template set<3, 2>(static_cast<T>(0));
    result.template set<3, 3>(static_cast<T>(x5 * (-x2 - x3) + 1));
    result.template set<3, 4>(static_cast<T>(x5 * (K_r * V * l_r - x7)));
    result.template set<3, 5>(static_cast<T>(-x4 * x9 * (K_f * V * delta * l_f + K_r * V * beta * l_r - beta * x7 - r * x2 - r * x3) + x5 * (K_r * beta * l_r - beta * x6 + delta * x6)));
    result.template set<4, 0>(static_cast<T>(0));
    result.template set<4, 1>(static_cast<T>(0));
    result.template set<4, 2>(static_cast<T>(0));
    result.template set<4, 3>(static_cast<T>(x13 * (2 * K_r * l_r - x10 - x11)));
    result.template set<4, 4>(static_cast<T>(x13 * (-x15 - x16) + 1));
    result.template set<4, 5>(static_cast<T>(x13 * (2 * K_f * delta - K_f * x17 - K_r * x17 - m * r * x14) - 0.02 * x12 * (2 * K_f * V * delta + 2 * K_r * l_r * r - beta * x15 - beta * x16 - r * x10 - r * x11) / (V * V * V)));
    result.template set<5, 0>(static_cast<T>(0));
    result.template set<5, 1>(static_cast<T>(0));
    result.template set<5, 2>(static_cast<T>(0));
    result.template set<5, 3>(static_cast<T>(0));
    result.template set<5, 4>(static_cast<T>(0));
    result.template set<5, 5>(static_cast<T>(1));

    return result;
}

template <typename T>
inline auto function(const X_Type<T> X, const U_Type<T> U, const Parameter_Type<T> Parameters) -> A_Type<T> {

    T theta = X.template get<2, 0>();

    T r = X.template get<3, 0>();

    T beta = X.template get<4, 0>();

    T V = X.template get<5, 0>();

    T delta = U.template get<0, 0>();

    T m = Parameters.m;

    T l_r = Parameters.l_r;

    T l_f = Parameters.l_f;

    T K_r = Parameters.K_r;

    T K_f = Parameters.K_f;

    T I = Parameters.I;

    return sympy_function<T>(m, beta, r, l_r, l_f, K_r, K_f, theta, delta, I, V);
}


} // namespace two_wheel_vehicle_model_ada_mpc_ekf_state_function_jacobian

namespace two_wheel_vehicle_model_ada_mpc_ekf_measurement_function {

template <typename T>
using A_Type = two_wheel_vehicle_model_ekf_A::type<T>;

template <typename T>
using C_Type = two_wheel_vehicle_model_ekf_C::type<T>;

template <typename T>
using X_Type = StateSpaceState_Type<T, A_Type<T>::COLS>;

template <typename T>
using Y_Type = StateSpaceOutput_Type<T, C_Type<T>::COLS>;

template <typename T>
inline auto sympy_function(const T px, const T V, const T r, const T py, const T theta) -> Y_Type<T> {

    Y_Type<T> result;

    result.template set<0, 0>(static_cast<T>(px));
    result.template set<1, 0>(static_cast<T>(py));
    result.template set<2, 0>(static_cast<T>(theta));
    result.template set<3, 0>(static_cast<T>(r));
    result.template set<4, 0>(static_cast<T>(V));

    return result;
}

template <typename T>
inline auto function(const X_Type<T> X, const Parameter_Type<T> Parameters) -> Y_Type<T> {
    static_cast<void>(Parameters);

    T px = X.template get<0, 0>();

    T py = X.template get<1, 0>();

    T theta = X.template get<2, 0>();

    T r = X.template get<3, 0>();

    T V = X.template get<5, 0>();

    return sympy_function<T>(px, V, r, py, theta);
}


} // namespace two_wheel_vehicle_model_ada_mpc_ekf_measurement_function

namespace two_wheel_vehicle_model_ada_mpc_ekf_measurement_function_jacobian {

template <typename T>
using A_Type = two_wheel_vehicle_model_ekf_A::type<T>;

template <typename T>
using C_Type = two_wheel_vehicle_model_ekf_C::type<T>;

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
    result.template set<0, 4>(static_cast<T>(0));
    result.template set<0, 5>(static_cast<T>(0));
    result.template set<1, 0>(static_cast<T>(0));
    result.template set<1, 1>(static_cast<T>(1));
    result.template set<1, 2>(static_cast<T>(0));
    result.template set<1, 3>(static_cast<T>(0));
    result.template set<1, 4>(static_cast<T>(0));
    result.template set<1, 5>(static_cast<T>(0));
    result.template set<2, 0>(static_cast<T>(0));
    result.template set<2, 1>(static_cast<T>(0));
    result.template set<2, 2>(static_cast<T>(1));
    result.template set<2, 3>(static_cast<T>(0));
    result.template set<2, 4>(static_cast<T>(0));
    result.template set<2, 5>(static_cast<T>(0));
    result.template set<3, 0>(static_cast<T>(0));
    result.template set<3, 1>(static_cast<T>(0));
    result.template set<3, 2>(static_cast<T>(0));
    result.template set<3, 3>(static_cast<T>(1));
    result.template set<3, 4>(static_cast<T>(0));
    result.template set<3, 5>(static_cast<T>(0));
    result.template set<4, 0>(static_cast<T>(0));
    result.template set<4, 1>(static_cast<T>(0));
    result.template set<4, 2>(static_cast<T>(0));
    result.template set<4, 3>(static_cast<T>(0));
    result.template set<4, 4>(static_cast<T>(0));
    result.template set<4, 5>(static_cast<T>(1));

    return result;
}

template <typename T>
inline auto function(const X_Type<T> X, const Parameter_Type<T> Parameters) -> C_Type<T> {
    static_cast<void>(X);
    static_cast<void>(Parameters);

    return sympy_function<T>();
}


} // namespace two_wheel_vehicle_model_ada_mpc_ekf_measurement_function_jacobian

namespace two_wheel_vehicle_model_ada_mpc_ekf {

using namespace PythonNumpy;
using namespace PythonControl;

constexpr std::size_t NUMBER_OF_DELAY = 0;

template <typename T>
using A_Type = two_wheel_vehicle_model_ekf_A::type<T>;

template <typename T>
using C_Type = two_wheel_vehicle_model_ekf_C::type<T>;

constexpr std::size_t STATE_SIZE = 6;

constexpr std::size_t INPUT_SIZE = 2;

constexpr std::size_t OUTPUT_SIZE = 5;

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
using Parameter_Type = two_wheel_vehicle_model_ada_mpc_ekf_parameter::Parameter_Type<T>;

template <typename T>
using type = ExtendedKalmanFilter_Type<
    A_Type<T>, C_Type<T>, U_Type<T>, Q_Type<T>, R_Type<T>, Parameter_Type<T>, NUMBER_OF_DELAY>;

template <typename T>
inline auto make() -> type<T> {

    auto Q = make_KalmanFilter_Q<STATE_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

    auto R = make_KalmanFilter_R<OUTPUT_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

    Parameter_Type<T> parameters;

    StateFunction_Object<X_Type<T>, U_Type<T>, Parameter_Type<T>> state_function_object =
        [](const X_Type<T>& X, const U_Type<T>& U, const Parameter_Type<T>& Parameters) {
        return two_wheel_vehicle_model_ada_mpc_ekf_state_function::function(X, U, Parameters);

        };

    StateFunctionJacobian_Object<A_Type<T>, X_Type<T>, U_Type<T>, Parameter_Type<T>> state_function_jacobian_object =
        [](const X_Type<T>& X, const U_Type<T>& U, const Parameter_Type<T>& Parameters) {
        return two_wheel_vehicle_model_ada_mpc_ekf_state_function_jacobian::function(X, U, Parameters);

        };

    MeasurementFunction_Object<Y_Type<T>, X_Type<T>, Parameter_Type<T>> measurement_function_object =
        [](const X_Type<T>& X, const Parameter_Type<T>& Parameters) {
        return two_wheel_vehicle_model_ada_mpc_ekf_measurement_function::function(X, Parameters);

        };

    MeasurementFunctionJacobian_Object<C_Type<T>, X_Type<T>, Parameter_Type<T>> measurement_function_jacobian_object =
        [](const X_Type<T>& X, const Parameter_Type<T>& Parameters) {
        return two_wheel_vehicle_model_ada_mpc_ekf_measurement_function_jacobian::function(X, Parameters);

        };

    return ExtendedKalmanFilter_Type<
        A_Type<T>, C_Type<T>, U_Type<T>, Q_Type<T>, R_Type<T>, Parameter_Type<T>, NUMBER_OF_DELAY>(
            Q, R, state_function_object, state_function_jacobian_object,
            measurement_function_object, measurement_function_jacobian_object,
            parameters);

}

} // namespace two_wheel_vehicle_model_ada_mpc_ekf

namespace two_wheel_vehicle_model_ada_mpc_B {

using namespace PythonNumpy;

using SparseAvailable_ada_mpc_B = SparseAvailable<
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<false, true>
>;

template <typename T>
using type = SparseMatrix_Type<T, SparseAvailable_ada_mpc_B>;

template <typename T>
inline auto make(void) -> type<T> {

    return make_SparseMatrix<SparseAvailable_ada_mpc_B>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

}

} // namespace two_wheel_vehicle_model_ada_mpc_B

namespace two_wheel_vehicle_model_ada_mpc_F {

using namespace PythonNumpy;

using SparseAvailable_ada_mpc_F = SparseAvailable <
    ColumnAvailable<true, false, false, false, false, true, true, false, false, false, false>,
    ColumnAvailable<false, true, true, false, false, false, false, true, false, false, false>,
    ColumnAvailable<false, false, true, true, false, false, false, false, true, false, false>,
    ColumnAvailable<false, false, false, true, true, false, false, false, false, true, false>,
    ColumnAvailable<false, false, false, false, false, true, false, false, false, false, true>,
    ColumnAvailable<true, false, false, false, false, true, true, false, false, false, false>,
    ColumnAvailable<false, true, true, true, false, false, false, true, false, false, false>,
    ColumnAvailable<false, false, true, true, true, false, false, false, true, false, false>,
    ColumnAvailable<false, false, false, true, true, false, false, false, false, true, false>,
    ColumnAvailable<false, false, false, false, false, true, false, false, false, false, true>,
    ColumnAvailable<true, false, false, false, false, true, true, false, false, false, false>,
    ColumnAvailable<false, true, true, true, true, false, false, true, false, false, false>,
    ColumnAvailable<false, false, true, true, true, false, false, false, true, false, false>,
    ColumnAvailable<false, false, false, true, true, false, false, false, false, true, false>,
    ColumnAvailable<false, false, false, false, false, true, false, false, false, false, true>,
    ColumnAvailable<true, false, false, false, false, true, true, false, false, false, false>,
    ColumnAvailable<false, true, true, true, true, false, false, true, false, false, false>,
    ColumnAvailable<false, false, true, true, true, false, false, false, true, false, false>,
    ColumnAvailable<false, false, false, true, true, false, false, false, false, true, false>,
    ColumnAvailable<false, false, false, false, false, true, false, false, false, false, true>
> ;

template <typename T>
using type = SparseMatrix_Type<T, SparseAvailable_ada_mpc_F>;

template <typename T>
inline auto make(void) -> type<T> {

    return make_SparseMatrix<SparseAvailable_ada_mpc_F>(
        static_cast<T>(1.0),
        static_cast<T>(0.01),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(0.1),
        static_cast<T>(1.0),
        static_cast<T>(0.05),
        static_cast<T>(0.0005),
        static_cast<T>(1.0),
        static_cast<T>(0.0097416),
        static_cast<T>(4.000000000000001e-05),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(2.0),
        static_cast<T>(0.03),
        static_cast<T>(1.0),
        static_cast<T>(2.0),
        static_cast<T>(0.30000000000000004),
        static_cast<T>(0.001),
        static_cast<T>(1.0),
        static_cast<T>(0.1),
        static_cast<T>(0.00148708),
        static_cast<T>(2.0000000000000003e-06),
        static_cast<T>(1.0),
        static_cast<T>(0.019231080256),
        static_cast<T>(0.00011804640000000003),
        static_cast<T>(1.0),
        static_cast<T>(2.0),
        static_cast<T>(1.0),
        static_cast<T>(3.0),
        static_cast<T>(0.060000000000000005),
        static_cast<T>(1.0),
        static_cast<T>(3.0),
        static_cast<T>(0.6),
        static_cast<T>(0.003974160000000001),
        static_cast<T>(4.000000000000001e-06),
        static_cast<T>(1.0),
        static_cast<T>(0.15000000000000002),
        static_cast<T>(0.0029486340128),
        static_cast<T>(7.902320000000002e-06),
        static_cast<T>(1.0),
        static_cast<T>(0.028474578121896958),
        static_cast<T>(0.00023225565382400005),
        static_cast<T>(1.0),
        static_cast<T>(3.0),
        static_cast<T>(1.0),
        static_cast<T>(4.0),
        static_cast<T>(0.09999999999999999),
        static_cast<T>(1.0),
        static_cast<T>(4.0),
        static_cast<T>(1.0),
        static_cast<T>(0.009871428025600001),
        static_cast<T>(1.9804640000000006e-05),
        static_cast<T>(1.0),
        static_cast<T>(0.2),
        static_cast<T>(0.004872362918894849),
        static_cast<T>(1.9515102691200004e-05),
        static_cast<T>(1.0),
        static_cast<T>(0.03747809104714121),
        static_cast<T>(0.00038081208627363587),
        static_cast<T>(1.0),
        static_cast<T>(4.0),
        static_cast<T>(1.0)
    );

}

} // namespace two_wheel_vehicle_model_ada_mpc_F

namespace two_wheel_vehicle_model_ada_mpc_Phi {

using namespace PythonNumpy;

using SparseAvailable_ada_mpc_Phi = SparseAvailable <
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<false, true>,
    ColumnAvailable<false, true>,
    ColumnAvailable<false, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<false, true>,
    ColumnAvailable<false, true>,
    ColumnAvailable<true, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<false, true>,
    ColumnAvailable<false, true>,
    ColumnAvailable<true, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<false, true>
> ;

template <typename T>
using type = SparseMatrix_Type<T, SparseAvailable_ada_mpc_Phi>;

template <typename T>
inline auto make(void) -> type<T> {

    return make_SparseMatrix<SparseAvailable_ada_mpc_Phi>(
        static_cast<T>(0.0008399999999999999),
        static_cast<T>(0.01),
        static_cast<T>(0.0001),
        static_cast<T>(4.2000000000000004e-05),
        static_cast<T>(0.0016587744),
        static_cast<T>(0.02),
        static_cast<T>(0.0003),
        static_cast<T>(8.400000000000001e-05),
        static_cast<T>(0.00012493872),
        static_cast<T>(0.002456827298304),
        static_cast<T>(0.03),
        static_cast<T>(0.0006000000000000001),
        static_cast<T>(0.0003338774400000001),
        static_cast<T>(0.00024778008491520005),
        static_cast<T>(0.0032346516300852325),
        static_cast<T>(0.04)
    );

}

} // namespace two_wheel_vehicle_model_ada_mpc_Phi

namespace two_wheel_vehicle_model_ada_mpc_solver_factor {

using namespace PythonNumpy;

using SparseAvailable_ada_mpc_solver_factor = SparseAvailable<
    ColumnAvailable<false, false, false, true, false, false, false, true, true, false, false, true, true, true, false, false, true, true, true, false>,
    ColumnAvailable<false, false, false, false, true, true, false, false, false, true, true, false, false, false, true, true, false, false, false, true>
>;

template <typename T>
using type = SparseMatrix_Type<T, SparseAvailable_ada_mpc_solver_factor>;

template <typename T>
inline auto make(void) -> type<T> {

    return make_SparseMatrix<SparseAvailable_ada_mpc_solver_factor>(
        static_cast<T>(0.008398307455265876),
        static_cast<T>(0.00041991537276329385),
        static_cast<T>(0.016584401678719263),
        static_cast<T>(0.0008398307455265877),
        static_cast<T>(0.001249135456699257),
        static_cast<T>(0.02456332263767524),
        static_cast<T>(0.0033381016589251024),
        static_cast<T>(0.0024773015885830195),
        static_cast<T>(0.0323399986906342),
        static_cast<T>(0.09708694504859493),
        static_cast<T>(0.0009708694504859493),
        static_cast<T>(0.19417389009718986),
        static_cast<T>(0.0029126083514578473),
        static_cast<T>(0.29126083514578477),
        static_cast<T>(0.005825216702915696),
        static_cast<T>(0.3883477801943797)
    );

}

} // namespace two_wheel_vehicle_model_ada_mpc_solver_factor

namespace two_wheel_vehicle_model_ada_mpc_Weight_U_Nc {

using namespace PythonNumpy;

template <typename T>
using type = DiagMatrix_Type<T, 2>;

template <typename T>
inline auto make(void) -> type<T> {

    return make_DiagMatrix<2>(
        static_cast<T>(0.1),
        static_cast<T>(0.1)
    );

}

} // namespace two_wheel_vehicle_model_ada_mpc_Weight_U_Nc

namespace two_wheel_vehicle_model_mpc_embedded_integrator_state_space_updater {

template <typename T, typename A_Updater_Output_Type>
class A_Updater {
public:
    static inline auto update(T m, T l_f, T l_r, T I, T K_f, T K_r, T r, T beta, T accel, T delta, T py, T V, T theta, T px) -> A_Updater_Output_Type {
        static_cast<void>(accel);
        static_cast<void>(py);
        static_cast<void>(px);

        return A_Updater::sympy_function(m, K_f, V, beta, I, theta, l_f, delta, K_r, r, l_r);
    }

    static inline auto sympy_function(T m, T K_f, T V, T beta, T I, T theta, T l_f, T delta, T K_r, T r, T l_r) -> A_Updater_Output_Type {
        A_Updater_Output_Type result;

        T x0 = static_cast<T>(0.01) * sin(theta);

        T x1 = -V * x0;

        T x2 = static_cast<T>(0.01) * cos(theta);

        T x3 = V * x2;

        T x4 = K_f * (l_f * l_f);

        T x5 = K_r * (l_r * l_r);

        T x6 = -x4 - x5;

        T x7 = 1 / V;

        T x8 = 1 / I;

        T x9 = static_cast<T>(0.02) * x8;

        T x10 = x7 * x9;

        T x11 = K_f * l_f;

        T x12 = V * x11;

        T x13 = K_r * V * l_r - x12;

        T x14 = K_r * beta * l_r - beta * x11 + delta * x11;

        T x15 = V * V;

        T x16 = 1 / x15;

        T x17 = x16 * (K_f * V * delta * l_f + K_r * V * beta * l_r - beta * x12 - r * x4 -
            r * x5);

        T x18 = 2 * x11;

        T x19 = m * x15;

        T x20 = 1 / m;

        T x21 = static_cast<T>(0.01) * x16 * x20;

        T x22 = 2 * V;

        T x23 = K_f * x22;

        T x24 = K_r * x22;

        T x25 = 2 * beta;

        T x26 = static_cast<T>(0.0002) * x8;

        T x27 = x26 * x7;

        result.template set<0, 0>(static_cast<T>(1));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<0, 2>(static_cast<T>(x1));
        result.template set<0, 3>(static_cast<T>(0));
        result.template set<0, 4>(static_cast<T>(0));
        result.template set<0, 5>(static_cast<T>(x2));
        result.template set<0, 6>(static_cast<T>(0.0));
        result.template set<0, 7>(static_cast<T>(0.0));
        result.template set<0, 8>(static_cast<T>(0.0));
        result.template set<0, 9>(static_cast<T>(0.0));
        result.template set<0, 10>(static_cast<T>(0.0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(1));
        result.template set<1, 2>(static_cast<T>(x3));
        result.template set<1, 3>(static_cast<T>(0));
        result.template set<1, 4>(static_cast<T>(0));
        result.template set<1, 5>(static_cast<T>(x0));
        result.template set<1, 6>(static_cast<T>(0.0));
        result.template set<1, 7>(static_cast<T>(0.0));
        result.template set<1, 8>(static_cast<T>(0.0));
        result.template set<1, 9>(static_cast<T>(0.0));
        result.template set<1, 10>(static_cast<T>(0.0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<2, 2>(static_cast<T>(1));
        result.template set<2, 3>(static_cast<T>(0.01));
        result.template set<2, 4>(static_cast<T>(0));
        result.template set<2, 5>(static_cast<T>(0));
        result.template set<2, 6>(static_cast<T>(0.0));
        result.template set<2, 7>(static_cast<T>(0.0));
        result.template set<2, 8>(static_cast<T>(0.0));
        result.template set<2, 9>(static_cast<T>(0.0));
        result.template set<2, 10>(static_cast<T>(0.0));
        result.template set<3, 0>(static_cast<T>(0));
        result.template set<3, 1>(static_cast<T>(0));
        result.template set<3, 2>(static_cast<T>(0));
        result.template set<3, 3>(static_cast<T>(x10 * x6 + 1));
        result.template set<3, 4>(static_cast<T>(x10 * x13));
        result.template set<3, 5>(static_cast<T>(x10 * x14 - x17 * x9));
        result.template set<3, 6>(static_cast<T>(0.0));
        result.template set<3, 7>(static_cast<T>(0.0));
        result.template set<3, 8>(static_cast<T>(0.0));
        result.template set<3, 9>(static_cast<T>(0.0));
        result.template set<3, 10>(static_cast<T>(0.0));
        result.template set<4, 0>(static_cast<T>(0));
        result.template set<4, 1>(static_cast<T>(0));
        result.template set<4, 2>(static_cast<T>(0));
        result.template set<4, 3>(static_cast<T>(x21 * (2 * K_r * l_r - x18 - x19)));
        result.template set<4, 4>(static_cast<T>(x21 * (-x23 - x24) + 1));
        result.template set<4, 5>(static_cast<T>(x21 * (2 * K_f * delta - K_f * x25 - K_r * x25 - m * r * x22) - 0.02 * x20 * (2 * K_f * V * delta + 2 * K_r * l_r * r - beta * x23 - beta * x24 - r * x18 - r * x19) / (V * V * V)));
        result.template set<4, 6>(static_cast<T>(0.0));
        result.template set<4, 7>(static_cast<T>(0.0));
        result.template set<4, 8>(static_cast<T>(0.0));
        result.template set<4, 9>(static_cast<T>(0.0));
        result.template set<4, 10>(static_cast<T>(0.0));
        result.template set<5, 0>(static_cast<T>(0));
        result.template set<5, 1>(static_cast<T>(0));
        result.template set<5, 2>(static_cast<T>(0));
        result.template set<5, 3>(static_cast<T>(0));
        result.template set<5, 4>(static_cast<T>(0));
        result.template set<5, 5>(static_cast<T>(1));
        result.template set<5, 6>(static_cast<T>(0.0));
        result.template set<5, 7>(static_cast<T>(0.0));
        result.template set<5, 8>(static_cast<T>(0.0));
        result.template set<5, 9>(static_cast<T>(0.0));
        result.template set<5, 10>(static_cast<T>(0.0));
        result.template set<6, 0>(static_cast<T>(1.0));
        result.template set<6, 1>(static_cast<T>(0));
        result.template set<6, 2>(static_cast<T>(x1));
        result.template set<6, 3>(static_cast<T>(0));
        result.template set<6, 4>(static_cast<T>(0));
        result.template set<6, 5>(static_cast<T>(x2));
        result.template set<6, 6>(static_cast<T>(1));
        result.template set<6, 7>(static_cast<T>(0));
        result.template set<6, 8>(static_cast<T>(0));
        result.template set<6, 9>(static_cast<T>(0));
        result.template set<6, 10>(static_cast<T>(0));
        result.template set<7, 0>(static_cast<T>(0));
        result.template set<7, 1>(static_cast<T>(1.0));
        result.template set<7, 2>(static_cast<T>(x3));
        result.template set<7, 3>(static_cast<T>(0));
        result.template set<7, 4>(static_cast<T>(0));
        result.template set<7, 5>(static_cast<T>(x0));
        result.template set<7, 6>(static_cast<T>(0));
        result.template set<7, 7>(static_cast<T>(1));
        result.template set<7, 8>(static_cast<T>(0));
        result.template set<7, 9>(static_cast<T>(0));
        result.template set<7, 10>(static_cast<T>(0));
        result.template set<8, 0>(static_cast<T>(0));
        result.template set<8, 1>(static_cast<T>(0));
        result.template set<8, 2>(static_cast<T>(0.05));
        result.template set<8, 3>(static_cast<T>(0.0005));
        result.template set<8, 4>(static_cast<T>(0));
        result.template set<8, 5>(static_cast<T>(0));
        result.template set<8, 6>(static_cast<T>(0));
        result.template set<8, 7>(static_cast<T>(0));
        result.template set<8, 8>(static_cast<T>(1));
        result.template set<8, 9>(static_cast<T>(0));
        result.template set<8, 10>(static_cast<T>(0));
        result.template set<9, 0>(static_cast<T>(0));
        result.template set<9, 1>(static_cast<T>(0));
        result.template set<9, 2>(static_cast<T>(0));
        result.template set<9, 3>(static_cast<T>(x27 * x6 + 0.01));
        result.template set<9, 4>(static_cast<T>(x13 * x27));
        result.template set<9, 5>(static_cast<T>(x14 * x27 - x17 * x26));
        result.template set<9, 6>(static_cast<T>(0));
        result.template set<9, 7>(static_cast<T>(0));
        result.template set<9, 8>(static_cast<T>(0));
        result.template set<9, 9>(static_cast<T>(1));
        result.template set<9, 10>(static_cast<T>(0));
        result.template set<10, 0>(static_cast<T>(0));
        result.template set<10, 1>(static_cast<T>(0));
        result.template set<10, 2>(static_cast<T>(0));
        result.template set<10, 3>(static_cast<T>(0));
        result.template set<10, 4>(static_cast<T>(0));
        result.template set<10, 5>(static_cast<T>(1.0));
        result.template set<10, 6>(static_cast<T>(0));
        result.template set<10, 7>(static_cast<T>(0));
        result.template set<10, 8>(static_cast<T>(0));
        result.template set<10, 9>(static_cast<T>(0));
        result.template set<10, 10>(static_cast<T>(1));

        return result;
    }


};

template <typename T, typename B_Updater_Output_Type>
class B_Updater {
public:
    static inline auto update(T m, T l_f, T l_r, T I, T K_f, T K_r, T r, T beta, T accel, T delta, T py, T V, T theta, T px) -> B_Updater_Output_Type {
        static_cast<void>(l_r);
        static_cast<void>(K_r);
        static_cast<void>(r);
        static_cast<void>(beta);
        static_cast<void>(accel);
        static_cast<void>(delta);
        static_cast<void>(py);
        static_cast<void>(theta);
        static_cast<void>(px);

        return B_Updater::sympy_function(m, K_f, I, l_f, V);
    }

    static inline auto sympy_function(T m, T K_f, T I, T l_f, T V) -> B_Updater_Output_Type {
        B_Updater_Output_Type result;

        T x0 = static_cast<T>(0.02) * K_f;

        T x1 = l_f / I;

        result.template set<0, 0>(static_cast<T>(0));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<3, 0>(static_cast<T>(x0 * x1));
        result.template set<3, 1>(static_cast<T>(0));
        result.template set<4, 0>(static_cast<T>(x0 / (V * m)));
        result.template set<4, 1>(static_cast<T>(0));
        result.template set<5, 0>(static_cast<T>(0));
        result.template set<5, 1>(static_cast<T>(0.01));
        result.template set<6, 0>(static_cast<T>(0));
        result.template set<6, 1>(static_cast<T>(0));
        result.template set<7, 0>(static_cast<T>(0));
        result.template set<7, 1>(static_cast<T>(0));
        result.template set<8, 0>(static_cast<T>(0));
        result.template set<8, 1>(static_cast<T>(0));
        result.template set<9, 0>(static_cast<T>(0.0002 * K_f * x1));
        result.template set<9, 1>(static_cast<T>(0));
        result.template set<10, 0>(static_cast<T>(0));
        result.template set<10, 1>(static_cast<T>(0.01));

        return result;
    }


};

template <typename T, typename C_Updater_Output_Type>
class C_Updater {
public:
    static inline auto update(T m, T l_f, T l_r, T I, T K_f, T K_r, T r, T beta, T accel, T delta, T py, T V, T theta, T px) -> C_Updater_Output_Type {
        static_cast<void>(m);
        static_cast<void>(l_f);
        static_cast<void>(l_r);
        static_cast<void>(I);
        static_cast<void>(K_f);
        static_cast<void>(K_r);
        static_cast<void>(r);
        static_cast<void>(beta);
        static_cast<void>(accel);
        static_cast<void>(delta);
        static_cast<void>(py);
        static_cast<void>(V);
        static_cast<void>(theta);
        static_cast<void>(px);

        return C_Updater::sympy_function();
    }

    static inline auto sympy_function() -> C_Updater_Output_Type {
        C_Updater_Output_Type result;

        result.template set<0, 0>(static_cast<T>(0.0));
        result.template set<0, 1>(static_cast<T>(0.0));
        result.template set<0, 2>(static_cast<T>(0.0));
        result.template set<0, 3>(static_cast<T>(0.0));
        result.template set<0, 4>(static_cast<T>(0.0));
        result.template set<0, 5>(static_cast<T>(0.0));
        result.template set<0, 6>(static_cast<T>(1));
        result.template set<0, 7>(static_cast<T>(0));
        result.template set<0, 8>(static_cast<T>(0));
        result.template set<0, 9>(static_cast<T>(0));
        result.template set<0, 10>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0.0));
        result.template set<1, 1>(static_cast<T>(0.0));
        result.template set<1, 2>(static_cast<T>(0.0));
        result.template set<1, 3>(static_cast<T>(0.0));
        result.template set<1, 4>(static_cast<T>(0.0));
        result.template set<1, 5>(static_cast<T>(0.0));
        result.template set<1, 6>(static_cast<T>(0));
        result.template set<1, 7>(static_cast<T>(1));
        result.template set<1, 8>(static_cast<T>(0));
        result.template set<1, 9>(static_cast<T>(0));
        result.template set<1, 10>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(0.0));
        result.template set<2, 1>(static_cast<T>(0.0));
        result.template set<2, 2>(static_cast<T>(0.0));
        result.template set<2, 3>(static_cast<T>(0.0));
        result.template set<2, 4>(static_cast<T>(0.0));
        result.template set<2, 5>(static_cast<T>(0.0));
        result.template set<2, 6>(static_cast<T>(0));
        result.template set<2, 7>(static_cast<T>(0));
        result.template set<2, 8>(static_cast<T>(1));
        result.template set<2, 9>(static_cast<T>(0));
        result.template set<2, 10>(static_cast<T>(0));
        result.template set<3, 0>(static_cast<T>(0.0));
        result.template set<3, 1>(static_cast<T>(0.0));
        result.template set<3, 2>(static_cast<T>(0.0));
        result.template set<3, 3>(static_cast<T>(0.0));
        result.template set<3, 4>(static_cast<T>(0.0));
        result.template set<3, 5>(static_cast<T>(0.0));
        result.template set<3, 6>(static_cast<T>(0));
        result.template set<3, 7>(static_cast<T>(0));
        result.template set<3, 8>(static_cast<T>(0));
        result.template set<3, 9>(static_cast<T>(1));
        result.template set<3, 10>(static_cast<T>(0));
        result.template set<4, 0>(static_cast<T>(0.0));
        result.template set<4, 1>(static_cast<T>(0.0));
        result.template set<4, 2>(static_cast<T>(0.0));
        result.template set<4, 3>(static_cast<T>(0.0));
        result.template set<4, 4>(static_cast<T>(0.0));
        result.template set<4, 5>(static_cast<T>(0.0));
        result.template set<4, 6>(static_cast<T>(0));
        result.template set<4, 7>(static_cast<T>(0));
        result.template set<4, 8>(static_cast<T>(0));
        result.template set<4, 9>(static_cast<T>(0));
        result.template set<4, 10>(static_cast<T>(1));

        return result;
    }


};

template <typename T>
class EmbeddedIntegrator_Updater {
public:
    template <typename X_Type, typename U_Type, typename Parameter_Type, typename EmbeddedIntegrator_Updater_Output_Type>
    static inline void update(const X_Type& X, const U_Type& U, const Parameter_Type& parameter, EmbeddedIntegrator_Updater_Output_Type& output) {
        T m = parameter.m;
        T l_f = parameter.l_f;
        T l_r = parameter.l_r;
        T I = parameter.I;
        T K_f = parameter.K_f;
        T K_r = parameter.K_r;
        T px = X.template get<0, 0>();
        T py = X.template get<1, 0>();
        T theta = X.template get<2, 0>();
        T r = X.template get<3, 0>();
        T beta = X.template get<4, 0>();
        T V = X.template get<5, 0>();
        T delta = U.template get<0, 0>();
        T accel = U.template get<1, 0>();

        auto A = A_Updater<T, typename EmbeddedIntegrator_Updater_Output_Type::A_Type>::update(m, l_f, l_r, I, K_f, K_r, r, beta, accel, delta, py, V, theta, px);

        auto B = B_Updater<T, typename EmbeddedIntegrator_Updater_Output_Type::B_Type>::update(m, l_f, l_r, I, K_f, K_r, r, beta, accel, delta, py, V, theta, px);

        auto C = C_Updater<T, typename EmbeddedIntegrator_Updater_Output_Type::C_Type>::update(m, l_f, l_r, I, K_f, K_r, r, beta, accel, delta, py, V, theta, px);

        output.A = A;
        output.B = B;
        output.C = C;
    }
};

} // namespace two_wheel_vehicle_model_mpc_embedded_integrator_state_space_updater

namespace two_wheel_vehicle_model_prediction_matrices_phi_f_updater {

class PredictionMatricesPhiF_Updater {
public:
    template <typename A_Type, typename B_Type, typename C_Type,
        typename Phi_Type, typename F_Type >
    static inline void update(const A_Type& A, const B_Type& B,
        const C_Type& C, Phi_Type& Phi, F_Type& F) {
        auto C_A_1 = C * A;
        auto C_A_2 = C_A_1 * A;
        auto C_A_3 = C_A_2 * A;
        auto C_A_4 = C_A_3 * A;
        auto C_A_0_B = C * B;
        auto C_A_1_B = C_A_1 * B;
        auto C_A_2_B = C_A_2 * B;
        auto C_A_3_B = C_A_3 * B;
        Phi.template set<0, 0>(C_A_0_B.template get<0, 0>());
        Phi.template set<0, 1>(C_A_0_B.template get<0, 1>());
        Phi.template set<1, 0>(C_A_0_B.template get<1, 0>());
        Phi.template set<1, 1>(C_A_0_B.template get<1, 1>());
        Phi.template set<2, 0>(C_A_0_B.template get<2, 0>());
        Phi.template set<2, 1>(C_A_0_B.template get<2, 1>());
        Phi.template set<3, 0>(C_A_0_B.template get<3, 0>());
        Phi.template set<3, 1>(C_A_0_B.template get<3, 1>());
        Phi.template set<4, 0>(C_A_0_B.template get<4, 0>());
        Phi.template set<4, 1>(C_A_0_B.template get<4, 1>());
        Phi.template set<5, 0>(C_A_1_B.template get<0, 0>());
        Phi.template set<5, 1>(C_A_1_B.template get<0, 1>());
        Phi.template set<6, 0>(C_A_1_B.template get<1, 0>());
        Phi.template set<6, 1>(C_A_1_B.template get<1, 1>());
        Phi.template set<7, 0>(C_A_1_B.template get<2, 0>());
        Phi.template set<7, 1>(C_A_1_B.template get<2, 1>());
        Phi.template set<8, 0>(C_A_1_B.template get<3, 0>());
        Phi.template set<8, 1>(C_A_1_B.template get<3, 1>());
        Phi.template set<9, 0>(C_A_1_B.template get<4, 0>());
        Phi.template set<9, 1>(C_A_1_B.template get<4, 1>());
        Phi.template set<10, 0>(C_A_2_B.template get<0, 0>());
        Phi.template set<10, 1>(C_A_2_B.template get<0, 1>());
        Phi.template set<11, 0>(C_A_2_B.template get<1, 0>());
        Phi.template set<11, 1>(C_A_2_B.template get<1, 1>());
        Phi.template set<12, 0>(C_A_2_B.template get<2, 0>());
        Phi.template set<12, 1>(C_A_2_B.template get<2, 1>());
        Phi.template set<13, 0>(C_A_2_B.template get<3, 0>());
        Phi.template set<13, 1>(C_A_2_B.template get<3, 1>());
        Phi.template set<14, 0>(C_A_2_B.template get<4, 0>());
        Phi.template set<14, 1>(C_A_2_B.template get<4, 1>());
        Phi.template set<15, 0>(C_A_3_B.template get<0, 0>());
        Phi.template set<15, 1>(C_A_3_B.template get<0, 1>());
        Phi.template set<16, 0>(C_A_3_B.template get<1, 0>());
        Phi.template set<16, 1>(C_A_3_B.template get<1, 1>());
        Phi.template set<17, 0>(C_A_3_B.template get<2, 0>());
        Phi.template set<17, 1>(C_A_3_B.template get<2, 1>());
        Phi.template set<18, 0>(C_A_3_B.template get<3, 0>());
        Phi.template set<18, 1>(C_A_3_B.template get<3, 1>());
        Phi.template set<19, 0>(C_A_3_B.template get<4, 0>());
        Phi.template set<19, 1>(C_A_3_B.template get<4, 1>());
        F.template set<0, 0>(C_A_1.template get<0, 0>());
        F.template set<0, 1>(C_A_1.template get<0, 1>());
        F.template set<0, 2>(C_A_1.template get<0, 2>());
        F.template set<0, 3>(C_A_1.template get<0, 3>());
        F.template set<0, 4>(C_A_1.template get<0, 4>());
        F.template set<0, 5>(C_A_1.template get<0, 5>());
        F.template set<0, 6>(C_A_1.template get<0, 6>());
        F.template set<0, 7>(C_A_1.template get<0, 7>());
        F.template set<0, 8>(C_A_1.template get<0, 8>());
        F.template set<0, 9>(C_A_1.template get<0, 9>());
        F.template set<0, 10>(C_A_1.template get<0, 10>());
        F.template set<1, 0>(C_A_1.template get<1, 0>());
        F.template set<1, 1>(C_A_1.template get<1, 1>());
        F.template set<1, 2>(C_A_1.template get<1, 2>());
        F.template set<1, 3>(C_A_1.template get<1, 3>());
        F.template set<1, 4>(C_A_1.template get<1, 4>());
        F.template set<1, 5>(C_A_1.template get<1, 5>());
        F.template set<1, 6>(C_A_1.template get<1, 6>());
        F.template set<1, 7>(C_A_1.template get<1, 7>());
        F.template set<1, 8>(C_A_1.template get<1, 8>());
        F.template set<1, 9>(C_A_1.template get<1, 9>());
        F.template set<1, 10>(C_A_1.template get<1, 10>());
        F.template set<2, 0>(C_A_1.template get<2, 0>());
        F.template set<2, 1>(C_A_1.template get<2, 1>());
        F.template set<2, 2>(C_A_1.template get<2, 2>());
        F.template set<2, 3>(C_A_1.template get<2, 3>());
        F.template set<2, 4>(C_A_1.template get<2, 4>());
        F.template set<2, 5>(C_A_1.template get<2, 5>());
        F.template set<2, 6>(C_A_1.template get<2, 6>());
        F.template set<2, 7>(C_A_1.template get<2, 7>());
        F.template set<2, 8>(C_A_1.template get<2, 8>());
        F.template set<2, 9>(C_A_1.template get<2, 9>());
        F.template set<2, 10>(C_A_1.template get<2, 10>());
        F.template set<3, 0>(C_A_1.template get<3, 0>());
        F.template set<3, 1>(C_A_1.template get<3, 1>());
        F.template set<3, 2>(C_A_1.template get<3, 2>());
        F.template set<3, 3>(C_A_1.template get<3, 3>());
        F.template set<3, 4>(C_A_1.template get<3, 4>());
        F.template set<3, 5>(C_A_1.template get<3, 5>());
        F.template set<3, 6>(C_A_1.template get<3, 6>());
        F.template set<3, 7>(C_A_1.template get<3, 7>());
        F.template set<3, 8>(C_A_1.template get<3, 8>());
        F.template set<3, 9>(C_A_1.template get<3, 9>());
        F.template set<3, 10>(C_A_1.template get<3, 10>());
        F.template set<4, 0>(C_A_1.template get<4, 0>());
        F.template set<4, 1>(C_A_1.template get<4, 1>());
        F.template set<4, 2>(C_A_1.template get<4, 2>());
        F.template set<4, 3>(C_A_1.template get<4, 3>());
        F.template set<4, 4>(C_A_1.template get<4, 4>());
        F.template set<4, 5>(C_A_1.template get<4, 5>());
        F.template set<4, 6>(C_A_1.template get<4, 6>());
        F.template set<4, 7>(C_A_1.template get<4, 7>());
        F.template set<4, 8>(C_A_1.template get<4, 8>());
        F.template set<4, 9>(C_A_1.template get<4, 9>());
        F.template set<4, 10>(C_A_1.template get<4, 10>());
        F.template set<5, 0>(C_A_2.template get<0, 0>());
        F.template set<5, 1>(C_A_2.template get<0, 1>());
        F.template set<5, 2>(C_A_2.template get<0, 2>());
        F.template set<5, 3>(C_A_2.template get<0, 3>());
        F.template set<5, 4>(C_A_2.template get<0, 4>());
        F.template set<5, 5>(C_A_2.template get<0, 5>());
        F.template set<5, 6>(C_A_2.template get<0, 6>());
        F.template set<5, 7>(C_A_2.template get<0, 7>());
        F.template set<5, 8>(C_A_2.template get<0, 8>());
        F.template set<5, 9>(C_A_2.template get<0, 9>());
        F.template set<5, 10>(C_A_2.template get<0, 10>());
        F.template set<6, 0>(C_A_2.template get<1, 0>());
        F.template set<6, 1>(C_A_2.template get<1, 1>());
        F.template set<6, 2>(C_A_2.template get<1, 2>());
        F.template set<6, 3>(C_A_2.template get<1, 3>());
        F.template set<6, 4>(C_A_2.template get<1, 4>());
        F.template set<6, 5>(C_A_2.template get<1, 5>());
        F.template set<6, 6>(C_A_2.template get<1, 6>());
        F.template set<6, 7>(C_A_2.template get<1, 7>());
        F.template set<6, 8>(C_A_2.template get<1, 8>());
        F.template set<6, 9>(C_A_2.template get<1, 9>());
        F.template set<6, 10>(C_A_2.template get<1, 10>());
        F.template set<7, 0>(C_A_2.template get<2, 0>());
        F.template set<7, 1>(C_A_2.template get<2, 1>());
        F.template set<7, 2>(C_A_2.template get<2, 2>());
        F.template set<7, 3>(C_A_2.template get<2, 3>());
        F.template set<7, 4>(C_A_2.template get<2, 4>());
        F.template set<7, 5>(C_A_2.template get<2, 5>());
        F.template set<7, 6>(C_A_2.template get<2, 6>());
        F.template set<7, 7>(C_A_2.template get<2, 7>());
        F.template set<7, 8>(C_A_2.template get<2, 8>());
        F.template set<7, 9>(C_A_2.template get<2, 9>());
        F.template set<7, 10>(C_A_2.template get<2, 10>());
        F.template set<8, 0>(C_A_2.template get<3, 0>());
        F.template set<8, 1>(C_A_2.template get<3, 1>());
        F.template set<8, 2>(C_A_2.template get<3, 2>());
        F.template set<8, 3>(C_A_2.template get<3, 3>());
        F.template set<8, 4>(C_A_2.template get<3, 4>());
        F.template set<8, 5>(C_A_2.template get<3, 5>());
        F.template set<8, 6>(C_A_2.template get<3, 6>());
        F.template set<8, 7>(C_A_2.template get<3, 7>());
        F.template set<8, 8>(C_A_2.template get<3, 8>());
        F.template set<8, 9>(C_A_2.template get<3, 9>());
        F.template set<8, 10>(C_A_2.template get<3, 10>());
        F.template set<9, 0>(C_A_2.template get<4, 0>());
        F.template set<9, 1>(C_A_2.template get<4, 1>());
        F.template set<9, 2>(C_A_2.template get<4, 2>());
        F.template set<9, 3>(C_A_2.template get<4, 3>());
        F.template set<9, 4>(C_A_2.template get<4, 4>());
        F.template set<9, 5>(C_A_2.template get<4, 5>());
        F.template set<9, 6>(C_A_2.template get<4, 6>());
        F.template set<9, 7>(C_A_2.template get<4, 7>());
        F.template set<9, 8>(C_A_2.template get<4, 8>());
        F.template set<9, 9>(C_A_2.template get<4, 9>());
        F.template set<9, 10>(C_A_2.template get<4, 10>());
        F.template set<10, 0>(C_A_3.template get<0, 0>());
        F.template set<10, 1>(C_A_3.template get<0, 1>());
        F.template set<10, 2>(C_A_3.template get<0, 2>());
        F.template set<10, 3>(C_A_3.template get<0, 3>());
        F.template set<10, 4>(C_A_3.template get<0, 4>());
        F.template set<10, 5>(C_A_3.template get<0, 5>());
        F.template set<10, 6>(C_A_3.template get<0, 6>());
        F.template set<10, 7>(C_A_3.template get<0, 7>());
        F.template set<10, 8>(C_A_3.template get<0, 8>());
        F.template set<10, 9>(C_A_3.template get<0, 9>());
        F.template set<10, 10>(C_A_3.template get<0, 10>());
        F.template set<11, 0>(C_A_3.template get<1, 0>());
        F.template set<11, 1>(C_A_3.template get<1, 1>());
        F.template set<11, 2>(C_A_3.template get<1, 2>());
        F.template set<11, 3>(C_A_3.template get<1, 3>());
        F.template set<11, 4>(C_A_3.template get<1, 4>());
        F.template set<11, 5>(C_A_3.template get<1, 5>());
        F.template set<11, 6>(C_A_3.template get<1, 6>());
        F.template set<11, 7>(C_A_3.template get<1, 7>());
        F.template set<11, 8>(C_A_3.template get<1, 8>());
        F.template set<11, 9>(C_A_3.template get<1, 9>());
        F.template set<11, 10>(C_A_3.template get<1, 10>());
        F.template set<12, 0>(C_A_3.template get<2, 0>());
        F.template set<12, 1>(C_A_3.template get<2, 1>());
        F.template set<12, 2>(C_A_3.template get<2, 2>());
        F.template set<12, 3>(C_A_3.template get<2, 3>());
        F.template set<12, 4>(C_A_3.template get<2, 4>());
        F.template set<12, 5>(C_A_3.template get<2, 5>());
        F.template set<12, 6>(C_A_3.template get<2, 6>());
        F.template set<12, 7>(C_A_3.template get<2, 7>());
        F.template set<12, 8>(C_A_3.template get<2, 8>());
        F.template set<12, 9>(C_A_3.template get<2, 9>());
        F.template set<12, 10>(C_A_3.template get<2, 10>());
        F.template set<13, 0>(C_A_3.template get<3, 0>());
        F.template set<13, 1>(C_A_3.template get<3, 1>());
        F.template set<13, 2>(C_A_3.template get<3, 2>());
        F.template set<13, 3>(C_A_3.template get<3, 3>());
        F.template set<13, 4>(C_A_3.template get<3, 4>());
        F.template set<13, 5>(C_A_3.template get<3, 5>());
        F.template set<13, 6>(C_A_3.template get<3, 6>());
        F.template set<13, 7>(C_A_3.template get<3, 7>());
        F.template set<13, 8>(C_A_3.template get<3, 8>());
        F.template set<13, 9>(C_A_3.template get<3, 9>());
        F.template set<13, 10>(C_A_3.template get<3, 10>());
        F.template set<14, 0>(C_A_3.template get<4, 0>());
        F.template set<14, 1>(C_A_3.template get<4, 1>());
        F.template set<14, 2>(C_A_3.template get<4, 2>());
        F.template set<14, 3>(C_A_3.template get<4, 3>());
        F.template set<14, 4>(C_A_3.template get<4, 4>());
        F.template set<14, 5>(C_A_3.template get<4, 5>());
        F.template set<14, 6>(C_A_3.template get<4, 6>());
        F.template set<14, 7>(C_A_3.template get<4, 7>());
        F.template set<14, 8>(C_A_3.template get<4, 8>());
        F.template set<14, 9>(C_A_3.template get<4, 9>());
        F.template set<14, 10>(C_A_3.template get<4, 10>());
        F.template set<15, 0>(C_A_4.template get<0, 0>());
        F.template set<15, 1>(C_A_4.template get<0, 1>());
        F.template set<15, 2>(C_A_4.template get<0, 2>());
        F.template set<15, 3>(C_A_4.template get<0, 3>());
        F.template set<15, 4>(C_A_4.template get<0, 4>());
        F.template set<15, 5>(C_A_4.template get<0, 5>());
        F.template set<15, 6>(C_A_4.template get<0, 6>());
        F.template set<15, 7>(C_A_4.template get<0, 7>());
        F.template set<15, 8>(C_A_4.template get<0, 8>());
        F.template set<15, 9>(C_A_4.template get<0, 9>());
        F.template set<15, 10>(C_A_4.template get<0, 10>());
        F.template set<16, 0>(C_A_4.template get<1, 0>());
        F.template set<16, 1>(C_A_4.template get<1, 1>());
        F.template set<16, 2>(C_A_4.template get<1, 2>());
        F.template set<16, 3>(C_A_4.template get<1, 3>());
        F.template set<16, 4>(C_A_4.template get<1, 4>());
        F.template set<16, 5>(C_A_4.template get<1, 5>());
        F.template set<16, 6>(C_A_4.template get<1, 6>());
        F.template set<16, 7>(C_A_4.template get<1, 7>());
        F.template set<16, 8>(C_A_4.template get<1, 8>());
        F.template set<16, 9>(C_A_4.template get<1, 9>());
        F.template set<16, 10>(C_A_4.template get<1, 10>());
        F.template set<17, 0>(C_A_4.template get<2, 0>());
        F.template set<17, 1>(C_A_4.template get<2, 1>());
        F.template set<17, 2>(C_A_4.template get<2, 2>());
        F.template set<17, 3>(C_A_4.template get<2, 3>());
        F.template set<17, 4>(C_A_4.template get<2, 4>());
        F.template set<17, 5>(C_A_4.template get<2, 5>());
        F.template set<17, 6>(C_A_4.template get<2, 6>());
        F.template set<17, 7>(C_A_4.template get<2, 7>());
        F.template set<17, 8>(C_A_4.template get<2, 8>());
        F.template set<17, 9>(C_A_4.template get<2, 9>());
        F.template set<17, 10>(C_A_4.template get<2, 10>());
        F.template set<18, 0>(C_A_4.template get<3, 0>());
        F.template set<18, 1>(C_A_4.template get<3, 1>());
        F.template set<18, 2>(C_A_4.template get<3, 2>());
        F.template set<18, 3>(C_A_4.template get<3, 3>());
        F.template set<18, 4>(C_A_4.template get<3, 4>());
        F.template set<18, 5>(C_A_4.template get<3, 5>());
        F.template set<18, 6>(C_A_4.template get<3, 6>());
        F.template set<18, 7>(C_A_4.template get<3, 7>());
        F.template set<18, 8>(C_A_4.template get<3, 8>());
        F.template set<18, 9>(C_A_4.template get<3, 9>());
        F.template set<18, 10>(C_A_4.template get<3, 10>());
        F.template set<19, 0>(C_A_4.template get<4, 0>());
        F.template set<19, 1>(C_A_4.template get<4, 1>());
        F.template set<19, 2>(C_A_4.template get<4, 2>());
        F.template set<19, 3>(C_A_4.template get<4, 3>());
        F.template set<19, 4>(C_A_4.template get<4, 4>());
        F.template set<19, 5>(C_A_4.template get<4, 5>());
        F.template set<19, 6>(C_A_4.template get<4, 6>());
        F.template set<19, 7>(C_A_4.template get<4, 7>());
        F.template set<19, 8>(C_A_4.template get<4, 8>());
        F.template set<19, 9>(C_A_4.template get<4, 9>());
        F.template set<19, 10>(C_A_4.template get<4, 10>());
    }
};

} // namespace two_wheel_vehicle_model_prediction_matrices_phi_f_updater


namespace two_wheel_vehicle_model_adaptive_mpc_phi_f_updater {

using namespace two_wheel_vehicle_model_mpc_embedded_integrator_state_space_updater;
using namespace two_wheel_vehicle_model_prediction_matrices_phi_f_updater;

template <typename T>
class Adaptive_MPC_Phi_F_Updater {
public:
    template <typename X_Type, typename U_Type, typename Parameter_Type,
        typename Phi_Type, typename F_Type, typename StateSpace_Type>
    static inline void update(const X_Type& X, const U_Type& U,
        const Parameter_Type& parameter, Phi_Type& Phi, F_Type& F) {

        StateSpace_Type state_space;
        EmbeddedIntegrator_Updater<T>::update(X, U, parameter, state_space);

        PredictionMatricesPhiF_Updater::update(
            state_space.A, state_space.B, state_space.C, Phi, F);

    }
};

} // namespace two_wheel_vehicle_model_adaptive_mpc_phi_f_updater



} // namespace PythonMPC_TwoWheelVehicleModelData

#endif // __TEST_MPC_TWO_WHEEL_VEHICLE_MODEL_DATA_HPP__
