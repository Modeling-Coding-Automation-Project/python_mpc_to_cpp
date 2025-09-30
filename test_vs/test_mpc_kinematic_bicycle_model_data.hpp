#ifndef __TEST_MPC_KINEMATIC_BICYCLE_MODEL_DATA_HPP__
#define __TEST_MPC_KINEMATIC_BICYCLE_MODEL_DATA_HPP__

#include "python_mpc.hpp"

namespace PythonMPC_KinematicBicycleModelData {

using namespace PythonMath;
using namespace PythonNumpy;
using namespace PythonControl;
using namespace PythonOptimization;

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

    X_Type<T> result;

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
constexpr std::size_t OUTPUT_SIZE = 4;

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

    PythonControl::StateFunction_Object<X_Type<T>, U_Type<T>, Parameter_Type<T>> state_function_object =
        [](const X_Type<T>& X, const U_Type<T>& U, const Parameter_Type<T>& Parameters) {
        return kinematic_bicycle_model_nmpc_ekf_state_function::function<T>(X, U, Parameters);
        };

    PythonControl::StateFunctionJacobian_Object<A_Type<T>, X_Type<T>, U_Type<T>, Parameter_Type<T>> state_function_jacobian_object =
        [](const X_Type<T>& X, const U_Type<T>& U, const Parameter_Type<T>& Parameters) {
        return kinematic_bicycle_model_nmpc_ekf_state_function_jacobian::function<T>(X, U, Parameters);
        };

    PythonControl::MeasurementFunction_Object<Y_Type<T>, X_Type<T>, Parameter_Type<T>> measurement_function_object =
        [](const X_Type<T>& X, const Parameter_Type<T>& Parameters) {
        return kinematic_bicycle_model_nmpc_ekf_measurement_function::function<T>(X, Parameters);
        };

    PythonControl::MeasurementFunctionJacobian_Object<C_Type<T>, X_Type<T>, Parameter_Type<T>> measurement_function_jacobian_object =
        [](const X_Type<T>& X, const Parameter_Type<T>& Parameters) {
        return kinematic_bicycle_model_nmpc_ekf_measurement_function_jacobian::function<T>(X, Parameters);
        };

    return ExtendedKalmanFilter_Type<
        A_Type<T>, C_Type<T>, U_Type<T>, Q_Type<T>, R_Type<T>, Parameter_Type<T>, NUMBER_OF_DELAY>(
            Q, R, state_function_object, state_function_jacobian_object,
            measurement_function_object, measurement_function_jacobian_object,
            parameters);

}

} // namespace kinematic_bicycle_model_nmpc_ekf

namespace kinematic_bicycle_model_sqp_state_function {

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function(const T delta, const T wheel_base, const T v, const T q3, const T px, const T delta_time, const T q0, const T py) -> X_Type {

        X_Type result;

        T x0 = delta_time * v;

        T x1 = x0 * tan(delta) / (2 * wheel_base);

        T x2 = cos(x1);

        T x3 = sin(x1);

        result.template set<0, 0>(static_cast<T>(px + x0 * (2 * (q0 * q0) - 1)));
        result.template set<1, 0>(static_cast<T>(py + 2 * q0 * q3 * x0));
        result.template set<2, 0>(static_cast<T>(q0 * x2 - q3 * x3));
        result.template set<3, 0>(static_cast<T>(q0 * x3 + q3 * x2));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> X_Type {

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

};

} // namespace kinematic_bicycle_model_sqp_state_function

namespace kinematic_bicycle_model_sqp_measurement_function {

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type, typename Y_Type>
class Function {
public:
    static inline auto sympy_function(const T q3, const T q0, const T py, const T px) -> Y_Type {

        Y_Type result;

        result.template set<0, 0>(static_cast<T>(px));
        result.template set<1, 0>(static_cast<T>(py));
        result.template set<2, 0>(static_cast<T>(q0));
        result.template set<3, 0>(static_cast<T>(q3));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> Y_Type {

        T px = X.template get<0, 0>();

        T py = X.template get<1, 0>();

        T q0 = X.template get<2, 0>();

        T q3 = X.template get<3, 0>();

        return sympy_function(q3, q0, py, px);
    }

};

} // namespace kinematic_bicycle_model_sqp_measurement_function

namespace kinematic_bicycle_model_sqp_state_jacobian_x {

using State_Jacobian_x_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<true, false, true, false>,
    ColumnAvailable<false, true, true, true>,
    ColumnAvailable<false, false, true, true>,
    ColumnAvailable<false, false, true, true>
>;

template <typename T>
using State_Jacobian_x_Type = SparseMatrix_Type<T, State_Jacobian_x_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function(const T delta, const T wheel_base, const T v, const T q3, const T delta_time, const T q0) -> State_Jacobian_x_Type<T> {

        State_Jacobian_x_Type<T> result;

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

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> State_Jacobian_x_Type<T> {

        T q0 = X.template get<2, 0>();

        T q3 = X.template get<3, 0>();

        T v = U.template get<0, 0>();

        T delta = U.template get<1, 0>();

        T wheel_base = Parameters.wheel_base;

        T delta_time = Parameters.delta_time;

        return sympy_function(delta, wheel_base, v, q3, delta_time, q0);
    }

};

} // namespace kinematic_bicycle_model_sqp_state_jacobian_x

namespace kinematic_bicycle_model_sqp_state_jacobian_u {

using State_Jacobian_u_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<true, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<true, true>,
    ColumnAvailable<true, true>
>;

template <typename T>
using State_Jacobian_u_Type = SparseMatrix_Type<T, State_Jacobian_u_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function(const T delta, const T wheel_base, const T v, const T q3, const T delta_time, const T q0) -> State_Jacobian_u_Type<T> {

        State_Jacobian_u_Type<T> result;

        T x0 = delta_time * q0;

        T x1 = static_cast<T>(1) / wheel_base;

        T x2 = tan(delta);

        T x3 = x1 * x2 / static_cast<T>(2);

        T x4 = delta_time * x3;

        T x5 = v * x4;

        T x6 = sin(x5);

        T x7 = x0 * x3;

        T x8 = cos(x5);

        T x9 = q3 * x4;

        T x10 = v * x1 * (x2 * x2 + static_cast<T>(1)) / static_cast<T>(2);

        T x11 = x0 * x10;

        T x12 = delta_time * q3 * x10;

        result.template set<0, 0>(static_cast<T>(delta_time * (static_cast<T>(2) * (q0 * q0) - static_cast<T>(1))));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(2 * q3 * x0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(-x6 * x7 - x8 * x9));
        result.template set<2, 1>(static_cast<T>(-x11 * x6 - x12 * x8));
        result.template set<3, 0>(static_cast<T>(-x6 * x9 + x7 * x8));
        result.template set<3, 1>(static_cast<T>(x11 * x8 - x12 * x6));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> State_Jacobian_u_Type<T> {

        T q0 = X.template get<2, 0>();

        T q3 = X.template get<3, 0>();

        T v = U.template get<0, 0>();

        T delta = U.template get<1, 0>();

        T wheel_base = Parameters.wheel_base;

        T delta_time = Parameters.delta_time;

        return sympy_function(delta, wheel_base, v, q3, delta_time, q0);
    }

};

} // namespace kinematic_bicycle_model_sqp_state_jacobian_u


namespace kinematic_bicycle_model_sqp_measurement_jacobian_x {

using Measurement_Jacobian_x_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<true, false, false, false>,
    ColumnAvailable<false, true, false, false>,
    ColumnAvailable<false, false, true, false>,
    ColumnAvailable<false, false, false, true>
>;

template <typename T>
using Measurement_Jacobian_x_Type = SparseMatrix_Type<T, Measurement_Jacobian_x_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function() -> Measurement_Jacobian_x_Type<T> {

        Measurement_Jacobian_x_Type<T> result;

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

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> Measurement_Jacobian_x_Type<T> {

        return sympy_function();
    }

};

} // namespace kinematic_bicycle_model_sqp_measurement_jacobian_x

namespace kinematic_bicycle_model_sqp_hessian_f_xx {

using State_Hessian_xx_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, true, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, true>,
    ColumnAvailable<false, false, true, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>
>;

template <typename T>
using State_Hessian_xx_Type = SparseMatrix_Type<T, State_Hessian_xx_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function(const T delta_time, const T v) -> State_Hessian_xx_Type<T> {

        State_Hessian_xx_Type<T> result;

        T x0 = delta_time * v;

        T x1 = 2 * x0;

        result.template set<0, 0>(static_cast<T>(0));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<0, 2>(static_cast<T>(0));
        result.template set<0, 3>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<1, 2>(static_cast<T>(0));
        result.template set<1, 3>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<2, 2>(static_cast<T>(4 * x0));
        result.template set<2, 3>(static_cast<T>(0));
        result.template set<3, 0>(static_cast<T>(0));
        result.template set<3, 1>(static_cast<T>(0));
        result.template set<3, 2>(static_cast<T>(0));
        result.template set<3, 3>(static_cast<T>(0));
        result.template set<4, 0>(static_cast<T>(0));
        result.template set<4, 1>(static_cast<T>(0));
        result.template set<4, 2>(static_cast<T>(0));
        result.template set<4, 3>(static_cast<T>(0));
        result.template set<5, 0>(static_cast<T>(0));
        result.template set<5, 1>(static_cast<T>(0));
        result.template set<5, 2>(static_cast<T>(0));
        result.template set<5, 3>(static_cast<T>(0));
        result.template set<6, 0>(static_cast<T>(0));
        result.template set<6, 1>(static_cast<T>(0));
        result.template set<6, 2>(static_cast<T>(0));
        result.template set<6, 3>(static_cast<T>(x1));
        result.template set<7, 0>(static_cast<T>(0));
        result.template set<7, 1>(static_cast<T>(0));
        result.template set<7, 2>(static_cast<T>(x1));
        result.template set<7, 3>(static_cast<T>(0));
        result.template set<8, 0>(static_cast<T>(0));
        result.template set<8, 1>(static_cast<T>(0));
        result.template set<8, 2>(static_cast<T>(0));
        result.template set<8, 3>(static_cast<T>(0));
        result.template set<9, 0>(static_cast<T>(0));
        result.template set<9, 1>(static_cast<T>(0));
        result.template set<9, 2>(static_cast<T>(0));
        result.template set<9, 3>(static_cast<T>(0));
        result.template set<10, 0>(static_cast<T>(0));
        result.template set<10, 1>(static_cast<T>(0));
        result.template set<10, 2>(static_cast<T>(0));
        result.template set<10, 3>(static_cast<T>(0));
        result.template set<11, 0>(static_cast<T>(0));
        result.template set<11, 1>(static_cast<T>(0));
        result.template set<11, 2>(static_cast<T>(0));
        result.template set<11, 3>(static_cast<T>(0));
        result.template set<12, 0>(static_cast<T>(0));
        result.template set<12, 1>(static_cast<T>(0));
        result.template set<12, 2>(static_cast<T>(0));
        result.template set<12, 3>(static_cast<T>(0));
        result.template set<13, 0>(static_cast<T>(0));
        result.template set<13, 1>(static_cast<T>(0));
        result.template set<13, 2>(static_cast<T>(0));
        result.template set<13, 3>(static_cast<T>(0));
        result.template set<14, 0>(static_cast<T>(0));
        result.template set<14, 1>(static_cast<T>(0));
        result.template set<14, 2>(static_cast<T>(0));
        result.template set<14, 3>(static_cast<T>(0));
        result.template set<15, 0>(static_cast<T>(0));
        result.template set<15, 1>(static_cast<T>(0));
        result.template set<15, 2>(static_cast<T>(0));
        result.template set<15, 3>(static_cast<T>(0));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> State_Hessian_xx_Type<T> {

        T v = U.template get<0, 0>();

        T delta_time = Parameters.delta_time;

        return sympy_function(delta_time, v);
    }

};

} // namespace kinematic_bicycle_model_sqp_hessian_f_xx

namespace kinematic_bicycle_model_sqp_hessian_f_xu {

using State_Hessian_xu_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<true, true>,
    ColumnAvailable<true, true>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<true, true>,
    ColumnAvailable<true, true>
>;

template <typename T>
using State_Hessian_xu_Type = SparseMatrix_Type<T, State_Hessian_xu_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function(const T delta, const T wheel_base, const T v, const T q3, const T delta_time, const T q0) -> State_Hessian_xu_Type<T> {

        State_Hessian_xu_Type<T> result;

        T x0 = delta_time * q0;

        T x1 = static_cast<T>(1) / wheel_base;

        T x2 = tan(delta);

        T x3 = delta_time * x1 * x2 / static_cast<T>(2);

        T x4 = v * x3;

        T x5 = sin(x4);

        T x6 = -x3 * x5;

        T x7 = delta_time * v * x1 * (x2 * x2 + 1) / static_cast<T>(2);

        T x8 = -x5 * x7;

        T x9 = cos(x4);

        T x10 = x3 * x9;

        T x11 = x7 * x9;

        result.template set<0, 0>(static_cast<T>(0));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(4 * x0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<3, 0>(static_cast<T>(0));
        result.template set<3, 1>(static_cast<T>(0));
        result.template set<4, 0>(static_cast<T>(0));
        result.template set<4, 1>(static_cast<T>(0));
        result.template set<5, 0>(static_cast<T>(0));
        result.template set<5, 1>(static_cast<T>(0));
        result.template set<6, 0>(static_cast<T>(2 * delta_time * q3));
        result.template set<6, 1>(static_cast<T>(0));
        result.template set<7, 0>(static_cast<T>(2 * x0));
        result.template set<7, 1>(static_cast<T>(0));
        result.template set<8, 0>(static_cast<T>(0));
        result.template set<8, 1>(static_cast<T>(0));
        result.template set<9, 0>(static_cast<T>(0));
        result.template set<9, 1>(static_cast<T>(0));
        result.template set<10, 0>(static_cast<T>(x6));
        result.template set<10, 1>(static_cast<T>(x8));
        result.template set<11, 0>(static_cast<T>(-x10));
        result.template set<11, 1>(static_cast<T>(-x11));
        result.template set<12, 0>(static_cast<T>(0));
        result.template set<12, 1>(static_cast<T>(0));
        result.template set<13, 0>(static_cast<T>(0));
        result.template set<13, 1>(static_cast<T>(0));
        result.template set<14, 0>(static_cast<T>(x10));
        result.template set<14, 1>(static_cast<T>(x11));
        result.template set<15, 0>(static_cast<T>(x6));
        result.template set<15, 1>(static_cast<T>(x8));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> State_Hessian_xu_Type<T> {

        T q0 = X.template get<2, 0>();

        T q3 = X.template get<3, 0>();

        T v = U.template get<0, 0>();

        T delta = U.template get<1, 0>();

        T wheel_base = Parameters.wheel_base;

        T delta_time = Parameters.delta_time;

        return sympy_function(delta, wheel_base, v, q3, delta_time, q0);
    }

};

} // namespace kinematic_bicycle_model_sqp_hessian_f_xu

namespace kinematic_bicycle_model_sqp_hessian_f_ux {

using State_Hessian_ux_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<false, false, true, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, true, true>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, true, true>,
    ColumnAvailable<false, false, true, true>,
    ColumnAvailable<false, false, true, true>,
    ColumnAvailable<false, false, true, true>
>;

template <typename T>
using State_Hessian_ux_Type = SparseMatrix_Type<T, State_Hessian_ux_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function(const T delta, const T wheel_base, const T v, const T q3, const T delta_time, const T q0) -> State_Hessian_ux_Type<T> {

        State_Hessian_ux_Type<T> result;

        T x0 = delta_time * q0;

        T x1 = static_cast<T>(1) / wheel_base;

        T x2 = tan(delta);

        T x3 = delta_time * x1 * x2 / static_cast<T>(2);

        T x4 = v * x3;

        T x5 = sin(x4);

        T x6 = -x3 * x5;

        T x7 = cos(x4);

        T x8 = x3 * x7;

        T x9 = delta_time * v * x1 * (x2 * x2 + 1) / static_cast<T>(2);

        T x10 = -x5 * x9;

        T x11 = x7 * x9;

        result.template set<0, 0>(static_cast<T>(0));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<0, 2>(static_cast<T>(4 * x0));
        result.template set<0, 3>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<1, 2>(static_cast<T>(0));
        result.template set<1, 3>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<2, 2>(static_cast<T>(2 * delta_time * q3));
        result.template set<2, 3>(static_cast<T>(2 * x0));
        result.template set<3, 0>(static_cast<T>(0));
        result.template set<3, 1>(static_cast<T>(0));
        result.template set<3, 2>(static_cast<T>(0));
        result.template set<3, 3>(static_cast<T>(0));
        result.template set<4, 0>(static_cast<T>(0));
        result.template set<4, 1>(static_cast<T>(0));
        result.template set<4, 2>(static_cast<T>(x6));
        result.template set<4, 3>(static_cast<T>(-x8));
        result.template set<5, 0>(static_cast<T>(0));
        result.template set<5, 1>(static_cast<T>(0));
        result.template set<5, 2>(static_cast<T>(x10));
        result.template set<5, 3>(static_cast<T>(-x11));
        result.template set<6, 0>(static_cast<T>(0));
        result.template set<6, 1>(static_cast<T>(0));
        result.template set<6, 2>(static_cast<T>(x8));
        result.template set<6, 3>(static_cast<T>(x6));
        result.template set<7, 0>(static_cast<T>(0));
        result.template set<7, 1>(static_cast<T>(0));
        result.template set<7, 2>(static_cast<T>(x11));
        result.template set<7, 3>(static_cast<T>(x10));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> State_Hessian_ux_Type<T> {

        T q0 = X.template get<2, 0>();

        T q3 = X.template get<3, 0>();

        T v = U.template get<0, 0>();

        T delta = U.template get<1, 0>();

        T wheel_base = Parameters.wheel_base;

        T delta_time = Parameters.delta_time;

        return sympy_function(delta, wheel_base, v, q3, delta_time, q0);
    }

};

} // namespace kinematic_bicycle_model_sqp_hessian_f_ux

namespace kinematic_bicycle_model_sqp_hessian_f_uu {

using State_Hessian_uu_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<true, true>,
    ColumnAvailable<true, true>,
    ColumnAvailable<true, true>,
    ColumnAvailable<true, true>
>;

template <typename T>
using State_Hessian_uu_Type = SparseMatrix_Type<T, State_Hessian_uu_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function(const T delta, const T wheel_base, const T v, const T q3, const T delta_time, const T q0) -> State_Hessian_uu_Type<T> {

        State_Hessian_uu_Type<T> result;

        T x0 = tan(delta);

        T x1 = v * x0;

        T x2 = static_cast<T>(1) / wheel_base;

        T x3 = delta_time * x2 / static_cast<T>(2);

        T x4 = x1 * x3;

        T x5 = cos(x4);

        T x6 = q0 * x5;

        T x7 = x0 * x0;

        T x8 = delta_time * delta_time;

        T x9 = static_cast<T>(1) / wheel_base / wheel_base;

        T x10 = x8 * x9 / static_cast<T>(4);

        T x11 = x10 * x7;

        T x12 = sin(x4);

        T x13 = x7 + 1;

        T x14 = x13 * x3;

        T x15 = q0 * x12;

        T x16 = q3 * x5;

        T x17 = x10 * x6;

        T x18 = x1 * x13;

        T x19 = q3 * v * x0 * x12 * x13 * x8 * x9 / 4 - x14 * x15 - x14 * x16 - x17 * x18;

        T x20 = 2 * x7 + static_cast<T>(2);

        T x21 = x20 * x4;

        T x22 = v * v;

        T x23 = x13 * x13;

        T x24 = x22 * x23;

        T x25 = q3 * x12;

        T x26 = x10 * x18;

        T x27 = delta_time * q0 * x13 * x2 * x5 / static_cast<T>(2) - x14 * x25 - x15 * x26 - x16 * x26;

        T x28 = x10 * x24;

        result.template set<0, 0>(static_cast<T>(0));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<3, 0>(static_cast<T>(0));
        result.template set<3, 1>(static_cast<T>(0));
        result.template set<4, 0>(static_cast<T>(q3 * x12 * x7 * x8 * x9 / 4 - x11 * x6));
        result.template set<4, 1>(static_cast<T>(x19));
        result.template set<5, 0>(static_cast<T>(x19));
        result.template set<5, 1>(static_cast<T>(q3 * x12 * x22 * x23 * x8 * x9 / 4 - x15 * x21 - x16 * x21 - x17 * x24));
        result.template set<6, 0>(static_cast<T>(-q0 * x11 * x12 - x11 * x16));
        result.template set<6, 1>(static_cast<T>(x27));
        result.template set<7, 0>(static_cast<T>(x27));
        result.template set<7, 1>(static_cast<T>(delta_time * q0 * v * x0 * x2 * x20 * x5 / 2 - x15 * x28 - x16 * x28 - x21 * x25));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> State_Hessian_uu_Type<T> {

        T q0 = X.template get<2, 0>();

        T q3 = X.template get<3, 0>();

        T v = U.template get<0, 0>();

        T delta = U.template get<1, 0>();

        T wheel_base = Parameters.wheel_base;

        T delta_time = Parameters.delta_time;

        return sympy_function(delta, wheel_base, v, q3, delta_time, q0);
    }

};

} // namespace kinematic_bicycle_model_sqp_hessian_f_uu

namespace kinematic_bicycle_model_sqp_hessian_h_xx {

using Measurement_Hessian_xx_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>
>;

template <typename T>
using Measurement_Hessian_xx_Type = SparseMatrix_Type<T, Measurement_Hessian_xx_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function() -> Measurement_Hessian_xx_Type<T> {

        Measurement_Hessian_xx_Type<T> result;

        result.template set<0, 0>(static_cast<T>(0));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<0, 2>(static_cast<T>(0));
        result.template set<0, 3>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<1, 2>(static_cast<T>(0));
        result.template set<1, 3>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<2, 2>(static_cast<T>(0));
        result.template set<2, 3>(static_cast<T>(0));
        result.template set<3, 0>(static_cast<T>(0));
        result.template set<3, 1>(static_cast<T>(0));
        result.template set<3, 2>(static_cast<T>(0));
        result.template set<3, 3>(static_cast<T>(0));
        result.template set<4, 0>(static_cast<T>(0));
        result.template set<4, 1>(static_cast<T>(0));
        result.template set<4, 2>(static_cast<T>(0));
        result.template set<4, 3>(static_cast<T>(0));
        result.template set<5, 0>(static_cast<T>(0));
        result.template set<5, 1>(static_cast<T>(0));
        result.template set<5, 2>(static_cast<T>(0));
        result.template set<5, 3>(static_cast<T>(0));
        result.template set<6, 0>(static_cast<T>(0));
        result.template set<6, 1>(static_cast<T>(0));
        result.template set<6, 2>(static_cast<T>(0));
        result.template set<6, 3>(static_cast<T>(0));
        result.template set<7, 0>(static_cast<T>(0));
        result.template set<7, 1>(static_cast<T>(0));
        result.template set<7, 2>(static_cast<T>(0));
        result.template set<7, 3>(static_cast<T>(0));
        result.template set<8, 0>(static_cast<T>(0));
        result.template set<8, 1>(static_cast<T>(0));
        result.template set<8, 2>(static_cast<T>(0));
        result.template set<8, 3>(static_cast<T>(0));
        result.template set<9, 0>(static_cast<T>(0));
        result.template set<9, 1>(static_cast<T>(0));
        result.template set<9, 2>(static_cast<T>(0));
        result.template set<9, 3>(static_cast<T>(0));
        result.template set<10, 0>(static_cast<T>(0));
        result.template set<10, 1>(static_cast<T>(0));
        result.template set<10, 2>(static_cast<T>(0));
        result.template set<10, 3>(static_cast<T>(0));
        result.template set<11, 0>(static_cast<T>(0));
        result.template set<11, 1>(static_cast<T>(0));
        result.template set<11, 2>(static_cast<T>(0));
        result.template set<11, 3>(static_cast<T>(0));
        result.template set<12, 0>(static_cast<T>(0));
        result.template set<12, 1>(static_cast<T>(0));
        result.template set<12, 2>(static_cast<T>(0));
        result.template set<12, 3>(static_cast<T>(0));
        result.template set<13, 0>(static_cast<T>(0));
        result.template set<13, 1>(static_cast<T>(0));
        result.template set<13, 2>(static_cast<T>(0));
        result.template set<13, 3>(static_cast<T>(0));
        result.template set<14, 0>(static_cast<T>(0));
        result.template set<14, 1>(static_cast<T>(0));
        result.template set<14, 2>(static_cast<T>(0));
        result.template set<14, 3>(static_cast<T>(0));
        result.template set<15, 0>(static_cast<T>(0));
        result.template set<15, 1>(static_cast<T>(0));
        result.template set<15, 2>(static_cast<T>(0));
        result.template set<15, 3>(static_cast<T>(0));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> Measurement_Hessian_xx_Type<T> {

        return sympy_function();
    }

};

} // namespace kinematic_bicycle_model_sqp_hessian_h_xx

namespace kinematic_bicycle_model_cost_matrices_U_min {

template <typename T>
using type = DenseMatrix_Type<T, 2, 1>;

template <typename T>
inline auto make(void) -> type<T> {

    return make_DenseMatrix<2, 1>(
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

}

} // namespace kinematic_bicycle_model_cost_matrices_U_min

namespace kinematic_bicycle_model_cost_matrices_U_max {

template <typename T>
using type = DenseMatrix_Type<T, 2, 1>;

template <typename T>
inline auto make(void) -> type<T> {

    return make_DenseMatrix<2, 1>(
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

}

} // namespace kinematic_bicycle_model_cost_matrices_U_max

namespace kinematic_bicycle_model_cost_matrices_Y_min {

using SparseAvailable_cost_matrices_Y_min = SparseAvailable<
    ColumnAvailable<false>,
    ColumnAvailable<false>,
    ColumnAvailable<false>,
    ColumnAvailable<false>
>;

template <typename T>
using type = SparseMatrix_Type<T, SparseAvailable_cost_matrices_Y_min>;

template <typename T>
inline auto make(void) -> type<T> {

    return make_SparseMatrixEmpty<T, 4, 1>();

}

} // namespace kinematic_bicycle_model_cost_matrices_Y_min

namespace kinematic_bicycle_model_cost_matrices_Y_max {

using SparseAvailable_cost_matrices_Y_max = SparseAvailable<
    ColumnAvailable<false>,
    ColumnAvailable<false>,
    ColumnAvailable<false>,
    ColumnAvailable<false>
>;

template <typename T>
using type = SparseMatrix_Type<T, SparseAvailable_cost_matrices_Y_max>;

template <typename T>
inline auto make(void) -> type<T> {

    return make_SparseMatrixEmpty<T, 4, 1>();

}

} // namespace kinematic_bicycle_model_cost_matrices_Y_max

namespace kinematic_bicycle_model_cost_matrices {

constexpr std::size_t NP = 10;

constexpr std::size_t INPUT_SIZE = 2;
constexpr std::size_t STATE_SIZE = 4;
constexpr std::size_t OUTPUT_SIZE = 4;

template <typename T>
using X_Type = StateSpaceState_Type<T, STATE_SIZE>;

template <typename T>
using U_Type = StateSpaceInput_Type<T, INPUT_SIZE>;

template <typename T>
using Y_Type = StateSpaceOutput_Type<T, OUTPUT_SIZE>;

template <typename T>
using State_Jacobian_X_Matrix_Type = kinematic_bicycle_model_sqp_state_jacobian_x::State_Jacobian_x_Type<T>;

template <typename T>
using State_Jacobian_U_Matrix_Type = kinematic_bicycle_model_sqp_state_jacobian_u::State_Jacobian_u_Type<T>;

template <typename T>
using Measurement_Jacobian_X_Matrix_Type = kinematic_bicycle_model_sqp_measurement_jacobian_x::Measurement_Jacobian_x_Type<T>;

template <typename T>
using State_Hessian_XX_Matrix_Type = kinematic_bicycle_model_sqp_hessian_f_xx::State_Hessian_xx_Type<T>;

template <typename T>
using State_Hessian_XU_Matrix_Type = kinematic_bicycle_model_sqp_hessian_f_xu::State_Hessian_xu_Type<T>;

template <typename T>
using State_Hessian_UX_Matrix_Type = kinematic_bicycle_model_sqp_hessian_f_ux::State_Hessian_ux_Type<T>;

template <typename T>
using State_Hessian_UU_Matrix_Type = kinematic_bicycle_model_sqp_hessian_f_uu::State_Hessian_uu_Type<T>;

template <typename T>
using Measurement_Hessian_XX_Matrix_Type = kinematic_bicycle_model_sqp_hessian_h_xx::Measurement_Hessian_xx_Type<T>;

template <typename T>
using Qx_Type = DiagMatrix_Type<T, STATE_SIZE>;

template <typename T>
using R_Type = DiagMatrix_Type<T, INPUT_SIZE>;

template <typename T>
using Qy_Type = DiagMatrix_Type<T, OUTPUT_SIZE>;

template <typename T>
using U_Min_Type = kinematic_bicycle_model_cost_matrices_U_min::type<T>;

template <typename T>
using U_Max_Type = kinematic_bicycle_model_cost_matrices_U_max::type<T>;

template <typename T>
using Y_Min_Type = kinematic_bicycle_model_cost_matrices_Y_min::type<T>;

template <typename T>
using Y_Max_Type = kinematic_bicycle_model_cost_matrices_Y_max::type<T>;

template <typename T>
using Reference_Trajectory_Type = DenseMatrix_Type<T, OUTPUT_SIZE, (NP + 1)>;

template <typename T>
using type = SQP_CostMatrices_NMPC_Type<T, NP, Parameter_Type<T>,
    U_Min_Type<T>, U_Max_Type<T>, Y_Min_Type<T>, Y_Max_Type<T>,
    State_Jacobian_X_Matrix_Type<T>,
    State_Jacobian_U_Matrix_Type<T>,
    Measurement_Jacobian_X_Matrix_Type<T>,
    State_Hessian_XX_Matrix_Type<T>,
    State_Hessian_XU_Matrix_Type<T>,
    State_Hessian_UX_Matrix_Type<T>,
    State_Hessian_UU_Matrix_Type<T>,
    Measurement_Hessian_XX_Matrix_Type<T>>;

template <typename T>
inline auto make() -> type<T> {

    auto U_min = kinematic_bicycle_model_cost_matrices_U_min::make<T>();

    U_min.template set<0, 0>(static_cast<T>(-1.0));
    U_min.template set<1, 0>(static_cast<T>(-1.5));

    auto U_max = kinematic_bicycle_model_cost_matrices_U_max::make<T>();

    U_max.template set<0, 0>(static_cast<T>(1.0));
    U_max.template set<1, 0>(static_cast<T>(1.5));

    auto Y_min = kinematic_bicycle_model_cost_matrices_Y_min::make<T>();

    auto Y_max = kinematic_bicycle_model_cost_matrices_Y_max::make<T>();

    Qx_Type<T> Qx = make_DiagMatrix<STATE_SIZE>(
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.0));

    R_Type<T> R = make_DiagMatrix<INPUT_SIZE>(
        static_cast<T>(0.05),
        static_cast<T>(0.05));

    Qy_Type<T> Qy = make_DiagMatrix<OUTPUT_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0));

    Reference_Trajectory_Type<T> reference_trajectory;

    type<T> cost_matrices =
        make_SQP_CostMatrices_NMPC<T, NP, Parameter_Type,
        U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
        State_Jacobian_X_Matrix_Type,
        State_Jacobian_U_Matrix_Type,
        Measurement_Jacobian_X_Matrix_Type,
        State_Hessian_XX_Matrix_Type,
        State_Hessian_XU_Matrix_Type,
        State_Hessian_UX_Matrix_Type,
        State_Hessian_UU_Matrix_Type,
        Measurement_Hessian_XX_Matrix_Type>(
            Qx, R, Qy, U_min, U_max, Y_min, Y_max);

    PythonOptimization::StateFunction_Object<X_Type<T>, U_Type<T>, Parameter_Type<T>> STATE_FUNCTION =
        kinematic_bicycle_model_sqp_state_function::Function<T, X_Type<T>, U_Type<T>, Parameter_Type<T>>::function;

    PythonOptimization::MeasurementFunction_Object<Y_Type<T>, X_Type<T>, U_Type<T>, Parameter_Type<T>> MEASUREMENT_FUNCTION =
        kinematic_bicycle_model_sqp_measurement_function::Function<T, X_Type<T>, U_Type<T>, Parameter_Type<T>, Y_Type<T>>::function;

    PythonOptimization::StateFunctionJacobian_X_Object<
        State_Jacobian_X_Matrix_Type<T>, X_Type<T>, U_Type<T>, Parameter_Type<T>> STATE_JACOBIAN_X_FUNCTION =
        kinematic_bicycle_model_sqp_state_jacobian_x::Function<T, X_Type<T>, U_Type<T>, Parameter_Type<T>>::function;

    PythonOptimization::StateFunctionJacobian_U_Object<
        State_Jacobian_U_Matrix_Type<T>, X_Type<T>, U_Type<T>, Parameter_Type<T>> STATE_JACOBIAN_U_FUNCTION =
        kinematic_bicycle_model_sqp_state_jacobian_u::Function<T, X_Type<T>, U_Type<T>, Parameter_Type<T>>::function;

    PythonOptimization::MeasurementFunctionJacobian_X_Object<
        Measurement_Jacobian_X_Matrix_Type<T>, X_Type<T>, U_Type<T>, Parameter_Type<T>> MEASUREMENT_JACOBIAN_X_FUNCTION =
        kinematic_bicycle_model_sqp_measurement_jacobian_x::Function<T, X_Type<T>, U_Type<T>, Parameter_Type<T>>::function;

    PythonOptimization::StateFunctionHessian_XX_Object<
        State_Hessian_XX_Matrix_Type<T>, X_Type<T>, U_Type<T>, Parameter_Type<T>> STATE_HESSIAN_XX_FUNCTION =
        kinematic_bicycle_model_sqp_hessian_f_xx::Function<T, X_Type<T>, U_Type<T>, Parameter_Type<T>>::function;

    PythonOptimization::StateFunctionHessian_XU_Object<
        State_Hessian_XU_Matrix_Type<T>, X_Type<T>, U_Type<T>, Parameter_Type<T>> STATE_HESSIAN_XU_FUNCTION =
        kinematic_bicycle_model_sqp_hessian_f_xu::Function<T, X_Type<T>, U_Type<T>, Parameter_Type<T>>::function;

    PythonOptimization::StateFunctionHessian_UX_Object<
        State_Hessian_UX_Matrix_Type<T>, X_Type<T>, U_Type<T>, Parameter_Type<T>> STATE_HESSIAN_UX_FUNCTION =
        kinematic_bicycle_model_sqp_hessian_f_ux::Function<T, X_Type<T>, U_Type<T>, Parameter_Type<T>>::function;

    PythonOptimization::StateFunctionHessian_UU_Object<
        State_Hessian_UU_Matrix_Type<T>, X_Type<T>, U_Type<T>, Parameter_Type<T>> STATE_HESSIAN_UU_FUNCTION =
        kinematic_bicycle_model_sqp_hessian_f_uu::Function<T, X_Type<T>, U_Type<T>, Parameter_Type<T>>::function;

    PythonOptimization::MeasurementFunctionHessian_XX_Object<
        Measurement_Hessian_XX_Matrix_Type<T>, X_Type<T>, U_Type<T>, Parameter_Type<T>> MEASUREMENT_HESSIAN_XX_FUNCTION =
        kinematic_bicycle_model_sqp_hessian_h_xx::Function<T, X_Type<T>, U_Type<T>, Parameter_Type<T>>::function;

    cost_matrices.set_function_objects(
        STATE_FUNCTION,
        MEASUREMENT_FUNCTION,
        STATE_JACOBIAN_X_FUNCTION,
        STATE_JACOBIAN_U_FUNCTION,
        MEASUREMENT_JACOBIAN_X_FUNCTION,
        STATE_HESSIAN_XX_FUNCTION,
        STATE_HESSIAN_XU_FUNCTION,
        STATE_HESSIAN_UX_FUNCTION,
        STATE_HESSIAN_UU_FUNCTION,
        MEASUREMENT_HESSIAN_XX_FUNCTION
    );

    return cost_matrices;

}

} // namespace kinematic_bicycle_model_cost_matrices


} // namespace PythonMPC_KinematicBicycleModelData

#endif // __TEST_MPC_KINEMATIC_BICYCLE_MODEL_DATA_HPP__
