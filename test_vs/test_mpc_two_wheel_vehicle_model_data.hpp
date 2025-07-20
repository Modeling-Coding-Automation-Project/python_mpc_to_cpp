#ifndef __TEST_MPC_TWO_WHEEL_VEHICLE_MODEL_DATA_HPP__
#define __TEST_MPC_TWO_WHEEL_VEHICLE_MODEL_DATA_HPP__

#include "python_mpc.hpp"

namespace PythonMPC_TwoWheelVehicleModelData {

using namespace PythonNumpy;
using namespace PythonControl;

constexpr std::size_t Np = 16;
constexpr std::size_t Nc = 1;

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

using type = SparseMatrix_Type<double, SparseAvailable_ekf_A>;

inline auto make(void) -> type {

    return make_SparseMatrix<SparseAvailable_ekf_A>(
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0)
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

using type = SparseMatrix_Type<double, SparseAvailable_ekf_C>;

inline auto make(void) -> type {

    return make_SparseMatrix<SparseAvailable_ekf_C>(
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0)
    );

}

} // namespace two_wheel_vehicle_model_ekf_C

namespace two_wheel_vehicle_model_ada_mpc_ekf_parameter {

class Parameter {
public:
    double m = static_cast<double>(2000);
    double l_f = static_cast<double>(1.4);
    double l_r = static_cast<double>(1.6);
    double I = static_cast<double>(4000);
    double K_f = static_cast<double>(12000.0);
    double K_r = static_cast<double>(11000.0);
};

using Parameter_Type = Parameter;

} // namespace two_wheel_vehicle_model_ada_mpc_ekf_parameter

using Parameter_Type = two_wheel_vehicle_model_ada_mpc_ekf_parameter::Parameter_Type;

namespace two_wheel_vehicle_model_ada_mpc_ekf_state_function {

using A_Type = two_wheel_vehicle_model_ekf_A::type;
using X_Type = StateSpaceState_Type<double, A_Type::COLS>;
using U_Type = StateSpaceInput_Type<double, 2>;

inline auto sympy_function(const double py, const double m, const double beta, const double r, const double l_r, const double l_f, const double K_r, const double K_f, const double px, const double theta, const double delta, const double I, const double accel, const double V) -> X_Type {

    X_Type result;

    double x0 = 0.01 * V;

    double x1 = K_f * V;

    double x2 = K_f * r;

    double x3 = V * V;

    result.template set<0, 0>(static_cast<double>(px + x0 * cos(theta)));
    result.template set<1, 0>(static_cast<double>(py + x0 * sin(theta)));
    result.template set<2, 0>(static_cast<double>(0.01 * r + theta));
    result.template set<3, 0>(static_cast<double>(r + 0.02 * (K_f * V * delta * l_f + K_r * V * beta * l_r - K_r * (l_r * l_r) * r - beta * l_f * x1 - l_f * l_f * x2) / (I * V)));
    result.template set<4, 0>(static_cast<double>(beta + 0.01 * (2 * K_f * V * delta - 2 * K_r * V * beta + 2 * K_r * l_r * r - 2 * beta * x1 - 2 * l_f * x2 - m * r * x3) / (m * x3)));
    result.template set<5, 0>(static_cast<double>(V + 0.01 * accel));

    return result;
}

inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> X_Type {

    double px = X.template get<0, 0>();

    double py = X.template get<1, 0>();

    double theta = X.template get<2, 0>();

    double r = X.template get<3, 0>();

    double beta = X.template get<4, 0>();

    double V = X.template get<5, 0>();

    double delta = U.template get<0, 0>();

    double accel = U.template get<1, 0>();

    double m = Parameters.m;

    double l_r = Parameters.l_r;

    double l_f = Parameters.l_f;

    double K_r = Parameters.K_r;

    double K_f = Parameters.K_f;

    double I = Parameters.I;

    return sympy_function(py, m, beta, r, l_r, l_f, K_r, K_f, px, theta, delta, I,
        accel, V);
}

} // namespace two_wheel_vehicle_model_ada_mpc_ekf_state_function

namespace two_wheel_vehicle_model_ada_mpc_ekf_state_function_jacobian {

using A_Type = two_wheel_vehicle_model_ekf_A::type;
using X_Type = StateSpaceState_Type<double, A_Type::COLS>;
using U_Type = StateSpaceInput_Type<double, 2>;

inline auto sympy_function(const double m, const double beta, const double r, const double l_r, const double l_f, const double K_r, const double K_f, const double theta, const double delta, const double I, const double V) -> A_Type {

    A_Type result;

    double x0 = 0.01 * sin(theta);

    double x1 = 0.01 * cos(theta);

    double x2 = K_f * (l_f * l_f);

    double x3 = K_r * (l_r * l_r);

    double x4 = 0.02 / I;

    double x5 = x4 / V;

    double x6 = K_f * l_f;

    double x7 = V * x6;

    double x8 = V * V;

    double x9 = 1 / x8;

    double x10 = 2 * x6;

    double x11 = m * x8;

    double x12 = 1 / m;

    double x13 = 0.01 * x12 * x9;

    double x14 = 2 * V;

    double x15 = K_f * x14;

    double x16 = K_r * x14;

    double x17 = 2 * beta;

    result.template set<0, 0>(static_cast<double>(1));
    result.template set<0, 1>(static_cast<double>(0));
    result.template set<0, 2>(static_cast<double>(-V * x0));
    result.template set<0, 3>(static_cast<double>(0));
    result.template set<0, 4>(static_cast<double>(0));
    result.template set<0, 5>(static_cast<double>(x1));
    result.template set<1, 0>(static_cast<double>(0));
    result.template set<1, 1>(static_cast<double>(1));
    result.template set<1, 2>(static_cast<double>(V * x1));
    result.template set<1, 3>(static_cast<double>(0));
    result.template set<1, 4>(static_cast<double>(0));
    result.template set<1, 5>(static_cast<double>(x0));
    result.template set<2, 0>(static_cast<double>(0));
    result.template set<2, 1>(static_cast<double>(0));
    result.template set<2, 2>(static_cast<double>(1));
    result.template set<2, 3>(static_cast<double>(0.01));
    result.template set<2, 4>(static_cast<double>(0));
    result.template set<2, 5>(static_cast<double>(0));
    result.template set<3, 0>(static_cast<double>(0));
    result.template set<3, 1>(static_cast<double>(0));
    result.template set<3, 2>(static_cast<double>(0));
    result.template set<3, 3>(static_cast<double>(x5 * (-x2 - x3) + 1));
    result.template set<3, 4>(static_cast<double>(x5 * (K_r * V * l_r - x7)));
    result.template set<3, 5>(static_cast<double>(-x4 * x9 * (K_f * V * delta * l_f + K_r * V * beta * l_r - beta * x7 - r * x2 - r * x3) + x5 * (K_r * beta * l_r - beta * x6 + delta * x6)));
    result.template set<4, 0>(static_cast<double>(0));
    result.template set<4, 1>(static_cast<double>(0));
    result.template set<4, 2>(static_cast<double>(0));
    result.template set<4, 3>(static_cast<double>(x13 * (2 * K_r * l_r - x10 - x11)));
    result.template set<4, 4>(static_cast<double>(x13 * (-x15 - x16) + 1));
    result.template set<4, 5>(static_cast<double>(x13 * (2 * K_f * delta - K_f * x17 - K_r * x17 - m * r * x14) - 0.02 * x12 * (2 * K_f * V * delta + 2 * K_r * l_r * r - beta * x15 - beta * x16 - r * x10 - r * x11) / (V * V * V)));
    result.template set<5, 0>(static_cast<double>(0));
    result.template set<5, 1>(static_cast<double>(0));
    result.template set<5, 2>(static_cast<double>(0));
    result.template set<5, 3>(static_cast<double>(0));
    result.template set<5, 4>(static_cast<double>(0));
    result.template set<5, 5>(static_cast<double>(1));

    return result;
}

inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> A_Type {

    double theta = X.template get<2, 0>();

    double r = X.template get<3, 0>();

    double beta = X.template get<4, 0>();

    double V = X.template get<5, 0>();

    double delta = U.template get<0, 0>();

    double m = Parameters.m;

    double l_r = Parameters.l_r;

    double l_f = Parameters.l_f;

    double K_r = Parameters.K_r;

    double K_f = Parameters.K_f;

    double I = Parameters.I;

    return sympy_function(m, beta, r, l_r, l_f, K_r, K_f, theta, delta, I, V);
}


} // namespace two_wheel_vehicle_model_ada_mpc_ekf_state_function_jacobian

namespace two_wheel_vehicle_model_ada_mpc_ekf_measurement_function {

using A_Type = two_wheel_vehicle_model_ekf_A::type;
using C_Type = two_wheel_vehicle_model_ekf_C::type;
using X_Type = StateSpaceState_Type<double, A_Type::COLS>;
using Y_Type = StateSpaceOutput_Type<double, C_Type::COLS>;

inline auto sympy_function(const double px, const double V, const double r, const double py, const double theta) -> Y_Type {

    Y_Type result;

    result.template set<0, 0>(static_cast<double>(px));
    result.template set<1, 0>(static_cast<double>(py));
    result.template set<2, 0>(static_cast<double>(theta));
    result.template set<3, 0>(static_cast<double>(r));
    result.template set<4, 0>(static_cast<double>(V));

    return result;
}

inline auto function(const X_Type X, const Parameter_Type Parameters) -> Y_Type {

    double px = X.template get<0, 0>();

    double py = X.template get<1, 0>();

    double theta = X.template get<2, 0>();

    double r = X.template get<3, 0>();

    double V = X.template get<5, 0>();

    return sympy_function(px, V, r, py, theta);
}


} // namespace two_wheel_vehicle_model_ada_mpc_ekf_measurement_function

namespace two_wheel_vehicle_model_ada_mpc_ekf_measurement_function_jacobian {

using A_Type = two_wheel_vehicle_model_ekf_A::type;
using C_Type = two_wheel_vehicle_model_ekf_C::type;
using X_Type = StateSpaceState_Type<double, A_Type::COLS>;
using Y_Type = StateSpaceOutput_Type<double, C_Type::COLS>;

inline auto sympy_function() -> C_Type {

    C_Type result;

    result.template set<0, 0>(static_cast<double>(1));
    result.template set<0, 1>(static_cast<double>(0));
    result.template set<0, 2>(static_cast<double>(0));
    result.template set<0, 3>(static_cast<double>(0));
    result.template set<0, 4>(static_cast<double>(0));
    result.template set<0, 5>(static_cast<double>(0));
    result.template set<1, 0>(static_cast<double>(0));
    result.template set<1, 1>(static_cast<double>(1));
    result.template set<1, 2>(static_cast<double>(0));
    result.template set<1, 3>(static_cast<double>(0));
    result.template set<1, 4>(static_cast<double>(0));
    result.template set<1, 5>(static_cast<double>(0));
    result.template set<2, 0>(static_cast<double>(0));
    result.template set<2, 1>(static_cast<double>(0));
    result.template set<2, 2>(static_cast<double>(1));
    result.template set<2, 3>(static_cast<double>(0));
    result.template set<2, 4>(static_cast<double>(0));
    result.template set<2, 5>(static_cast<double>(0));
    result.template set<3, 0>(static_cast<double>(0));
    result.template set<3, 1>(static_cast<double>(0));
    result.template set<3, 2>(static_cast<double>(0));
    result.template set<3, 3>(static_cast<double>(1));
    result.template set<3, 4>(static_cast<double>(0));
    result.template set<3, 5>(static_cast<double>(0));
    result.template set<4, 0>(static_cast<double>(0));
    result.template set<4, 1>(static_cast<double>(0));
    result.template set<4, 2>(static_cast<double>(0));
    result.template set<4, 3>(static_cast<double>(0));
    result.template set<4, 4>(static_cast<double>(0));
    result.template set<4, 5>(static_cast<double>(1));

    return result;
}

inline auto function(const X_Type X, const Parameter_Type Parameters) -> C_Type {

    return sympy_function();
}


} // namespace two_wheel_vehicle_model_ada_mpc_ekf_measurement_function_jacobian

namespace two_wheel_vehicle_model_ada_mpc_ekf {

using namespace PythonNumpy;
using namespace PythonControl;

constexpr std::size_t NUMBER_OF_DELAY = 0;

using A_Type = two_wheel_vehicle_model_ekf_A::type;

using C_Type = two_wheel_vehicle_model_ekf_C::type;

constexpr std::size_t STATE_SIZE = A_Type::COLS;
constexpr std::size_t INPUT_SIZE = 2;
constexpr std::size_t OUTPUT_SIZE = C_Type::COLS;

using X_Type = StateSpaceState_Type<double, STATE_SIZE>;
using U_Type = StateSpaceInput_Type<double, INPUT_SIZE>;
using Y_Type = StateSpaceOutput_Type<double, OUTPUT_SIZE>;

using Q_Type = KalmanFilter_Q_Type<double, STATE_SIZE>;

using R_Type = KalmanFilter_R_Type<double, OUTPUT_SIZE>;

using Parameter_Type = two_wheel_vehicle_model_ada_mpc_ekf_parameter::Parameter_Type;

using type = ExtendedKalmanFilter_Type<
    A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>;

inline auto make() -> type {

    auto A = two_wheel_vehicle_model_ekf_A::make();

    auto C = two_wheel_vehicle_model_ekf_C::make();

    auto Q = make_KalmanFilter_Q<STATE_SIZE>(
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0)
    );

    auto R = make_KalmanFilter_R<OUTPUT_SIZE>(
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0)
    );

    Parameter_Type parameters;

    StateFunction_Object<X_Type, U_Type, Parameter_Type> state_function_object =
        [](const X_Type& X, const U_Type& U, const Parameter_Type& Parameters) {
        return two_wheel_vehicle_model_ada_mpc_ekf_state_function::function(X, U, Parameters);

        };

    StateFunctionJacobian_Object<A_Type, X_Type, U_Type, Parameter_Type> state_function_jacobian_object =
        [](const X_Type& X, const U_Type& U, const Parameter_Type& Parameters) {
        return two_wheel_vehicle_model_ada_mpc_ekf_state_function_jacobian::function(X, U, Parameters);

        };

    MeasurementFunction_Object<Y_Type, X_Type, Parameter_Type> measurement_function_object =
        [](const X_Type& X, const Parameter_Type& Parameters) {
        return two_wheel_vehicle_model_ada_mpc_ekf_measurement_function::function(X, Parameters);

        };

    MeasurementFunctionJacobian_Object<C_Type, X_Type, Parameter_Type> measurement_function_jacobian_object =
        [](const X_Type& X, const Parameter_Type& Parameters) {
        return two_wheel_vehicle_model_ada_mpc_ekf_measurement_function_jacobian::function(X, Parameters);

        };

    return ExtendedKalmanFilter_Type<
        A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>(
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

using type = SparseMatrix_Type<double, SparseAvailable_ada_mpc_B>;

inline auto make(void) -> type {

    return make_SparseMatrix<SparseAvailable_ada_mpc_B>(
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0)
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

using type = SparseMatrix_Type<double, SparseAvailable_ada_mpc_F>;

inline auto make(void) -> type {

    return make_SparseMatrix<SparseAvailable_ada_mpc_F>(
        static_cast<double>(1.0),
        static_cast<double>(0.01),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(0.1),
        static_cast<double>(1.0),
        static_cast<double>(0.05),
        static_cast<double>(0.0005),
        static_cast<double>(1.0),
        static_cast<double>(0.0097416),
        static_cast<double>(4.000000000000001e-05),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(1.0),
        static_cast<double>(2.0),
        static_cast<double>(0.03),
        static_cast<double>(1.0),
        static_cast<double>(2.0),
        static_cast<double>(0.30000000000000004),
        static_cast<double>(0.001),
        static_cast<double>(1.0),
        static_cast<double>(0.1),
        static_cast<double>(0.00148708),
        static_cast<double>(2.0000000000000003e-06),
        static_cast<double>(1.0),
        static_cast<double>(0.019231080256),
        static_cast<double>(0.00011804640000000003),
        static_cast<double>(1.0),
        static_cast<double>(2.0),
        static_cast<double>(1.0),
        static_cast<double>(3.0),
        static_cast<double>(0.060000000000000005),
        static_cast<double>(1.0),
        static_cast<double>(3.0),
        static_cast<double>(0.6),
        static_cast<double>(0.003974160000000001),
        static_cast<double>(4.000000000000001e-06),
        static_cast<double>(1.0),
        static_cast<double>(0.15000000000000002),
        static_cast<double>(0.0029486340128),
        static_cast<double>(7.902320000000002e-06),
        static_cast<double>(1.0),
        static_cast<double>(0.028474578121896958),
        static_cast<double>(0.00023225565382400005),
        static_cast<double>(1.0),
        static_cast<double>(3.0),
        static_cast<double>(1.0),
        static_cast<double>(4.0),
        static_cast<double>(0.09999999999999999),
        static_cast<double>(1.0),
        static_cast<double>(4.0),
        static_cast<double>(1.0),
        static_cast<double>(0.009871428025600001),
        static_cast<double>(1.9804640000000006e-05),
        static_cast<double>(1.0),
        static_cast<double>(0.2),
        static_cast<double>(0.004872362918894849),
        static_cast<double>(1.9515102691200004e-05),
        static_cast<double>(1.0),
        static_cast<double>(0.03747809104714121),
        static_cast<double>(0.00038081208627363587),
        static_cast<double>(1.0),
        static_cast<double>(4.0),
        static_cast<double>(1.0),
        static_cast<double>(5.0),
        static_cast<double>(0.15),
        static_cast<double>(1.0),
        static_cast<double>(5.0),
        static_cast<double>(1.5),
        static_cast<double>(0.019616153863389697),
        static_cast<double>(5.883484538240002e-05),
        static_cast<double>(1.0),
        static_cast<double>(0.25),
        static_cast<double>(0.00724626747125191),
        static_cast<double>(3.85557070048818e-05),
        static_cast<double>(1.0),
        static_cast<double>(0.04624747951858725),
        static_cast<double>(0.0005619657724779072),
        static_cast<double>(1.0),
        static_cast<double>(5.0),
        static_cast<double>(1.0),
        static_cast<double>(6.0),
        static_cast<double>(0.21000000000000002),
        static_cast<double>(1.0),
        static_cast<double>(6.0),
        static_cast<double>(2.1),
        static_cast<double>(0.03410868880589352),
        static_cast<double>(0.00013594625939216363),
        static_cast<double>(1.0),
        static_cast<double>(0.3),
        static_cast<double>(0.010058641447181273),
        static_cast<double>(6.665399562877716e-05),
        static_cast<double>(1.0),
        static_cast<double>(0.05478846994736397),
        static_cast<double>(0.0007740304777852644),
        static_cast<double>(1.0),
        static_cast<double>(6.0),
        static_cast<double>(1.0),
        static_cast<double>(7.0),
        static_cast<double>(0.28),
        static_cast<double>(1.0),
        static_cast<double>(7.0),
        static_cast<double>(2.8000000000000003),
        static_cast<double>(0.05422597170025606),
        static_cast<double>(0.00026925425064971796),
        static_cast<double>(1.0),
        static_cast<double>(0.35),
        static_cast<double>(0.013298064944549472),
        static_cast<double>(0.00010535551951804039),
        static_cast<double>(1.0),
        static_cast<double>(0.06310665750158446),
        static_cast<double>(0.0010153816565856592),
        static_cast<double>(1.0),
        static_cast<double>(7.0),
        static_cast<double>(1.0),
        static_cast<double>(8.0),
        static_cast<double>(0.36000000000000004),
        static_cast<double>(1.0),
        static_cast<double>(8.0),
        static_cast<double>(3.6000000000000005),
        static_cast<double>(0.080822101589355),
        static_cast<double>(0.00047996528968579874),
        static_cast<double>(1.0),
        static_cast<double>(0.39999999999999997),
        static_cast<double>(0.016953397819628693),
        static_cast<double>(0.00015612460234732336),
        static_cast<double>(1.0),
        static_cast<double>(0.07120750888571019),
        static_cast<double>(0.001284454508490527),
        static_cast<double>(1.0),
        static_cast<double>(8.0),
        static_cast<double>(1.0),
        static_cast<double>(9.0),
        static_cast<double>(0.45000000000000007),
        static_cast<double>(1.0),
        static_cast<double>(9.0),
        static_cast<double>(4.5),
        static_cast<double>(0.11472889722861239),
        static_cast<double>(0.0007922144943804454),
        static_cast<double>(1.0),
        static_cast<double>(0.44999999999999996),
        static_cast<double>(0.021013773263914202),
        static_cast<double>(0.0002203473277718497),
        static_cast<double>(1.0),
        static_cast<double>(0.07909636506737922),
        static_cast<double>(0.0015797420903380858),
        static_cast<double>(1.0),
        static_cast<double>(9.0),
        static_cast<double>(1.0),
        static_cast<double>(10.0),
        static_cast<double>(0.55),
        static_cast<double>(1.0),
        static_cast<double>(10.0),
        static_cast<double>(5.5),
        static_cast<double>(0.1567564437564408),
        static_cast<double>(0.0012329091499241448),
        static_cast<double>(1.0),
        static_cast<double>(0.49999999999999994),
        static_cast<double>(0.025468591517283163),
        static_cast<double>(0.000299334432288754),
        static_cast<double>(1.0),
        static_cast<double>(0.08677844395250198),
        static_cast<double>(0.001899793482529827),
        static_cast<double>(1.0),
        static_cast<double>(10.0),
        static_cast<double>(1.0),
        static_cast<double>(11.0),
        static_cast<double>(0.66),
        static_cast<double>(1.0),
        static_cast<double>(11.0),
        static_cast<double>(6.6),
        static_cast<double>(0.2076936267910071),
        static_cast<double>(0.0018315780145016529),
        static_cast<double>(1.0),
        static_cast<double>(0.5499999999999999),
        static_cast<double>(0.030307513714908262),
        static_cast<double>(0.0003943241064152453),
        static_cast<double>(1.0),
        static_cast<double>(0.09425884300942264),
        static_cast<double>(0.002243212008241649),
        static_cast<double>(1.0),
        static_cast<double>(11.0),
        static_cast<double>(1.0),
        static_cast<double>(12.0),
        static_cast<double>(0.78),
        static_cast<double>(1.0),
        static_cast<double>(12.0),
        static_cast<double>(7.799999999999999),
        static_cast<double>(0.2683086542208236),
        static_cast<double>(0.0026202262273321434),
        static_cast<double>(1.0),
        static_cast<double>(0.6),
        static_cast<double>(0.03552045586537939),
        static_cast<double>(0.0005064847068273278),
        static_cast<double>(1.0),
        static_cast<double>(0.10154254184293741),
        static_cast<double>(0.002608653504089782),
        static_cast<double>(1.0),
        static_cast<double>(12.0),
        static_cast<double>(1.0),
        static_cast<double>(13.0),
        static_cast<double>(0.91),
        static_cast<double>(1.0),
        static_cast<double>(13.0),
        static_cast<double>(9.1),
        static_cast<double>(0.3393495659515824),
        static_cast<double>(0.003633195640986799),
        static_cast<double>(1.0),
        static_cast<double>(0.65),
        static_cast<double>(0.04109758295752627),
        static_cast<double>(0.0006369173820318169),
        static_cast<double>(1.0),
        static_cast<double>(0.10863440471895534),
        static_cast<double>(0.002994824640867467),
        static_cast<double>(1.0),
        static_cast<double>(13.0),
        static_cast<double>(1.0),
        static_cast<double>(14.0),
        static_cast<double>(1.05),
        static_cast<double>(1.0),
        static_cast<double>(14.0),
        static_cast<double>(10.5),
        static_cast<double>(0.42154473186663494),
        static_cast<double>(0.004907030405050432),
        static_cast<double>(1.0),
        static_cast<double>(0.7000000000000001),
        static_cast<double>(0.04702930319347404),
        static_cast<double>(0.0007866586140751902),
        static_cast<double>(1.0),
        static_cast<double>(0.11553918304058014),
        static_cast<double>(0.0034004812930033368),
        static_cast<double>(1.0),
        static_cast<double>(14.0),
        static_cast<double>(1.0),
        static_cast<double>(15.0),
        static_cast<double>(1.2),
        static_cast<double>(1.0),
        static_cast<double>(15.0),
        static_cast<double>(12.0),
        static_cast<double>(0.515603338253583),
        static_cast<double>(0.006480347633200812),
        static_cast<double>(1.0),
        static_cast<double>(0.7500000000000001),
        static_cast<double>(0.05330626234550304),
        static_cast<double>(0.000956682678725357),
        static_cast<double>(1.0),
        static_cast<double>(0.12226151777638497),
        static_cast<double>(0.003824426955426581),
        static_cast<double>(1.0),
        static_cast<double>(15.0),
        static_cast<double>(1.0),
        static_cast<double>(16.0),
        static_cast<double>(1.3599999999999999),
        static_cast<double>(1.0),
        static_cast<double>(16.0),
        static_cast<double>(13.6),
        static_cast<double>(0.622215862944589),
        static_cast<double>(0.008393712990651525),
        static_cast<double>(1.0),
        static_cast<double>(0.8000000000000002),
        static_cast<double>(0.05991933823432229),
        static_cast<double>(0.001147904026496686),
        static_cast<double>(1.0),
        static_cast<double>(0.12880594184164534),
        static_cast<double>(0.00426551120655731),
        static_cast<double>(1.0),
        static_cast<double>(16.0),
        static_cast<double>(1.0)
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

using type = SparseMatrix_Type<double, SparseAvailable_ada_mpc_Phi>;

inline auto make(void) -> type {

    return make_SparseMatrix<SparseAvailable_ada_mpc_Phi>(
        static_cast<double>(0.0008399999999999999),
        static_cast<double>(0.01),
        static_cast<double>(0.0001),
        static_cast<double>(4.2000000000000004e-05),
        static_cast<double>(0.0016587744),
        static_cast<double>(0.02),
        static_cast<double>(0.0003),
        static_cast<double>(8.400000000000001e-05),
        static_cast<double>(0.00012493872),
        static_cast<double>(0.002456827298304),
        static_cast<double>(0.03),
        static_cast<double>(0.0006000000000000001),
        static_cast<double>(0.0003338774400000001),
        static_cast<double>(0.00024778008491520005),
        static_cast<double>(0.0032346516300852325),
        static_cast<double>(0.04),
        static_cast<double>(0.001),
        static_cast<double>(0.0008294376098304002),
        static_cast<double>(0.0004095126664194618),
        static_cast<double>(0.0039927293929951455),
        static_cast<double>(0.05),
        static_cast<double>(0.0015),
        static_cast<double>(0.0016484629426693235),
        static_cast<double>(0.0006091491360692191),
        static_cast<double>(0.004731531868831064),
        static_cast<double>(0.060000000000000005),
        static_cast<double>(0.0021000000000000003),
        static_cast<double>(0.0028667612148077616),
        static_cast<double>(0.0008457257295107723),
        static_cast<double>(0.005451519841311997),
        static_cast<double>(0.06999999999999999),
        static_cast<double>(0.0028000000000000004),
        static_cast<double>(0.004558212673829306),
        static_cast<double>(0.0011183017215763722),
        static_cast<double>(0.006153143810012123),
        static_cast<double>(0.08),
        static_cast<double>(0.0036000000000000003),
        static_cast<double>(0.00679481611698205),
        static_cast<double>(0.0014259589120769781),
        static_cast<double>(0.0068368442005015415),
        static_cast<double>(0.09),
        static_cast<double>(0.0045000000000000005),
        static_cast<double>(0.009646733941136007),
        static_cast<double>(0.0017678011221020553),
        static_cast<double>(0.007503051570743911),
        static_cast<double>(0.09999999999999999),
        static_cast<double>(0.0055000000000000005),
        static_cast<double>(0.013182336185340118),
        static_cast<double>(0.002142953700639251),
        static_cast<double>(0.008152186813800524),
        static_cast<double>(0.11),
        static_cast<double>(0.006600000000000001),
        static_cast<double>(0.017468243586618616),
        static_cast<double>(0.002550563041329277),
        static_cast<double>(0.008784661356890404),
        static_cast<double>(0.12),
        static_cast<double>(0.0078000000000000005),
        static_cast<double>(0.02256936966927717),
        static_cast<double>(0.002989796109173797),
        static_cast<double>(0.009400877356855821),
        static_cast<double>(0.13),
        static_cast<double>(0.0091),
        static_cast<double>(0.028548961887624767),
        static_cast<double>(0.0034598399770165884),
        static_cast<double>(0.01000122789208266),
        static_cast<double>(0.14),
        static_cast<double>(0.0105),
        static_cast<double>(0.03546864184165794),
        static_cast<double>(0.003959901371620721),
        static_cast<double>(0.010586097150924774),
        static_cast<double>(0.15000000000000002),
        static_cast<double>(0.012),
        static_cast<double>(0.043388444584899384),
        static_cast<double>(0.00448920622916696),
        static_cast<double>(0.011155860616681458),
        static_cast<double>(0.16)
    );

}

} // namespace two_wheel_vehicle_model_ada_mpc_Phi



} // namespace PythonMPC_TwoWheelVehicleModelData

#endif // __TEST_MPC_TWO_WHEEL_VEHICLE_MODEL_DATA_HPP__
