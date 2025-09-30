#include <type_traits>
#include <iostream>
#include <cmath>

#include "python_mpc.hpp"

#include "test_mpc_servo_motor_data.hpp"
#include "test_mpc_two_wheel_vehicle_model_data.hpp"
#include "test_Adaptive_MPC_Phi_F_Updater_Function.hpp"
#include "test_mpc_kinematic_bicycle_model_data.hpp"

#include "MCAP_tester.hpp"

using namespace Tester;


template <typename T>
void check_MPC_PredictionMatrices(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonMPC;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-5);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;


    /* 定義 */
    constexpr std::size_t NP = 10;
    constexpr std::size_t NC = 2;
    constexpr std::size_t Number_Of_State = 3;
    constexpr std::size_t Number_Of_Input = 1;
    constexpr std::size_t Number_Of_Output = 1;

    using F_Type = DenseMatrix_Type<T, (NP* Number_Of_Output), Number_Of_State>;

    F_Type F = make_DenseMatrix<(NP * Number_Of_Output), Number_Of_State>(
        static_cast<T>(0.7), static_cast<T>(0.2), static_cast<T>(1.0),
        static_cast<T>(1.13), static_cast<T>(0.5), static_cast<T>(1.0),
        static_cast<T>(1.341), static_cast<T>(0.826), static_cast<T>(1.0),
        static_cast<T>(1.3909), static_cast<T>(1.129), static_cast<T>(1.0),
        static_cast<T>(1.33493), static_cast<T>(1.38138), static_cast<T>(1.0),
        static_cast<T>(1.220037), static_cast<T>(1.57209), static_cast<T>(1.0),
        static_cast<T>(1.0823989), static_cast<T>(1.7016794), static_cast<T>(1.0),
        static_cast<T>(0.94717541), static_cast<T>(1.7778233), static_cast<T>(1.0),
        static_cast<T>(0.8296758), static_cast<T>(1.81169372), static_cast<T>(1.0),
        static_cast<T>(0.73726494), static_cast<T>(1.81529014), static_cast<T>(1.0)
    );

    using Phi_SparseAvailable = SparseAvailable<
        ColumnAvailable<true, false>,
        ColumnAvailable<true, true>,
        ColumnAvailable<true, true>,
        ColumnAvailable<true, true>, 
        ColumnAvailable<true, true>, 
        ColumnAvailable<true, true>, 
        ColumnAvailable<true, true>, 
        ColumnAvailable<true, true>, 
        ColumnAvailable<true, true>, 
        ColumnAvailable<true, true>>;

    using Phi_Type = SparseMatrix_Type<T, Phi_SparseAvailable>;

    Phi_Type Phi = make_SparseMatrix<Phi_SparseAvailable>(
        static_cast<T>(0.1),
        static_cast<T>(0.21), static_cast<T>(0.1),
        static_cast<T>(0.313), static_cast<T>(0.21),
        static_cast<T>(0.3993), static_cast<T>(0.313),
        static_cast<T>(0.46489), static_cast<T>(0.3993),
        static_cast<T>(0.509769), static_cast<T>(0.46489),
        static_cast<T>(0.5364217), static_cast<T>(0.509769),
        static_cast<T>(0.54857577), static_cast<T>(0.5364217),
        static_cast<T>(0.5502822), static_cast<T>(0.54857577),
        static_cast<T>(0.54530632), static_cast<T>(0.5502822)
    );

    MPC_PredictionMatrices<F_Type, Phi_Type, 
    NP, NC, Number_Of_Input, Number_Of_State, Number_Of_Output> prediction_matrices(F, Phi);

    MPC_PredictionMatrices_Type<F_Type, Phi_Type,
        NP, NC, Number_Of_Input, Number_Of_State, Number_Of_Output>
        prediction_matrices_copy = prediction_matrices;
    MPC_PredictionMatrices_Type<F_Type, Phi_Type,
        NP, NC, Number_Of_Input, Number_Of_State, Number_Of_Output>
        prediction_matrices_move = make_MPC_PredictionMatrices<F_Type, Phi_Type,
        NP, NC, Number_Of_Input, Number_Of_State, Number_Of_Output>();
        
    prediction_matrices_move = std::move(prediction_matrices_copy);
    prediction_matrices = prediction_matrices_move;


    tester.expect_near(prediction_matrices.F.matrix.data,
        F.matrix.data, NEAR_LIMIT_STRICT,
        "check F matrix.");
    tester.expect_near(prediction_matrices.Phi.matrix.values,
        Phi.matrix.values, NEAR_LIMIT_STRICT,
        "check Phi matrix.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_MPC_ReferenceTrajectory(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonMPC;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-5);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 定義 */
    constexpr std::size_t NP = 3;

    auto Fx = make_DenseMatrix<9, 1>(
        static_cast<T>(0.5), static_cast<T>(0.1), static_cast<T>(0.2),
        static_cast<T>(0.3), static_cast<T>(0.4), static_cast<T>(0.5),
        static_cast<T>(0.6), static_cast<T>(0.7), static_cast<T>(0.9));

    auto ref_vector = make_DenseMatrix<3, 1>(
        static_cast<T>(0.1), static_cast<T>(0.2), static_cast<T>(0.3));
    auto ref_trajectory = make_DenseMatrix<3, 3>(
        static_cast<T>(0.1), static_cast<T>(0.2), static_cast<T>(0.3),
        static_cast<T>(0.4), static_cast<T>(0.5), static_cast<T>(0.6),
        static_cast<T>(0.7), static_cast<T>(0.8), static_cast<T>(0.9));


    MPC_ReferenceTrajectory<decltype(ref_vector), NP> reference_trajectory(ref_vector);
    MPC_ReferenceTrajectory<decltype(ref_vector), NP> reference_trajectory_copy(reference_trajectory);
    MPC_ReferenceTrajectory_Type<decltype(ref_vector), NP> reference_trajectory_move =
        make_MPC_ReferenceTrajectory<decltype(ref_vector), NP>();
    reference_trajectory_move = std::move(reference_trajectory_copy);
    reference_trajectory = reference_trajectory_move;

    auto dif = reference_trajectory.calculate_dif(Fx);

    auto dif_answer = make_DenseMatrix<9, 1>(
        static_cast<T>(-0.4), static_cast<T>(0.1), static_cast<T>(0.1),
        static_cast<T>(-0.2), static_cast<T>(-0.2), static_cast<T>(-0.2),
        static_cast<T>(-0.5), static_cast<T>(-0.5), static_cast<T>(-0.6));

    tester.expect_near(dif.matrix.data, dif_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check reference trajectory difference.");


    MPC_ReferenceTrajectory<decltype(ref_trajectory), NP> reference_trajectory_2(ref_trajectory);
    MPC_ReferenceTrajectory<decltype(ref_trajectory), NP> reference_trajectory_2_copy(reference_trajectory_2);
    MPC_ReferenceTrajectory_Type<decltype(ref_trajectory), NP> reference_trajectory_2_move =
        make_MPC_ReferenceTrajectory<decltype(ref_trajectory), NP>();
    reference_trajectory_2_move = std::move(reference_trajectory_2_copy);
    reference_trajectory_2 = reference_trajectory_2_move;

    auto dif_2 = reference_trajectory_2.calculate_dif(Fx);

    auto dif_2_answer = make_DenseMatrix<9, 1>(
        static_cast<T>(-0.4), static_cast<T>(0.3), static_cast<T>(0.5),
        static_cast<T>(-0.1), static_cast<T>(0.1), static_cast<T>(0.3),
        static_cast<T>(-0.3), static_cast<T>(-0.1), static_cast<T>(0.0));

    tester.expect_near(dif_2.matrix.data, dif_2_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check reference trajectory difference folloing mode.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_LTI_MPC_NoConstraints(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonMPC;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-5);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 定義 */
    constexpr std::size_t NP = PythonMPC_ServoMotorData::NP;
    constexpr std::size_t NC = PythonMPC_ServoMotorData::NC;

    constexpr std::size_t INPUT_SIZE = PythonMPC_ServoMotorData::INPUT_SIZE;
    constexpr std::size_t STATE_SIZE = PythonMPC_ServoMotorData::STATE_SIZE;
    constexpr std::size_t OUTPUT_SIZE = PythonMPC_ServoMotorData::OUTPUT_SIZE;

    constexpr std::size_t AUGMENTED_STATE_SIZE = PythonMPC_ServoMotorData::AUGMENTED_STATE_SIZE;

    auto A = make_DenseMatrix<4, 4>(
        static_cast<T>(1.0), static_cast<T>(0.05), static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(-2.56038538), static_cast<T>(0.95000025), static_cast<T>(0.12801927), static_cast<T>(0.0),
        static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(1.0), static_cast<T>(0.05),
        static_cast<T>(6.40099503), static_cast<T>(0.0), static_cast<T>(-0.32004975), static_cast<T>(0.49)
    );

    auto B = make_DenseMatrix<4, 1>(
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.05)
    );

    auto C = make_DenseMatrix<2, 4>(
        static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(1280.19900634), static_cast<T>(0.0), static_cast<T>(-64.00995032), static_cast<T>(0.0)
    );

    auto D = make_DenseMatrixZeros<T, OUTPUT_SIZE, INPUT_SIZE>();

    T dt = static_cast<T>(0.05);

    auto sys = make_DiscreteStateSpace(A, B, C, D, dt);

    auto Q = make_KalmanFilter_Q<STATE_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

    auto R = make_KalmanFilter_R<OUTPUT_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

    auto kalman_filter = make_LinearKalmanFilter(sys, Q, R);

    kalman_filter.G = make_DenseMatrix<STATE_SIZE, OUTPUT_SIZE>(
        static_cast<T>(0.04893929), static_cast<T>(0.00074138),
        static_cast<T>(0.00827874), static_cast<T>(0.00030475),
        static_cast<T>(9.78774203e-01), static_cast<T>(-7.95038792e-04),
        static_cast<T>(2.86510380e-03), static_cast<T>(-3.43928205e-06)
    );

    auto F = PythonMPC_ServoMotorData::get_F<T>();

    auto Phi = PythonMPC_ServoMotorData::get_Phi<T>();

    MPC_PredictionMatrices<decltype(F), decltype(Phi),
        NP, NC, INPUT_SIZE, AUGMENTED_STATE_SIZE, OUTPUT_SIZE> prediction_matrices(F, Phi);

    auto ref = make_DenseMatrix<OUTPUT_SIZE, 1>(
        static_cast<T>(1.0), static_cast<T>(0.0));

    MPC_ReferenceTrajectory_Type<decltype(ref), NP> reference_trajectory(ref);

    auto solver_factor = PythonMPC_ServoMotorData::get_solver_factor<T>();

    LTI_MPC_NoConstraints<decltype(kalman_filter), decltype(prediction_matrices),
        decltype(reference_trajectory)> lti_mpc(
            kalman_filter, prediction_matrices, reference_trajectory, solver_factor);


    LTI_MPC_NoConstraints_Type<decltype(kalman_filter), decltype(prediction_matrices),
        decltype(reference_trajectory)> lti_mpc_copy(lti_mpc);
    LTI_MPC_NoConstraints_Type<decltype(kalman_filter), decltype(prediction_matrices),
        decltype(reference_trajectory)> lti_mpc_move = lti_mpc_copy;
    lti_mpc = std::move(lti_mpc_move);

    auto Y = make_StateSpaceOutput<OUTPUT_SIZE>(static_cast<T>(0.0), static_cast<T>(0.0));

    auto U = lti_mpc.update(ref, Y);

    auto U_answer = make_StateSpaceInput<INPUT_SIZE>(static_cast<T>(23.53521535));

    tester.expect_near(U.matrix.data, U_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LTI MPC No Constraints, update.");


    /* 遅れありの定義 */
    constexpr std::size_t NUMBER_OF_DELAY = 1;

    auto sys_delay = make_DiscreteStateSpace<NUMBER_OF_DELAY>(A, B, C, D, dt);
    auto kalman_filter_delay = make_LinearKalmanFilter(sys_delay, Q, R);

    LTI_MPC_NoConstraints<decltype(kalman_filter_delay), decltype(prediction_matrices),
        decltype(reference_trajectory)> lti_mpc_delay(
            kalman_filter_delay, prediction_matrices, reference_trajectory, solver_factor);

    auto Y_delay = make_StateSpaceOutput<OUTPUT_SIZE>(static_cast<T>(0.0), static_cast<T>(0.0));

    auto U_delay = lti_mpc_delay.update(ref, Y_delay);
    U_delay = lti_mpc_delay.update(ref, Y_delay);
    U_delay = lti_mpc_delay.update(ref, Y_delay);

    Y_delay(1, 0) = static_cast<T>(-3.76621991);

    U_delay = lti_mpc_delay.update(ref, Y_delay);

    auto U_delay_answer = make_StateSpaceInput<INPUT_SIZE>(static_cast<T>(67.73585786));

    tester.expect_near(U_delay.matrix.data, U_delay_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LTI MPC No Constraints, delay, update.");


    tester.throw_error_if_test_failed();
}


template <typename T>
void check_DU_U_Y_Limits(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonMPC;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-5);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 定義 */
    using Delta_U_Min_SparseAvailable = SparseAvailable<
        ColumnAvailable<true>,
        ColumnAvailable<true>,
        ColumnAvailable<false>
    >;
    auto delta_U_Min = make_SparseMatrix<Delta_U_Min_SparseAvailable>(
        static_cast<T>(-0.1),
        static_cast<T>(0.2)
    );

    using Delta_U_Max_SparseAvailable = SparseAvailable<
        ColumnAvailable<true>,
        ColumnAvailable<false>,
        ColumnAvailable<false>
    >;
    auto delta_U_Max = make_SparseMatrix<Delta_U_Max_SparseAvailable>(
        static_cast<T>(-0.3)
    );

    using U_Min_SparseAvailable = SparseAvailable<
        ColumnAvailable<false>,
        ColumnAvailable<true>,
        ColumnAvailable<false>
    >;
    auto U_Min = make_SparseMatrix<U_Min_SparseAvailable>(
        static_cast<T>(0.4)
    );

    using U_Max_SparseAvailable = SparseAvailable<
        ColumnAvailable<false>,
        ColumnAvailable<false>,
        ColumnAvailable<true>
    >;
    auto U_Max = make_SparseMatrix<U_Max_SparseAvailable>(
        static_cast<T>(-0.5)
    );

    using Y_Min_SparseAvailable = SparseAvailable<
        ColumnAvailable<false>,
        ColumnAvailable<true>
    >;
    auto Y_Min = make_SparseMatrix<Y_Min_SparseAvailable>(
        static_cast<T>(0.6)
    );

    auto Y_Max = make_SparseMatrixEmpty<T, 2, 1>();

    DU_U_Y_Limits_Type<decltype(delta_U_Min), decltype(delta_U_Max),
        decltype(U_Min), decltype(U_Max),
        decltype(Y_Min), decltype(Y_Max)> limits = make_DU_U_Y_Limits(
            delta_U_Min, delta_U_Max, U_Min, U_Max, Y_Min, Y_Max);

    decltype(limits) limits_copy(limits);
    decltype(limits) limits_move = std::move(limits_copy);
    limits = limits_move;

    std::size_t number_of_delta_U_constraints = decltype(limits)::NUMBER_OF_DELTA_U_CONSTRAINTS;

    tester.expect_near(static_cast<T>(number_of_delta_U_constraints), static_cast<T>(3),
        NEAR_LIMIT_STRICT,
        "check number of delta U constraints.");

    std::size_t number_of_U_constraints = decltype(limits)::NUMBER_OF_U_CONSTRAINTS;

    tester.expect_near(static_cast<T>(number_of_U_constraints), static_cast<T>(2),
        NEAR_LIMIT_STRICT,
        "check number of U constraints.");

    std::size_t number_of_Y_constraints = decltype(limits)::NUMBER_OF_Y_CONSTRAINTS;

    tester.expect_near(static_cast<T>(number_of_Y_constraints), static_cast<T>(1),
        NEAR_LIMIT_STRICT,
        "check number of Y constraints.");

    std::size_t number_of_all_constraints = decltype(limits)::NUMBER_OF_ALL_CONSTRAINTS;

    tester.expect_near(static_cast<T>(number_of_all_constraints), static_cast<T>(6),
        NEAR_LIMIT_STRICT,
        "check number of all constraints.");



    tester.throw_error_if_test_failed();
}


template <typename T>
void check_LMPC_QP_Solver(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonMPC;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-5);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 定義 */
    constexpr std::size_t NP = PythonMPC_ServoMotorData::NP;
    constexpr std::size_t NC = PythonMPC_ServoMotorData::NC;

    constexpr std::size_t INPUT_SIZE = PythonMPC_ServoMotorData::INPUT_SIZE;
    //constexpr std::size_t STATE_SIZE = PythonMPC_ServoMotorData::STATE_SIZE;
    constexpr std::size_t OUTPUT_SIZE = PythonMPC_ServoMotorData::OUTPUT_SIZE;

    constexpr std::size_t AUGMENTED_STATE_SIZE = PythonMPC_ServoMotorData::AUGMENTED_STATE_SIZE;

    constexpr std::size_t NUMBER_OF_VARIABLES = INPUT_SIZE * NC;

    using U_Type = DenseMatrix_Type<T, INPUT_SIZE, 1>;

    U_Type U;

    using X_augmented_Type = DenseMatrix_Type<T, AUGMENTED_STATE_SIZE, 1>;

    X_augmented_Type X_augmented;

    auto F = PythonMPC_ServoMotorData::get_F<T>();

    auto Phi = PythonMPC_ServoMotorData::get_Phi<T>();

    using F_Type = decltype(F);

    using Phi_Type = decltype(Phi);

    using Weight_U_Nc_Type = DiagMatrix_Type<T, INPUT_SIZE * NC>;

    Weight_U_Nc_Type weight_U_NC = make_DiagMatrixIdentity<T, INPUT_SIZE * NC>();

    auto delta_U_min = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(-101));
    auto delta_U_max = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(102));

    auto U_min = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(-181));
    auto U_max = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(182));

    auto Y_min = make_DenseMatrix<OUTPUT_SIZE, 1>(
        static_cast<T>(-11), static_cast<T>(-102));
    auto Y_max = make_DenseMatrix<OUTPUT_SIZE, 1>(
        static_cast<T>(13), static_cast<T>(104));

    using Delta_U_min_Type = decltype(delta_U_min);
    using Delta_U_max_Type = decltype(delta_U_max);
    using U_min_Type = decltype(U_min);
    using U_max_Type = decltype(U_max);
    using Y_min_Type = decltype(Y_min);
    using Y_max_Type = decltype(Y_max);

    LMPC_QP_Solver_Type<NUMBER_OF_VARIABLES, OUTPUT_SIZE,
        U_Type, X_augmented_Type, Phi_Type, F_Type, Weight_U_Nc_Type,
        Delta_U_min_Type, Delta_U_max_Type,
        U_min_Type, U_max_Type,
        Y_min_Type, Y_max_Type> lti_mpc_qp_solver =
        make_LMPC_QP_Solver<NUMBER_OF_VARIABLES, OUTPUT_SIZE>(
            U, X_augmented, Phi, F, weight_U_NC,
            delta_U_min, delta_U_max, U_min, U_max, Y_min, Y_max);


    auto M_answer = make_DenseMatrix<
        decltype(lti_mpc_qp_solver)::NUMBER_OF_ALL_CONSTRAINTS,
        NUMBER_OF_VARIABLES>(
            static_cast<T>(-1.0), static_cast<T>(0.0),
            static_cast<T>(1.0), static_cast<T>(0.0),
            static_cast<T>(-1.0), static_cast<T>(0.0),
            static_cast<T>(1.0), static_cast<T>(0.0),
            static_cast<T>(0.0), static_cast<T>(0.0),
            static_cast<T>(0.0), static_cast<T>(0.0),
            static_cast<T>(0.0), static_cast<T>(0.0),
            static_cast<T>(0.0), static_cast<T>(0.0)
        );

    tester.expect_near(lti_mpc_qp_solver.M.matrix.data,
        M_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LTI MPC QP Solver, M matrix.");

    auto gamma_answer = make_DenseMatrix<
        decltype(lti_mpc_qp_solver)::NUMBER_OF_ALL_CONSTRAINTS, 1>(
            static_cast<T>(101.0),
            static_cast<T>(102.0),
            static_cast<T>(181.0),
            static_cast<T>(182.0),
            static_cast<T>(11.0),
            static_cast<T>(102.0),
            static_cast<T>(13.0),
            static_cast<T>(104.0)
        );

    tester.expect_near(lti_mpc_qp_solver.gamma.matrix.data,
        gamma_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LTI MPC QP Solver, gamma vector.");

    LMPC_QP_Solver_Type<NUMBER_OF_VARIABLES, OUTPUT_SIZE,
        U_Type, X_augmented_Type, Phi_Type, F_Type, Weight_U_Nc_Type,
        Delta_U_min_Type, Delta_U_max_Type,
        U_min_Type, U_max_Type,
        Y_min_Type, Y_max_Type> lti_mpc_qp_solver_copy(lti_mpc_qp_solver);

    LMPC_QP_Solver_Type<NUMBER_OF_VARIABLES, OUTPUT_SIZE,
        U_Type, X_augmented_Type, Phi_Type, F_Type, Weight_U_Nc_Type,
        Delta_U_min_Type, Delta_U_max_Type,
        U_min_Type, U_max_Type,
        Y_min_Type, Y_max_Type> lti_mpc_qp_solver_move = lti_mpc_qp_solver_copy;

    lti_mpc_qp_solver = std::move(lti_mpc_qp_solver_move);

    tester.expect_near(lti_mpc_qp_solver.gamma.matrix.data,
        gamma_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LTI MPC QP Solver, gamma vector, copy move.");

    /* 計算 */
    auto ref_vector = make_DenseMatrix<OUTPUT_SIZE, 1>(
        static_cast<T>(1), static_cast<T>(0));
    MPC_ReferenceTrajectory<decltype(ref_vector), NP> reference_trajectory(ref_vector);

    auto Y_min_Empty = make_SparseMatrixEmpty<T, OUTPUT_SIZE, 1>();
    auto Y_max_Empty = make_SparseMatrixEmpty<T, OUTPUT_SIZE, 1>();

    weight_U_NC = make_DiagMatrix<INPUT_SIZE* NC>(
        static_cast<T>(0.001), static_cast<T>(0.001)
    );

    LMPC_QP_Solver_Type<NUMBER_OF_VARIABLES, OUTPUT_SIZE,
        U_Type, X_augmented_Type, Phi_Type, F_Type, Weight_U_Nc_Type,
        Delta_U_min_Type, Delta_U_max_Type,
        U_min_Type, U_max_Type,
        decltype(Y_min_Empty), decltype(Y_max_Empty)> qp_solver =
        make_LMPC_QP_Solver<NUMBER_OF_VARIABLES, OUTPUT_SIZE>(
            U, X_augmented, Phi, F, weight_U_NC,
            delta_U_min, delta_U_max, U_min, U_max, Y_min_Empty, Y_max_Empty);

    auto delta_U = qp_solver.solve(Phi, F, reference_trajectory, X_augmented);

    auto delta_U_answer = make_DenseMatrix<NUMBER_OF_VARIABLES, 1>(
        static_cast<T>(23.53521535), static_cast<T>(19.90670428));

    tester.expect_near(delta_U.matrix.data, delta_U_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LTI MPC QP Solver, solve, delta U.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_LTI_MPC(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonMPC;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-5);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 定義 */
    constexpr std::size_t NP = PythonMPC_ServoMotorData::NP;
    constexpr std::size_t NC = PythonMPC_ServoMotorData::NC;

    constexpr std::size_t INPUT_SIZE = PythonMPC_ServoMotorData::INPUT_SIZE;
    constexpr std::size_t STATE_SIZE = PythonMPC_ServoMotorData::STATE_SIZE;
    constexpr std::size_t OUTPUT_SIZE = PythonMPC_ServoMotorData::OUTPUT_SIZE;

    constexpr std::size_t AUGMENTED_STATE_SIZE = PythonMPC_ServoMotorData::AUGMENTED_STATE_SIZE;

    auto A = make_DenseMatrix<4, 4>(
        static_cast<T>(1.0), static_cast<T>(0.05), static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(-2.56038538), static_cast<T>(0.95000025), static_cast<T>(0.12801927), static_cast<T>(0.0),
        static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(1.0), static_cast<T>(0.05),
        static_cast<T>(6.40099503), static_cast<T>(0.0), static_cast<T>(-0.32004975), static_cast<T>(0.49)
    );

    auto B = make_DenseMatrix<4, 1>(
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.05)
    );

    auto C = make_DenseMatrix<2, 4>(
        static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(1280.19900634), static_cast<T>(0.0), static_cast<T>(-64.00995032), static_cast<T>(0.0)
    );

    auto D = make_DenseMatrixZeros<T, OUTPUT_SIZE, INPUT_SIZE>();

    T dt = static_cast<T>(0.05);

    auto sys = make_DiscreteStateSpace(A, B, C, D, dt);

    auto Q = make_KalmanFilter_Q<STATE_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

    auto R = make_KalmanFilter_R<OUTPUT_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

    auto kalman_filter = make_LinearKalmanFilter(sys, Q, R);

    kalman_filter.G = make_DenseMatrix<STATE_SIZE, OUTPUT_SIZE>(
        static_cast<T>(0.04893929), static_cast<T>(0.00074138),
        static_cast<T>(0.00827874), static_cast<T>(0.00030475),
        static_cast<T>(9.78774203e-01), static_cast<T>(-7.95038792e-04),
        static_cast<T>(2.86510380e-03), static_cast<T>(-3.43928205e-06)
    );

    auto F = PythonMPC_ServoMotorData::get_F<T>();

    auto Phi = PythonMPC_ServoMotorData::get_Phi<T>();

    MPC_PredictionMatrices<decltype(F), decltype(Phi),
        NP, NC, INPUT_SIZE, AUGMENTED_STATE_SIZE, OUTPUT_SIZE> prediction_matrices(F, Phi);

    auto ref = make_DenseMatrix<OUTPUT_SIZE, 1>(
        static_cast<T>(1.0), static_cast<T>(0.0));

    MPC_ReferenceTrajectory_Type<decltype(ref), NP> reference_trajectory(ref);

    auto solver_factor = PythonMPC_ServoMotorData::get_solver_factor<T>();

    auto Weight_U_Nc = make_DiagMatrixIdentity<T, INPUT_SIZE * NC>() * 
        static_cast<T>(0.001);

    auto delta_U_min = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(-100));
    auto delta_U_max = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(100));

    auto U_min = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(-180));
    auto U_max = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(180));

    auto Y_min = make_SparseMatrixEmpty<T, OUTPUT_SIZE, 1>();
    auto Y_max = make_SparseMatrixEmpty<T, OUTPUT_SIZE, 1>();

    LTI_MPC<decltype(kalman_filter), decltype(prediction_matrices),
        decltype(reference_trajectory),
        decltype(delta_U_min),
        decltype(delta_U_max),
        decltype(U_min), decltype(U_max),
        decltype(Y_min), decltype(Y_max),
        decltype(solver_factor)
    > lti_mpc = make_LTI_MPC(
        kalman_filter, prediction_matrices, reference_trajectory, Weight_U_Nc,
        delta_U_min, delta_U_max, U_min, U_max, Y_min, Y_max,
        solver_factor);

    LTI_MPC_Type<decltype(kalman_filter), decltype(prediction_matrices),
        decltype(reference_trajectory),
        decltype(delta_U_min),
        decltype(delta_U_max),
        decltype(U_min), decltype(U_max),
        decltype(Y_min), decltype(Y_max),
        decltype(solver_factor)
    > lti_mpc_copy(lti_mpc);

    LTI_MPC_Type<decltype(kalman_filter), decltype(prediction_matrices),
        decltype(reference_trajectory),
        decltype(delta_U_min),
        decltype(delta_U_max),
        decltype(U_min), decltype(U_max),
        decltype(Y_min), decltype(Y_max),
        decltype(solver_factor)
    > lti_mpc_move = lti_mpc_copy;

    lti_mpc = std::move(lti_mpc_move);

    /* 計算 */
    auto Y = make_StateSpaceOutput<OUTPUT_SIZE>(static_cast<T>(0.0), static_cast<T>(0.0));

    auto U = lti_mpc.update(ref, Y);

    auto U_answer = make_StateSpaceInput<INPUT_SIZE>(static_cast<T>(23.535215353823776));

    tester.expect_near(U.matrix.data, U_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LTI MPC, update.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_LTV_MPC_NoConstraints(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonMPC;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-4);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 定義 */
    constexpr std::size_t NP = PythonMPC_ServoMotorData::NP;
    constexpr std::size_t NC = PythonMPC_ServoMotorData::NC;

    constexpr std::size_t INPUT_SIZE = PythonMPC_ServoMotorData::INPUT_SIZE;
    constexpr std::size_t STATE_SIZE = PythonMPC_ServoMotorData::STATE_SIZE;
    constexpr std::size_t OUTPUT_SIZE = PythonMPC_ServoMotorData::OUTPUT_SIZE;

    constexpr std::size_t AUGMENTED_STATE_SIZE = PythonMPC_ServoMotorData::AUGMENTED_STATE_SIZE;

    auto A = make_DenseMatrix<4, 4>(
        static_cast<T>(1.0), static_cast<T>(0.05), static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(-2.56038538), static_cast<T>(0.95000025), static_cast<T>(0.12801927), static_cast<T>(0.0),
        static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(1.0), static_cast<T>(0.05),
        static_cast<T>(6.40099503), static_cast<T>(0.0), static_cast<T>(-0.32004975), static_cast<T>(0.49)
    );
    using A_Type = decltype(A);

    auto B = make_DenseMatrix<4, 1>(
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.05)
    );
    using B_Type = decltype(B);

    auto C = make_DenseMatrix<2, 4>(
        static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(1280.19900634), static_cast<T>(0.0), static_cast<T>(-64.00995032), static_cast<T>(0.0)
    );
    using C_Type = decltype(C);

    auto D = make_DenseMatrixZeros<T, OUTPUT_SIZE, INPUT_SIZE>();

    T dt = static_cast<T>(0.05);

    auto sys = make_DiscreteStateSpace(A, B, C, D, dt);

    auto Q = make_KalmanFilter_Q<STATE_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

    auto R = make_KalmanFilter_R<OUTPUT_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

    using Weight_U_Nc_Type = DiagMatrix_Type<T, INPUT_SIZE* NC>;

    Weight_U_Nc_Type weight_U_NC = make_DiagMatrixIdentity<T, INPUT_SIZE* NC>();
    weight_U_NC = weight_U_NC * static_cast<T>(0.001);

    auto kalman_filter = make_LinearKalmanFilter(sys, Q, R);
    using LKF_Type = decltype(kalman_filter);

    kalman_filter.converge_G();

    auto F = PythonMPC_ServoMotorData::get_F<T>();
    using F_Type = decltype(F);

    auto Phi = PythonMPC_ServoMotorData::get_Phi<T>();
    using Phi_Type = decltype(Phi);

    using PredictionMatrices_Type = MPC_PredictionMatrices<decltype(F), decltype(Phi),
        NP, NC, INPUT_SIZE, AUGMENTED_STATE_SIZE, OUTPUT_SIZE>;

    PredictionMatrices_Type prediction_matrices(F, Phi);

    auto ref = make_DenseMatrix<OUTPUT_SIZE, 1>(
        static_cast<T>(1.0), static_cast<T>(0.0));

    using ReferenceTrajectory_Type = MPC_ReferenceTrajectory<decltype(ref), NP>;
    ReferenceTrajectory_Type reference_trajectory(ref);

    auto solver_factor = PythonMPC_ServoMotorData::get_solver_factor<T>();
    using SolverFactor_Type = decltype(solver_factor);

    using Parameter_Type = PythonMPC_ServoMotorData::Parameter_Type<T>;

    using EmbeddedIntegratorStateSpace_Type =
        typename EmbeddedIntegratorTypes<A_Type, B_Type, C_Type>::StateSpace_Type;

    MPC_StateSpace_Updater_Function_Object<
        Parameter_Type, typename LKF_Type::DiscreteStateSpace_Type>
        MPC_StateSpace_Updater_Function = 
        PythonMPC_ServoMotorData::mpc_state_space_updater::MPC_StateSpace_Updater::update<
        Parameter_Type, typename LKF_Type::DiscreteStateSpace_Type>;

    LTV_MPC_Phi_F_Updater_Function_Object<
        EmbeddedIntegratorStateSpace_Type, Parameter_Type, Phi_Type, F_Type>
        LTV_MPC_Phi_F_Updater_Function =
        PythonMPC_ServoMotorData::ltv_mpc_phi_f_updater::LTV_MPC_Phi_F_Updater::update<
            EmbeddedIntegratorStateSpace_Type, Parameter_Type, Phi_Type, F_Type>;

    LTV_MPC_NoConstraints<LKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,
        Parameter_Type, SolverFactor_Type> ltv_mpc = make_LTV_MPC_NoConstraints(
            kalman_filter, prediction_matrices, reference_trajectory, solver_factor,
            weight_U_NC, MPC_StateSpace_Updater_Function, LTV_MPC_Phi_F_Updater_Function);

    LTV_MPC_NoConstraints_Type<LKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,
        Parameter_Type, SolverFactor_Type> ltv_mpc_copy(ltv_mpc);

    LTV_MPC_NoConstraints_Type<LKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,
        Parameter_Type, SolverFactor_Type> ltv_mpc_move = ltv_mpc_copy;

    ltv_mpc = std::move(ltv_mpc_move);

    /* 計算 */
    auto Y = make_StateSpaceOutput<OUTPUT_SIZE>(static_cast<T>(0.0), static_cast<T>(0.0));

    auto U = ltv_mpc.update_manipulation(ref, Y);

    auto U_answer = make_StateSpaceInput<INPUT_SIZE>(static_cast<T>(23.535215353823776));

    tester.expect_near(U.matrix.data, U_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LTV MPC, update_manipulation.");

    /* パラメータ更新 */
    T A_3_0_answer = static_cast<T>(2.56039801);
    T B_3_0_answer = static_cast<T>(0.02);
    T Phi_39_0_answer = static_cast<T>(2.26675737e-03);
    T F_39_0_answer = static_cast<T>(-3.90212671e+01);
    T solver_factor_0_39_answer = static_cast<T>(1.58999462);

    Parameter_Type parameter;
    parameter.Mmotor = static_cast<T>(250);

    ltv_mpc.update_parameters(parameter);

    T A_3_0 = ltv_mpc.get_kalman_filter().state_space.A(3, 0);
    T B_3_0 = ltv_mpc.get_kalman_filter().state_space.B(3, 0);
    T Phi_39_0 = ltv_mpc.get_prediction_matrices().Phi(39, 0);
    T F_39_0 = ltv_mpc.get_prediction_matrices().F(39, 0);
    T solver_factor_0_39 = ltv_mpc.get_solver_factor()(0, 39);

    tester.expect_near(A_3_0, A_3_0_answer, NEAR_LIMIT_STRICT,
        "check LTV MPC No Constraints, update_parameters, A(3, 0).");

    tester.expect_near(B_3_0, B_3_0_answer, NEAR_LIMIT_STRICT,
        "check LTV MPC No Constraints, update_parameters, B(3, 0).");

    tester.expect_near(Phi_39_0, Phi_39_0_answer, NEAR_LIMIT_STRICT,
        "check LTV MPC No Constraints, update_parameters, Phi(39, 0).");

    tester.expect_near(F_39_0, F_39_0_answer, NEAR_LIMIT_STRICT,
        "check LTV MPC No Constraints, update_parameters, F(39, 0).");

    tester.expect_near(solver_factor_0_39, solver_factor_0_39_answer, NEAR_LIMIT_STRICT,
        "check LTV MPC No Constraints, update_parameters, solver_factor(0, 39).");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_LTV_MPC(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonMPC;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-4);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 定義 */
    constexpr std::size_t NP = PythonMPC_ServoMotorData::NP;
    constexpr std::size_t NC = PythonMPC_ServoMotorData::NC;

    constexpr std::size_t INPUT_SIZE = PythonMPC_ServoMotorData::INPUT_SIZE;
    constexpr std::size_t STATE_SIZE = PythonMPC_ServoMotorData::STATE_SIZE;
    constexpr std::size_t OUTPUT_SIZE = PythonMPC_ServoMotorData::OUTPUT_SIZE;

    constexpr std::size_t AUGMENTED_STATE_SIZE = PythonMPC_ServoMotorData::AUGMENTED_STATE_SIZE;

    auto A = make_DenseMatrix<4, 4>(
        static_cast<T>(1.0), static_cast<T>(0.05), static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(-2.56038538), static_cast<T>(0.95000025), static_cast<T>(0.12801927), static_cast<T>(0.0),
        static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(1.0), static_cast<T>(0.05),
        static_cast<T>(6.40099503), static_cast<T>(0.0), static_cast<T>(-0.32004975), static_cast<T>(0.49)
    );
    using A_Type = decltype(A);

    auto B = make_DenseMatrix<4, 1>(
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.05)
    );
    using B_Type = decltype(B);

    auto C = make_DenseMatrix<2, 4>(
        static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(1280.19900634), static_cast<T>(0.0), static_cast<T>(-64.00995032), static_cast<T>(0.0)
    );
    using C_Type = decltype(C);

    auto D = make_DenseMatrixZeros<T, OUTPUT_SIZE, INPUT_SIZE>();

    T dt = static_cast<T>(0.05);

    auto sys = make_DiscreteStateSpace(A, B, C, D, dt);

    auto Q = make_KalmanFilter_Q<STATE_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

    auto R = make_KalmanFilter_R<OUTPUT_SIZE>(
        static_cast<T>(1.0),
        static_cast<T>(1.0)
    );

    auto kalman_filter = make_LinearKalmanFilter(sys, Q, R);
    using LKF_Type = decltype(kalman_filter);

    kalman_filter.G = make_DenseMatrix<STATE_SIZE, OUTPUT_SIZE>(
        static_cast<T>(0.04893929), static_cast<T>(0.00074138),
        static_cast<T>(0.00827874), static_cast<T>(0.00030475),
        static_cast<T>(9.78774203e-01), static_cast<T>(-7.95038792e-04),
        static_cast<T>(2.86510380e-03), static_cast<T>(-3.43928205e-06)
    );

    auto F = PythonMPC_ServoMotorData::get_F<T>();
    using F_Type = decltype(F);

    auto Phi = PythonMPC_ServoMotorData::get_Phi<T>();
    using Phi_Type = decltype(Phi);

    MPC_PredictionMatrices<decltype(F), decltype(Phi),
        NP, NC, INPUT_SIZE, AUGMENTED_STATE_SIZE, OUTPUT_SIZE> prediction_matrices(F, Phi);

    auto ref = make_DenseMatrix<OUTPUT_SIZE, 1>(
        static_cast<T>(1.0), static_cast<T>(0.0));

    MPC_ReferenceTrajectory_Type<decltype(ref), NP> reference_trajectory(ref);

    auto solver_factor = PythonMPC_ServoMotorData::get_solver_factor<T>();

    using Parameter_Type = PythonMPC_ServoMotorData::Parameter_Type<T>;

    auto Weight_U_Nc = make_DiagMatrixIdentity<T, INPUT_SIZE* NC>()*
        static_cast<T>(0.001);

    auto delta_U_min = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(-100));
    auto delta_U_max = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(100));

    auto U_min = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(-180));
    auto U_max = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(180));

    auto Y_min = make_SparseMatrixEmpty<T, OUTPUT_SIZE, 1>();
    auto Y_max = make_SparseMatrixEmpty<T, OUTPUT_SIZE, 1>();

    using EmbeddedIntegratorStateSpace_Type =
        typename EmbeddedIntegratorTypes<A_Type, B_Type, C_Type>::StateSpace_Type;

    MPC_StateSpace_Updater_Function_Object<
        Parameter_Type, typename LKF_Type::DiscreteStateSpace_Type>
        MPC_StateSpace_Updater_Function =
        PythonMPC_ServoMotorData::mpc_state_space_updater::MPC_StateSpace_Updater::update<
        Parameter_Type, typename LKF_Type::DiscreteStateSpace_Type>;

    LTV_MPC_Phi_F_Updater_Function_Object<
        EmbeddedIntegratorStateSpace_Type, Parameter_Type, Phi_Type, F_Type>
        LTV_MPC_Phi_F_Updater_Function =
        PythonMPC_ServoMotorData::ltv_mpc_phi_f_updater::LTV_MPC_Phi_F_Updater::update<
        EmbeddedIntegratorStateSpace_Type, Parameter_Type, Phi_Type, F_Type>;

    LTV_MPC<decltype(kalman_filter), decltype(prediction_matrices),
        decltype(reference_trajectory), Parameter_Type,
        decltype(delta_U_min),
        decltype(delta_U_max),
        decltype(U_min), decltype(U_max),
        decltype(Y_min), decltype(Y_max),
        decltype(solver_factor)
    > ltv_mpc = make_LTV_MPC(kalman_filter, prediction_matrices,
        reference_trajectory, Weight_U_Nc,
        MPC_StateSpace_Updater_Function,
        LTV_MPC_Phi_F_Updater_Function,
        delta_U_min, delta_U_max, U_min, U_max, Y_min, Y_max,
        solver_factor);

    LTV_MPC_Type<decltype(kalman_filter), decltype(prediction_matrices),
        decltype(reference_trajectory), Parameter_Type,
        decltype(delta_U_min),
        decltype(delta_U_max),
        decltype(U_min), decltype(U_max),
        decltype(Y_min), decltype(Y_max),
        decltype(solver_factor)
    > ltv_mpc_copy(ltv_mpc);

    LTV_MPC_Type<decltype(kalman_filter), decltype(prediction_matrices),
        decltype(reference_trajectory), Parameter_Type,
        decltype(delta_U_min),
        decltype(delta_U_max),
        decltype(U_min), decltype(U_max),
        decltype(Y_min), decltype(Y_max),
        decltype(solver_factor)
    > ltv_mpc_move = ltv_mpc_copy;

    ltv_mpc = std::move(ltv_mpc_move);

    /* 計算 */
    auto Y = make_StateSpaceOutput<OUTPUT_SIZE>(static_cast<T>(0.0), static_cast<T>(0.0));

    auto U = ltv_mpc.update_manipulation(ref, Y);

    auto U_answer = make_StateSpaceInput<INPUT_SIZE>(static_cast<T>(23.535215353823776));

    tester.expect_near(U.matrix.data, U_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LTV MPC, update.");

    /* パラメータ更新 */
    T A_3_0_answer = static_cast<T>(2.56039801);
    T B_3_0_answer = static_cast<T>(0.02);
    T Phi_39_0_answer = static_cast<T>(2.26675737e-03);
    T F_39_0_answer = static_cast<T>(-3.90212671e+01);
    T solver_factor_0_39_answer = static_cast<T>(1.58999462);

    Parameter_Type parameter;
    parameter.Mmotor = static_cast<T>(250);

    ltv_mpc.update_parameters(parameter);

    T A_3_0 = ltv_mpc.get_kalman_filter().state_space.A(3, 0);
    T B_3_0 = ltv_mpc.get_kalman_filter().state_space.B(3, 0);
    T Phi_39_0 = ltv_mpc.get_prediction_matrices().Phi(39, 0);
    T F_39_0 = ltv_mpc.get_prediction_matrices().F(39, 0);
    T solver_factor_0_39 = ltv_mpc.get_solver_factor()(0, 39);

    tester.expect_near(A_3_0, A_3_0_answer, NEAR_LIMIT_STRICT,
        "check LTV MPC, update_parameters, A(3, 0).");

    tester.expect_near(B_3_0, B_3_0_answer, NEAR_LIMIT_STRICT,
        "check LTV MPC, update_parameters, B(3, 0).");

    tester.expect_near(Phi_39_0, Phi_39_0_answer, NEAR_LIMIT_STRICT,
        "check LTV MPC, update_parameters, Phi(39, 0).");

    tester.expect_near(F_39_0, F_39_0_answer, NEAR_LIMIT_STRICT,
        "check LTV MPC, update_parameters, F(39, 0).");

    tester.expect_near(solver_factor_0_39, solver_factor_0_39_answer, NEAR_LIMIT_STRICT,
        "check LTV MPC, update_parameters, solver_factor(0, 39).");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_Adaptive_MPC_NoConstraints(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonMPC;

    MCAPTester<T> tester;

    // There is a Floating point Numerical instability problem.
    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0);

    /* 定義 */
    constexpr std::size_t NP = PythonMPC_TwoWheelVehicleModelData::NP;
    constexpr std::size_t NC = PythonMPC_TwoWheelVehicleModelData::NC;

    constexpr std::size_t INPUT_SIZE = PythonMPC_TwoWheelVehicleModelData::INPUT_SIZE;
    constexpr std::size_t STATE_SIZE = PythonMPC_TwoWheelVehicleModelData::STATE_SIZE;
    constexpr std::size_t OUTPUT_SIZE = PythonMPC_TwoWheelVehicleModelData::OUTPUT_SIZE;

    constexpr std::size_t AUGMENTED_STATE_SIZE = PythonMPC_TwoWheelVehicleModelData::AUGMENTED_STATE_SIZE;

    using EKF_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_ekf::type<T>;

    using A_Type = typename EKF_Type::A_Type;

    using B_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_B::type<T>;

    using C_Type = typename EKF_Type::C_Type;

    using X_Type = StateSpaceState_Type<T, STATE_SIZE>;

    using Y_Type = StateSpaceOutput_Type<T, OUTPUT_SIZE>;

    using U_Type = StateSpaceInput_Type<T, INPUT_SIZE>;

    using F_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_F::type<T>;

    using Phi_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_Phi::type<T>;

    using SolverFactor_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_solver_factor::type<T>;

    using PredictionMatrices_Type = MPC_PredictionMatrices_Type<
        F_Type, Phi_Type, NP, NC, INPUT_SIZE, AUGMENTED_STATE_SIZE, OUTPUT_SIZE>;

    using Ref_Type = DenseMatrix_Type<T, OUTPUT_SIZE, 1>;

    using ReferenceTrajectory_Type = MPC_ReferenceTrajectory_Type<
        Ref_Type, NP>;

    using Parameter_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_ekf_parameter::Parameter_Type<T>;

    using Weight_U_Nc_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_Weight_U_Nc::type<T>;

    using EmbeddedIntegratorStateSpace_Type =
        typename EmbeddedIntegratorTypes<A_Type, B_Type, C_Type>::StateSpace_Type;

    auto kalman_filter = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_ekf::make<T>();
    kalman_filter.X_hat.template set<0, 0>(static_cast<T>(0.0));
    kalman_filter.X_hat.template set<1, 0>(static_cast<T>(0.0));
    kalman_filter.X_hat.template set<2, 0>(static_cast<T>(0.0));
    kalman_filter.X_hat.template set<3, 0>(static_cast<T>(0.0));
    kalman_filter.X_hat.template set<4, 0>(static_cast<T>(0.0));
    kalman_filter.X_hat.template set<5, 0>(static_cast<T>(10.0));

    auto F = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_F::make<T>();

    auto Phi = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_Phi::make<T>();

    auto solver_factor = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_solver_factor::make<T>();

    PredictionMatrices_Type prediction_matrices(F, Phi);

    ReferenceTrajectory_Type reference_trajectory;

    Weight_U_Nc_Type Weight_U_Nc = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_Weight_U_Nc::make<T>();

    auto Adaptive_MPC_Phi_F_Updater_Function = get_adaptive_mpc_phi_f_updater_function<T>();

    AdaptiveMPC_NoConstraints_Type<B_Type,
        EKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,
        Parameter_Type, SolverFactor_Type> ada_mpc =
        make_AdaptiveMPC_NoConstraints<B_Type, EKF_Type, PredictionMatrices_Type,
            ReferenceTrajectory_Type, Parameter_Type,
            SolverFactor_Type, Weight_U_Nc_Type,
            X_Type, U_Type,
            EmbeddedIntegratorStateSpace_Type>(
    kalman_filter, prediction_matrices, reference_trajectory, solver_factor,
    Weight_U_Nc, Adaptive_MPC_Phi_F_Updater_Function);

    AdaptiveMPC_NoConstraints_Type<B_Type,
        EKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,
        Parameter_Type, SolverFactor_Type> ada_mpc_copy(ada_mpc);

    AdaptiveMPC_NoConstraints_Type <B_Type,
        EKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,
        Parameter_Type, SolverFactor_Type> ada_mpc_move = ada_mpc_copy;

    ada_mpc = std::move(ada_mpc_move);

    /* 代入チェック */
    T F_19_10 = ada_mpc.get_F().template get<19, 10>();
    T F_19_10_answer = static_cast<T>(1.0);

    tester.expect_near(F_19_10, F_19_10_answer, NEAR_LIMIT_STRICT,
        "check Adaptive MPC No Constraints, F(19, 10).");

    T Phi_19_1 = ada_mpc.get_Phi().template get<19, 1>();
    T Phi_19_1_answer = static_cast<T>(0.04);

    tester.expect_near(Phi_19_1, Phi_19_1_answer, NEAR_LIMIT_STRICT,
        "check Adaptive MPC No Constraints, Phi(19, 1).");

    T solver_factor_1_19 = ada_mpc.get_solver_factor().template get <1, 19>();
    T solver_factor_1_19_answer = static_cast<T>(0.3883477801943797);

    tester.expect_near(solver_factor_1_19, solver_factor_1_19_answer, NEAR_LIMIT_STRICT,
        "check Adaptive MPC No Constraints, solver_factor(1, 19).");

    Parameter_Type parameter_test;
    parameter_test.m = static_cast<T>(0.0);
    ada_mpc.update_parameters(parameter_test);
    parameter_test.m = static_cast<T>(2000);
    ada_mpc.update_parameters(parameter_test);

    /* 計算 */
    Ref_Type ref;
    ref(0, 0) = static_cast<T>(0.15);
    ref(1, 0) = static_cast<T>(0.0);
    ref(2, 0) = static_cast<T>(0.0);
    ref(3, 0) = static_cast<T>(0.0);
    ref(4, 0) = static_cast<T>(15.0);

    Y_Type y_measured;
    y_measured(0, 0) = static_cast<T>(0.1);
    y_measured(1, 0) = static_cast<T>(0.0);
    y_measured(2, 0) = static_cast<T>(0.0);
    y_measured(3, 0) = static_cast<T>(0.0);
    y_measured(4, 0) = static_cast<T>(10.0);

    U_Type U_answer;
    U_answer(0, 0) = static_cast<T>(0.0);
    U_answer(1, 0) = static_cast<T>(4.85143464);

    auto U = ada_mpc.update_manipulation(ref, y_measured);

    tester.expect_near(U.matrix.data, U_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check Adaptive MPC No Constraints, update_manipulation, U.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_Adaptive_MPC(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonMPC;

    MCAPTester<T> tester;

    // There is a Floating point Numerical instability problem.
    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0);

    /* 定義 */
    constexpr std::size_t NP = PythonMPC_TwoWheelVehicleModelData::NP;
    constexpr std::size_t NC = PythonMPC_TwoWheelVehicleModelData::NC;

    constexpr std::size_t INPUT_SIZE = PythonMPC_TwoWheelVehicleModelData::INPUT_SIZE;
    constexpr std::size_t STATE_SIZE = PythonMPC_TwoWheelVehicleModelData::STATE_SIZE;
    constexpr std::size_t OUTPUT_SIZE = PythonMPC_TwoWheelVehicleModelData::OUTPUT_SIZE;

    constexpr std::size_t AUGMENTED_STATE_SIZE = PythonMPC_TwoWheelVehicleModelData::AUGMENTED_STATE_SIZE;

    using EKF_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_ekf::type<T>;

    using A_Type = typename EKF_Type::A_Type;

    using B_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_B::type<T>;

    using C_Type = typename EKF_Type::C_Type;

    using X_Type = StateSpaceState_Type<T, STATE_SIZE>;

    using Y_Type = StateSpaceOutput_Type<T, OUTPUT_SIZE>;

    using U_Type = StateSpaceInput_Type<T, INPUT_SIZE>;

    using F_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_F::type<T>;

    using Phi_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_Phi::type<T>;

    using SolverFactor_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_solver_factor::type<T>;

    using PredictionMatrices_Type = MPC_PredictionMatrices_Type<
        F_Type, Phi_Type, NP, NC, INPUT_SIZE, AUGMENTED_STATE_SIZE, OUTPUT_SIZE>;

    using Ref_Type = DenseMatrix_Type<T, OUTPUT_SIZE, 1>;

    using ReferenceTrajectory_Type = MPC_ReferenceTrajectory_Type<
        Ref_Type, NP>;

    using Parameter_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_ekf_parameter::Parameter_Type<T>;

    using Weight_U_Nc_Type = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_Weight_U_Nc::type<T>;

    using EmbeddedIntegratorStateSpace_Type =
        typename EmbeddedIntegratorTypes<A_Type, B_Type, C_Type>::StateSpace_Type;

    auto kalman_filter = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_ekf::make<T>();
    kalman_filter.X_hat.template set<0, 0>(static_cast<T>(0.0));
    kalman_filter.X_hat.template set<1, 0>(static_cast<T>(0.0));
    kalman_filter.X_hat.template set<2, 0>(static_cast<T>(0.0));
    kalman_filter.X_hat.template set<3, 0>(static_cast<T>(0.0));
    kalman_filter.X_hat.template set<4, 0>(static_cast<T>(0.0));
    kalman_filter.X_hat.template set<5, 0>(static_cast<T>(10.0));

    auto F = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_F::make<T>();

    auto Phi = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_Phi::make<T>();

    auto solver_factor = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_solver_factor::make<T>();

    PredictionMatrices_Type prediction_matrices(F, Phi);

    ReferenceTrajectory_Type reference_trajectory;

    Weight_U_Nc_Type Weight_U_Nc = PythonMPC_TwoWheelVehicleModelData::two_wheel_vehicle_model_ada_mpc_Weight_U_Nc::make<T>();

    auto Adaptive_MPC_Phi_F_Updater_Function = get_adaptive_mpc_phi_f_updater_function<T>();


    auto delta_U_min = make_SparseMatrixEmpty<T, INPUT_SIZE, 1>();
    auto delta_U_max = make_SparseMatrixEmpty<T, INPUT_SIZE, 1>();

    auto U_min = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(-1.0),
        static_cast<T>(-10.0));
    auto U_max = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(1.0),
        static_cast<T>(10.0));

    auto Y_min = make_SparseMatrixEmpty<T, INPUT_SIZE, 1>();
    auto Y_max = make_SparseMatrixEmpty<T, INPUT_SIZE, 1>();

    AdaptiveMPC_Type<B_Type, EKF_Type, PredictionMatrices_Type,
        ReferenceTrajectory_Type, Parameter_Type, decltype(delta_U_min),
        decltype(delta_U_max), decltype(U_min), decltype(U_max), decltype(Y_min),
        decltype(Y_max), SolverFactor_Type> ada_mpc = make_AdaptiveMPC<
        B_Type, EKF_Type, PredictionMatrices_Type,
        ReferenceTrajectory_Type, Parameter_Type,
        decltype(delta_U_min), decltype(delta_U_max),
        decltype(U_min), decltype(U_max), decltype(Y_min),
        decltype(Y_max), SolverFactor_Type,
        Weight_U_Nc_Type, X_Type, U_Type,
        EmbeddedIntegratorStateSpace_Type>(
            kalman_filter,
            prediction_matrices, reference_trajectory,
            Weight_U_Nc, Adaptive_MPC_Phi_F_Updater_Function,
            delta_U_min, delta_U_max,
            U_min, U_max, Y_min, Y_max,
            solver_factor);

    AdaptiveMPC_Type<B_Type, EKF_Type, PredictionMatrices_Type,
        ReferenceTrajectory_Type, Parameter_Type, decltype(delta_U_min),
        decltype(delta_U_max), decltype(U_min), decltype(U_max), decltype(Y_min),
        decltype(Y_max), SolverFactor_Type> ada_mpc_copy(ada_mpc);

    AdaptiveMPC_Type <B_Type, EKF_Type, PredictionMatrices_Type,
        ReferenceTrajectory_Type, Parameter_Type, decltype(delta_U_min),
        decltype(delta_U_max), decltype(U_min), decltype(U_max), decltype(Y_min),
        decltype(Y_max), SolverFactor_Type> ada_mpc_move = ada_mpc_copy;

    ada_mpc = std::move(ada_mpc_move);

    /* 代入チェック */
    T F_19_10 = ada_mpc.get_F().template get<19, 10>();
    T F_19_10_answer = static_cast<T>(1.0);

    tester.expect_near(F_19_10, F_19_10_answer, NEAR_LIMIT_STRICT,
        "check Adaptive MPC No Constraints, F(19, 10).");

    T Phi_19_1 = ada_mpc.get_Phi().template get<19, 1>();
    T Phi_19_1_answer = static_cast<T>(0.04);

    tester.expect_near(Phi_19_1, Phi_19_1_answer, NEAR_LIMIT_STRICT,
        "check Adaptive MPC No Constraints, Phi(19, 1).");

    T solver_factor_1_19 = ada_mpc.get_solver_factor().template get <1, 19>();
    T solver_factor_1_19_answer = static_cast<T>(0.3883477801943797);

    tester.expect_near(solver_factor_1_19, solver_factor_1_19_answer, NEAR_LIMIT_STRICT,
        "check Adaptive MPC No Constraints, solver_factor(1, 19).");

    Parameter_Type parameter_test;
    parameter_test.m = static_cast<T>(0.0);
    ada_mpc.update_parameters(parameter_test);
    parameter_test.m = static_cast<T>(2000);
    ada_mpc.update_parameters(parameter_test);

    /* 計算 */
    Ref_Type ref;
    ref(0, 0) = static_cast<T>(0.15);
    ref(1, 0) = static_cast<T>(0.0);
    ref(2, 0) = static_cast<T>(0.0);
    ref(3, 0) = static_cast<T>(0.0);
    ref(4, 0) = static_cast<T>(15.0);

    Y_Type y_measured;
    y_measured(0, 0) = static_cast<T>(0.1);
    y_measured(1, 0) = static_cast<T>(0.0);
    y_measured(2, 0) = static_cast<T>(0.0);
    y_measured(3, 0) = static_cast<T>(0.0);
    y_measured(4, 0) = static_cast<T>(10.0);

    U_Type U_answer;
    U_answer(0, 0) = static_cast<T>(0.0);
    U_answer(1, 0) = static_cast<T>(4.85143464);

    auto U = ada_mpc.update_manipulation(ref, y_measured);

    tester.expect_near(U.matrix.data, U_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check Adaptive MPC, update_manipulation, U.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_Nonlinear_MPC(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonMPC;

    MCAPTester<T> tester;

    // There is a Floating point Numerical instability problem.
    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0);

    using namespace PythonMPC_KinematicBicycleModelData;

    /* 定義 */
    constexpr std::size_t NP = kinematic_bicycle_model_cost_matrices::NP;

    //constexpr std::size_t INPUT_SIZE = kinematic_bicycle_model_cost_matrices::INPUT_SIZE;
    constexpr std::size_t STATE_SIZE = kinematic_bicycle_model_cost_matrices::STATE_SIZE;
    constexpr std::size_t OUTPUT_SIZE = kinematic_bicycle_model_cost_matrices::OUTPUT_SIZE;

    T delta_time = static_cast<T>(0.1);

    auto X_initial = make_DenseMatrix<STATE_SIZE, 1>(
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.0)
    );

    using EKF_Type = kinematic_bicycle_model_nmpc_ekf::type<T>;

    using Cost_Matrices_Type = kinematic_bicycle_model_cost_matrices::type<T>;

    auto kalman_filter = kinematic_bicycle_model_nmpc_ekf::make<T>();

    auto cost_matrices = kinematic_bicycle_model_cost_matrices::make<T>();

    using Nonlinear_MPC_Type = NonlinearMPC_TwiceDifferentiable_Type<EKF_Type, Cost_Matrices_Type>;

    NonlinearMPC_TwiceDifferentiable_Type<EKF_Type, Cost_Matrices_Type> nonlinear_mpc
        = make_NonlinearMPC_TwiceDifferentiable(kalman_filter, cost_matrices, delta_time, X_initial);

    /* コピー、ムーブ */
    nonlinear_mpc.U_horizon(0, 0) = static_cast<T>(1.0);

    Nonlinear_MPC_Type nonlinear_mpc_copy(nonlinear_mpc);
    Nonlinear_MPC_Type nonlinear_mpc_move = nonlinear_mpc_copy;
    nonlinear_mpc = std::move(nonlinear_mpc_move);

    tester.expect_near(nonlinear_mpc.U_horizon.matrix.data,
        nonlinear_mpc_copy.U_horizon.matrix.data, NEAR_LIMIT_STRICT,
        "check Nonlinear MPC, copy, U_horizon(0, 0).");

    nonlinear_mpc.U_horizon(0, 0) = static_cast<T>(0.0);

    /* 計算 */
    using Parameter_Type = kinematic_bicycle_model_nmpc_ekf_parameter::Parameter_Type<T>;
    Parameter_Type parameter;

    nonlinear_mpc.set_solver_max_iteration(5);

    auto reference = make_DenseMatrixOnes<T, OUTPUT_SIZE, 1>();

    nonlinear_mpc.set_reference_trajectory(reference);


    using ReferenceTrajectory_Type = DenseMatrix_Type<T, OUTPUT_SIZE, NP>;

    ReferenceTrajectory_Type reference_trajectory({
        {static_cast<T>(0.0), static_cast<T>(-0.05178651), static_cast<T>(-0.10357302), static_cast<T>(-0.15535954), static_cast<T>(-0.20714605),
        static_cast<T>(-0.25893256), static_cast<T>(-0.31071907), static_cast<T>(-0.36250559), static_cast<T>(-4.1429210), static_cast<T>(-0.46607861) },
        { static_cast<T>(0.0), static_cast<T>(-0.02239073), static_cast<T>(-0.04478147), static_cast<T>(-0.0671722), static_cast<T>(-0.08956293),
        static_cast<T>(-0.11195366), static_cast<T>(-0.1343444), static_cast<T>(-0.15673513), static_cast<T>(-0.17912586), static_cast<T>(-0.2015166) },
        { static_cast<T>(3.26794897e-07), static_cast<T>(1.05347955e-02), static_cast<T>(2.10680951e-02), static_cast<T>(3.15990565e-02), static_cast<T>(4.21265112e-02),
        static_cast<T>(5.26492908e-02), static_cast<T>(6.31662274e-02), static_cast<T>(7.36761540e-02), static_cast<T>(8.41779041e-02), static_cast<T>(9.46703123e-02) },
        { static_cast<T>(-1.0), static_cast<T>(-0.99994451), static_cast<T>(-0.99977804), static_cast<T>(-0.99950063), static_cast<T>(-0.99911228),
        static_cast<T>(-0.99861306), static_cast<T>(-0.99800302), static_cast<T>(-0.99728222), static_cast<T>(-0.99645074), static_cast<T>(-0.99550868) }
    });

    auto y_measured = make_DenseMatrix<OUTPUT_SIZE, 1>(
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(3.26794897e-07),
        static_cast<T>(-0.9999999999999466)
    );

    nonlinear_mpc.update_parameters(parameter);

    //auto u_from_mpc = nonlinear_mpc.update_manipulation(reference_trajectory, y_measured);


    tester.throw_error_if_test_failed();
}


int main(void) {

    check_MPC_PredictionMatrices<double>();

    check_MPC_PredictionMatrices<float>();

    check_MPC_ReferenceTrajectory<double>();

    check_MPC_ReferenceTrajectory<float>();

    check_LTI_MPC_NoConstraints<double>();

    check_LTI_MPC_NoConstraints<float>();

    check_DU_U_Y_Limits<double>();

    check_DU_U_Y_Limits<float>();

    check_LMPC_QP_Solver<double>();

    check_LMPC_QP_Solver<float>();

    check_LTI_MPC<double>();

    check_LTI_MPC<float>();

    check_LTV_MPC_NoConstraints<double>();

    check_LTV_MPC_NoConstraints<float>();

    check_LTV_MPC<double>();

    check_LTV_MPC<float>();

    check_Adaptive_MPC_NoConstraints<double>();

    check_Adaptive_MPC_NoConstraints<float>();

    check_Adaptive_MPC<double>();

    check_Adaptive_MPC<float>();

    check_Nonlinear_MPC<double>();


    return 0;
}
