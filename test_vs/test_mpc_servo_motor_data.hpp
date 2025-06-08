#ifndef __TEST_MPC_SERVO_MOTOR_DATA_HPP__
#define __TEST_MPC_SERVO_MOTOR_DATA_HPP__

#include "python_mpc.hpp"

using namespace PythonNumpy;

namespace PythonMPC_ServoMotorData {

static constexpr std::size_t Np = 20;
static constexpr std::size_t Nc = 2;

static constexpr std::size_t INPUT_SIZE = 1;
static constexpr std::size_t STATE_SIZE = 4;
static constexpr std::size_t OUTPUT_SIZE = 2;

static constexpr std::size_t AUGMENTED_STATE_SIZE = STATE_SIZE + OUTPUT_SIZE;

template <typename T>
auto get_F(void) -> DenseMatrix_Type<T, Np* OUTPUT_SIZE, AUGMENTED_STATE_SIZE> {

    auto F = make_DenseMatrix<Np * OUTPUT_SIZE, AUGMENTED_STATE_SIZE>(
        static_cast<T>(1.0), static_cast<T>(0.05), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(6.40099503e+00), static_cast<T>(3.20049752e-01), static_cast<T>(-3.20049752e-01), static_cast<T>(-1.60024876e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(1.87198073), static_cast<T>(0.14750001), static_cast<T>(0.00640096), static_cast<T>(0.0), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(1.18801075e+01), static_cast<T>(9.44146846e-01), static_cast<T>(-5.94005376e-01), static_cast<T>(-3.98461941e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(2.49432386e+00), static_cast<T>(2.83724085e-01), static_cast<T>(2.52838072e-02), static_cast<T>(3.20048173e-04), static_cast<T>(1.00000000e+00), static_cast<T>(0.00000000e+00),
        static_cast<T>(1.56086675e+01), static_cast<T>(1.81099486e+00), static_cast<T>(-7.80433373e-01), static_cast<T>(-6.52273915e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(2.76992948e+00), static_cast<T>(4.44254143e-01), static_cast<T>(6.15035259e-02), static_cast<T>(1.42101397e-03), static_cast<T>(1.00000000e+00), static_cast<T>(0.00000000e+00),
        static_cast<T>(1.69552975e+01), static_cast<T>(2.82092869e+00), static_cast<T>(-8.47764876e-01), static_cast<T>(-8.69855781e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(2.64156357), static_cast<T>(0.61053802), static_cast<T>(0.11792182), static_cast<T>(0.00377147), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(1.55768337e+01), static_cast<T>(3.84769758e+00), static_cast<T>(-7.78841685e-01), static_cast<T>(-1.01013665e-01), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(2.10249213), static_cast<T>(0.76208945), static_cast<T>(0.19487539), static_cast<T>(0.00774411), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(1.14796521e+01), static_cast<T>(4.75420509e+00), static_cast<T>(-5.73982606e-01), static_cast<T>(-1.04441267e-01), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(1.20081947), static_cast<T>(0.87910977), static_cast<T>(0.28995903), static_cast<T>(0.01353839), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(5.03952190e+00), static_cast<T>(5.41052836e+00), static_cast<T>(-2.51976095e-01), static_cast<T>(-9.58778389e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(0.0366188), static_cast<T>(0.94519547), static_cast<T>(0.39816906), static_cast<T>(0.02113176), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(-3.02623438e+00), static_cast<T>(5.71202913e+00), static_cast<T>(1.51311719e-01), static_cast<T>(-7.55814334e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(-1.24818158), static_cast<T>(0.94976687), static_cast<T>(0.51240908), static_cast<T>(0.03026302), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(-1.17340316e+01), static_cast<T>(5.59516711e+00), static_cast<T>(5.86701581e-01), static_cast<T>(-4.54718040e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(-2.48623738), static_cast<T>(0.88986968), static_cast<T>(0.62431187), static_cast<T>(0.04044933), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(-1.99498855e+01), static_cast<T>(5.04875831e+00), static_cast<T>(9.97494273e-01), static_cast<T>(-8.94859249e-03), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(-3.50573074), static_cast<T>(0.77106455), static_cast<T>(0.72528654), static_cast<T>(0.05103577), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(-2.65329373e+01), static_cast<T>(4.11887711e+00), static_cast<T>(1.32664687e+00), static_cast<T>(2.94874158e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(-4.15327346), static_cast<T>(0.60722498), static_cast<T>(0.80766367), static_cast<T>(0.06127185), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(-3.04891062e+01), static_cast<T>(2.90633716e+00), static_cast<T>(1.52445531e+00), static_cast<T>(6.47786894e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(-4.31580259), static_cast<T>(0.4192002), static_cast<T>(0.86579013), static_cast<T>(0.07040639), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(-3.11148063e+01), static_cast<T>(1.55661546e+00), static_cast<T>(1.55574032e+00), static_cast<T>(9.19618358e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(-3.9384457), static_cast<T>(0.23245017), static_cast<T>(0.89692229), static_cast<T>(0.07778864), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(-2.81106995e+01), static_cast<T>(2.43094506e-01), static_cast<T>(1.40553497e+00), static_cast<T>(1.06845828e-01), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(-3.03568303), static_cast<T>(0.07390543), static_cast<T>(0.90178415), static_cast<T>(0.08296255), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(-2.16482005e+01), static_cast<T>(-8.54545383e-01), static_cast<T>(1.08241002e+00), static_cast<T>(1.06628717e-01), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(-1.69386656), static_cast<T>(-0.03157397), static_cast<T>(0.88469333), static_cast<T>(0.08574086), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(-1.23767100e+01), static_cast<T>(-1.57417860e+00), static_cast<T>(6.18835502e-01), static_cast<T>(9.03660848e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(-0.06419823), static_cast<T>(-0.06468861), static_cast<T>(0.85320991), static_cast<T>(0.08624769), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(-1.36677828), static_cast<T>(-1.79425581), static_cast<T>(0.06833891), static_cast<T>(0.05921867), static_cast<T>(0.0), static_cast<T>(0.005),
        static_cast<T>(1.65350056), static_cast<T>(-0.01466411), static_cast<T>(0.81732497), static_cast<T>(0.08492186), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(1.00072615e+01), static_cast<T>(-1.45283262e+00), static_cast<T>(-5.00363075e-01), static_cast<T>(1.64316060e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(3.23463074), static_cast<T>(0.11874412), static_cast<T>(0.78826846), static_cast<T>(0.08247796), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(2.02332466e+01), static_cast<T>(-5.59778521e-01), static_cast<T>(-1.01166233e+00), static_cast<T>(-3.29691544e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
        static_cast<T>(4.45854104), static_cast<T>(0.32453848), static_cast<T>(0.77707295), static_cast<T>(0.07982762), static_cast<T>(1.0), static_cast<T>(0.0),
        static_cast<T>(2.78564549e+01), static_cast<T>(7.99922346e-01), static_cast<T>(-1.39282275e+00), static_cast<T>(-8.27404896e-02), static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03)
    );

    return F;
}

template <typename T>
auto get_Phi(void) -> DenseMatrix_Type<T, Np* OUTPUT_SIZE, INPUT_SIZE* Nc> {

    auto Phi = make_DenseMatrix<Np * OUTPUT_SIZE, INPUT_SIZE* Nc>(
        static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(-0.00080012), static_cast<T>(0.0),
        static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(-0.00199231), static_cast<T>(-0.00080012),
        static_cast<T>(1.60024087e-05), static_cast<T>(0.00000000e+00),
        static_cast<T>(-0.00326137), static_cast<T>(-0.00199231),
        static_cast<T>(7.10506984e-05), static_cast<T>(1.60024087e-05),
        static_cast<T>(-0.00434928), static_cast<T>(-0.00326137),
        static_cast<T>(1.88573657e-04), static_cast<T>(7.10506984e-05),
        static_cast<T>(-0.00505068), static_cast<T>(-0.00434928),
        static_cast<T>(0.00038721), static_cast<T>(0.00018857),
        static_cast<T>(-0.00522206), static_cast<T>(-0.00505068),
        static_cast<T>(0.00067692), static_cast<T>(0.00038721),
        static_cast<T>(-0.00479389), static_cast<T>(-0.00522206),
        static_cast<T>(0.00105659), static_cast<T>(0.00067692),
        static_cast<T>(-0.00377907), static_cast<T>(-0.00479389),
        static_cast<T>(0.00151315), static_cast<T>(0.00105659),
        static_cast<T>(-0.00227359), static_cast<T>(-0.00377907),
        static_cast<T>(0.00202247), static_cast<T>(0.00151315),
        static_cast<T>(-0.00044743), static_cast<T>(-0.00227359),
        static_cast<T>(0.00255179), static_cast<T>(0.00202247),
        static_cast<T>(0.00147437), static_cast<T>(-0.00044743),
        static_cast<T>(0.00306359), static_cast<T>(0.00255179),
        static_cast<T>(0.00323893), static_cast<T>(0.00147437),
        static_cast<T>(0.00352032), static_cast<T>(0.00306359),
        static_cast<T>(0.00459809), static_cast<T>(0.00323893),
        static_cast<T>(0.00388943), static_cast<T>(0.00352032),
        static_cast<T>(0.00534229), static_cast<T>(0.00459809),
        static_cast<T>(0.00414813), static_cast<T>(0.00388943),
        static_cast<T>(0.00533144), static_cast<T>(0.00534229),
        static_cast<T>(0.00428704), static_cast<T>(0.00414813),
        static_cast<T>(0.0045183), static_cast<T>(0.00533144),
        static_cast<T>(0.00431238), static_cast<T>(0.00428704),
        static_cast<T>(0.00296093), static_cast<T>(0.0045183),
        static_cast<T>(0.00424609), static_cast<T>(0.00431238),
        static_cast<T>(0.00082158), static_cast<T>(0.00296093),
        static_cast<T>(0.0041239), static_cast<T>(0.00424609),
        static_cast<T>(-0.00164846), static_cast<T>(0.00082158)
    );

    return Phi;
}

template <typename T>
auto get_solver_factor(void) -> DenseMatrix_Type<T, INPUT_SIZE* Nc, Np* OUTPUT_SIZE> {

    auto solver_factor = make_DenseMatrix<INPUT_SIZE* Nc, Np* OUTPUT_SIZE>(
        static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(-0.61712582),
        static_cast<T>(0.0), static_cast<T>(-1.37253004), static_cast<T>(0.01234246), static_cast<T>(-2.10681113),
        static_cast<T>(0.05151826), static_cast<T>(-2.68560539), static_cast<T>(0.13087129), static_cast<T>(-3.00344895),
        static_cast<T>(0.25996854), static_cast<T>(-2.99176747), static_cast<T>(0.44267964), static_cast<T>(-2.62637253),
        static_cast<T>(0.67609028), static_cast<T>(-1.93147658), static_cast<T>(0.95035764), static_cast<T>(-0.97846721),
        static_cast<T>(1.24954104), static_cast<T>(0.12123854), static_cast<T>(1.55333459), static_cast<T>(1.2289357),
        static_cast<T>(1.83951378), static_cast<T>(2.1957415), static_cast<T>(2.08680549), static_cast<T>(2.88211328),
        static_cast<T>(2.27781806), static_cast<T>(3.1773287), static_cast<T>(2.40163807), static_cast<T>(3.01631341),
        static_cast<T>(2.45572085), static_cast<T>(2.39138187), static_cast<T>(2.44677353), static_cast<T>(1.35698262),
        static_cast<T>(2.39044617), static_cast<T>(0.02635839), static_cast<T>(2.30979565), static_cast<T>(-1.43994868),
        static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(1.64113254e-01),
        static_cast<T>(0.0), static_cast<T>(-2.17342891e-01), static_cast<T>(-3.28224888e-03), static_cast<T>(-8.89763918e-01),
        static_cast<T>(-2.05354976e-03), static_cast<T>(-1.65948420e+00), static_cast<T>(1.69089053e-02), static_cast<T>(-2.36675556e+00),
        static_cast<T>(6.81127627e-02), static_cast<T>(-2.88035416e+00), static_cast<T>(1.64091318e-01), static_cast<T>(-3.10225715e+00),
        static_cast<T>(3.12877768e-01), static_cast<T>(-2.97542263e+00), static_cast<T>(5.16269769e-01), static_cast<T>(-2.49025719e+00),
        static_cast<T>(7.69000379e-01), static_cast<T>(-1.68699270e+00), static_cast<T>(1.05889942e+00), static_cast<T>(-6.52458525e-01),
        static_cast<T>(1.36804327e+00), static_cast<T>(4.89151155e-01), static_cast<T>(1.67477910e+00), static_cast<T>(1.59089796e+00),
        static_cast<T>(1.95639525e+00), static_cast<T>(2.50160503e+00), static_cast<T>(2.19211286e+00), static_cast<T>(3.08606322e+00),
        static_cast<T>(2.36601279e+00), static_cast<T>(3.24435143e+00), static_cast<T>(2.46949680e+00), static_cast<T>(2.92762204e+00),
        static_cast<T>(2.50291993e+00), static_cast<T>(2.14800028e+00), static_cast<T>(2.47611977e+00), static_cast<T>(9.80885766e-01)
    );

    return solver_factor;
}

} // namespace PythonMPC_ServoMotorData

#endif __TEST_MPC_SERVO_MOTOR_DATA_HPP__
