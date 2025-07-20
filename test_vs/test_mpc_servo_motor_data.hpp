#ifndef __TEST_MPC_SERVO_MOTOR_DATA_HPP__
#define __TEST_MPC_SERVO_MOTOR_DATA_HPP__

#include "python_mpc.hpp"

using namespace PythonNumpy;

namespace PythonMPC_ServoMotorData {

static constexpr std::size_t NP = 20;
static constexpr std::size_t NC = 2;

static constexpr std::size_t INPUT_SIZE = 1;
static constexpr std::size_t STATE_SIZE = 4;
static constexpr std::size_t OUTPUT_SIZE = 2;

static constexpr std::size_t AUGMENTED_STATE_SIZE = STATE_SIZE + OUTPUT_SIZE;

template <typename T>
auto get_F(void)
    -> DenseMatrix_Type<T, NP * OUTPUT_SIZE, AUGMENTED_STATE_SIZE> {

  auto F = make_DenseMatrix<NP * OUTPUT_SIZE, AUGMENTED_STATE_SIZE>(
      static_cast<T>(1.0), static_cast<T>(0.05), static_cast<T>(0.0),
      static_cast<T>(0.0), static_cast<T>(1.0), static_cast<T>(0.0),
      static_cast<T>(6.40099503e+00), static_cast<T>(3.20049752e-01),
      static_cast<T>(-3.20049752e-01), static_cast<T>(-1.60024876e-02),
      static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
      static_cast<T>(1.87198073), static_cast<T>(0.14750001),
      static_cast<T>(0.00640096), static_cast<T>(0.0), static_cast<T>(1.0),
      static_cast<T>(0.0), static_cast<T>(1.18801075e+01),
      static_cast<T>(9.44146846e-01), static_cast<T>(-5.94005376e-01),
      static_cast<T>(-3.98461941e-02), static_cast<T>(0.00000000e+00),
      static_cast<T>(5.00000000e-03), static_cast<T>(2.49432386e+00),
      static_cast<T>(2.83724085e-01), static_cast<T>(2.52838072e-02),
      static_cast<T>(3.20048173e-04), static_cast<T>(1.00000000e+00),
      static_cast<T>(0.00000000e+00), static_cast<T>(1.56086675e+01),
      static_cast<T>(1.81099486e+00), static_cast<T>(-7.80433373e-01),
      static_cast<T>(-6.52273915e-02), static_cast<T>(0.00000000e+00),
      static_cast<T>(5.00000000e-03), static_cast<T>(2.76992948e+00),
      static_cast<T>(4.44254143e-01), static_cast<T>(6.15035259e-02),
      static_cast<T>(1.42101397e-03), static_cast<T>(1.00000000e+00),
      static_cast<T>(0.00000000e+00), static_cast<T>(1.69552975e+01),
      static_cast<T>(2.82092869e+00), static_cast<T>(-8.47764876e-01),
      static_cast<T>(-8.69855781e-02), static_cast<T>(0.00000000e+00),
      static_cast<T>(5.00000000e-03), static_cast<T>(2.64156357),
      static_cast<T>(0.61053802), static_cast<T>(0.11792182),
      static_cast<T>(0.00377147), static_cast<T>(1.0), static_cast<T>(0.0),
      static_cast<T>(1.55768337e+01), static_cast<T>(3.84769758e+00),
      static_cast<T>(-7.78841685e-01), static_cast<T>(-1.01013665e-01),
      static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
      static_cast<T>(2.10249213), static_cast<T>(0.76208945),
      static_cast<T>(0.19487539), static_cast<T>(0.00774411),
      static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(1.14796521e+01),
      static_cast<T>(4.75420509e+00), static_cast<T>(-5.73982606e-01),
      static_cast<T>(-1.04441267e-01), static_cast<T>(0.00000000e+00),
      static_cast<T>(5.00000000e-03), static_cast<T>(1.20081947),
      static_cast<T>(0.87910977), static_cast<T>(0.28995903),
      static_cast<T>(0.01353839), static_cast<T>(1.0), static_cast<T>(0.0),
      static_cast<T>(5.03952190e+00), static_cast<T>(5.41052836e+00),
      static_cast<T>(-2.51976095e-01), static_cast<T>(-9.58778389e-02),
      static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
      static_cast<T>(0.0366188), static_cast<T>(0.94519547),
      static_cast<T>(0.39816906), static_cast<T>(0.02113176),
      static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(-3.02623438e+00),
      static_cast<T>(5.71202913e+00), static_cast<T>(1.51311719e-01),
      static_cast<T>(-7.55814334e-02), static_cast<T>(0.00000000e+00),
      static_cast<T>(5.00000000e-03), static_cast<T>(-1.24818158),
      static_cast<T>(0.94976687), static_cast<T>(0.51240908),
      static_cast<T>(0.03026302), static_cast<T>(1.0), static_cast<T>(0.0),
      static_cast<T>(-1.17340316e+01), static_cast<T>(5.59516711e+00),
      static_cast<T>(5.86701581e-01), static_cast<T>(-4.54718040e-02),
      static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
      static_cast<T>(-2.48623738), static_cast<T>(0.88986968),
      static_cast<T>(0.62431187), static_cast<T>(0.04044933),
      static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(-1.99498855e+01),
      static_cast<T>(5.04875831e+00), static_cast<T>(9.97494273e-01),
      static_cast<T>(-8.94859249e-03), static_cast<T>(0.00000000e+00),
      static_cast<T>(5.00000000e-03), static_cast<T>(-3.50573074),
      static_cast<T>(0.77106455), static_cast<T>(0.72528654),
      static_cast<T>(0.05103577), static_cast<T>(1.0), static_cast<T>(0.0),
      static_cast<T>(-2.65329373e+01), static_cast<T>(4.11887711e+00),
      static_cast<T>(1.32664687e+00), static_cast<T>(2.94874158e-02),
      static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
      static_cast<T>(-4.15327346), static_cast<T>(0.60722498),
      static_cast<T>(0.80766367), static_cast<T>(0.06127185),
      static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(-3.04891062e+01),
      static_cast<T>(2.90633716e+00), static_cast<T>(1.52445531e+00),
      static_cast<T>(6.47786894e-02), static_cast<T>(0.00000000e+00),
      static_cast<T>(5.00000000e-03), static_cast<T>(-4.31580259),
      static_cast<T>(0.4192002), static_cast<T>(0.86579013),
      static_cast<T>(0.07040639), static_cast<T>(1.0), static_cast<T>(0.0),
      static_cast<T>(-3.11148063e+01), static_cast<T>(1.55661546e+00),
      static_cast<T>(1.55574032e+00), static_cast<T>(9.19618358e-02),
      static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
      static_cast<T>(-3.9384457), static_cast<T>(0.23245017),
      static_cast<T>(0.89692229), static_cast<T>(0.07778864),
      static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(-2.81106995e+01),
      static_cast<T>(2.43094506e-01), static_cast<T>(1.40553497e+00),
      static_cast<T>(1.06845828e-01), static_cast<T>(0.00000000e+00),
      static_cast<T>(5.00000000e-03), static_cast<T>(-3.03568303),
      static_cast<T>(0.07390543), static_cast<T>(0.90178415),
      static_cast<T>(0.08296255), static_cast<T>(1.0), static_cast<T>(0.0),
      static_cast<T>(-2.16482005e+01), static_cast<T>(-8.54545383e-01),
      static_cast<T>(1.08241002e+00), static_cast<T>(1.06628717e-01),
      static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
      static_cast<T>(-1.69386656), static_cast<T>(-0.03157397),
      static_cast<T>(0.88469333), static_cast<T>(0.08574086),
      static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(-1.23767100e+01),
      static_cast<T>(-1.57417860e+00), static_cast<T>(6.18835502e-01),
      static_cast<T>(9.03660848e-02), static_cast<T>(0.00000000e+00),
      static_cast<T>(5.00000000e-03), static_cast<T>(-0.06419823),
      static_cast<T>(-0.06468861), static_cast<T>(0.85320991),
      static_cast<T>(0.08624769), static_cast<T>(1.0), static_cast<T>(0.0),
      static_cast<T>(-1.36677828), static_cast<T>(-1.79425581),
      static_cast<T>(0.06833891), static_cast<T>(0.05921867),
      static_cast<T>(0.0), static_cast<T>(0.005), static_cast<T>(1.65350056),
      static_cast<T>(-0.01466411), static_cast<T>(0.81732497),
      static_cast<T>(0.08492186), static_cast<T>(1.0), static_cast<T>(0.0),
      static_cast<T>(1.00072615e+01), static_cast<T>(-1.45283262e+00),
      static_cast<T>(-5.00363075e-01), static_cast<T>(1.64316060e-02),
      static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03),
      static_cast<T>(3.23463074), static_cast<T>(0.11874412),
      static_cast<T>(0.78826846), static_cast<T>(0.08247796),
      static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(2.02332466e+01),
      static_cast<T>(-5.59778521e-01), static_cast<T>(-1.01166233e+00),
      static_cast<T>(-3.29691544e-02), static_cast<T>(0.00000000e+00),
      static_cast<T>(5.00000000e-03), static_cast<T>(4.45854104),
      static_cast<T>(0.32453848), static_cast<T>(0.77707295),
      static_cast<T>(0.07982762), static_cast<T>(1.0), static_cast<T>(0.0),
      static_cast<T>(2.78564549e+01), static_cast<T>(7.99922346e-01),
      static_cast<T>(-1.39282275e+00), static_cast<T>(-8.27404896e-02),
      static_cast<T>(0.00000000e+00), static_cast<T>(5.00000000e-03));

  return F;
}

template <typename T>
auto get_Phi(void) -> DenseMatrix_Type<T, NP * OUTPUT_SIZE, INPUT_SIZE * NC> {

  auto Phi = make_DenseMatrix<NP * OUTPUT_SIZE, INPUT_SIZE * NC>(
      static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0),
      static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0),
      static_cast<T>(-0.00080012), static_cast<T>(0.0), static_cast<T>(0.0),
      static_cast<T>(0.0), static_cast<T>(-0.00199231),
      static_cast<T>(-0.00080012), static_cast<T>(1.60024087e-05),
      static_cast<T>(0.00000000e+00), static_cast<T>(-0.00326137),
      static_cast<T>(-0.00199231), static_cast<T>(7.10506984e-05),
      static_cast<T>(1.60024087e-05), static_cast<T>(-0.00434928),
      static_cast<T>(-0.00326137), static_cast<T>(1.88573657e-04),
      static_cast<T>(7.10506984e-05), static_cast<T>(-0.00505068),
      static_cast<T>(-0.00434928), static_cast<T>(0.00038721),
      static_cast<T>(0.00018857), static_cast<T>(-0.00522206),
      static_cast<T>(-0.00505068), static_cast<T>(0.00067692),
      static_cast<T>(0.00038721), static_cast<T>(-0.00479389),
      static_cast<T>(-0.00522206), static_cast<T>(0.00105659),
      static_cast<T>(0.00067692), static_cast<T>(-0.00377907),
      static_cast<T>(-0.00479389), static_cast<T>(0.00151315),
      static_cast<T>(0.00105659), static_cast<T>(-0.00227359),
      static_cast<T>(-0.00377907), static_cast<T>(0.00202247),
      static_cast<T>(0.00151315), static_cast<T>(-0.00044743),
      static_cast<T>(-0.00227359), static_cast<T>(0.00255179),
      static_cast<T>(0.00202247), static_cast<T>(0.00147437),
      static_cast<T>(-0.00044743), static_cast<T>(0.00306359),
      static_cast<T>(0.00255179), static_cast<T>(0.00323893),
      static_cast<T>(0.00147437), static_cast<T>(0.00352032),
      static_cast<T>(0.00306359), static_cast<T>(0.00459809),
      static_cast<T>(0.00323893), static_cast<T>(0.00388943),
      static_cast<T>(0.00352032), static_cast<T>(0.00534229),
      static_cast<T>(0.00459809), static_cast<T>(0.00414813),
      static_cast<T>(0.00388943), static_cast<T>(0.00533144),
      static_cast<T>(0.00534229), static_cast<T>(0.00428704),
      static_cast<T>(0.00414813), static_cast<T>(0.0045183),
      static_cast<T>(0.00533144), static_cast<T>(0.00431238),
      static_cast<T>(0.00428704), static_cast<T>(0.00296093),
      static_cast<T>(0.0045183), static_cast<T>(0.00424609),
      static_cast<T>(0.00431238), static_cast<T>(0.00082158),
      static_cast<T>(0.00296093), static_cast<T>(0.0041239),
      static_cast<T>(0.00424609), static_cast<T>(-0.00164846),
      static_cast<T>(0.00082158));

  return Phi;
}

template <typename T>
auto get_solver_factor(void)
    -> DenseMatrix_Type<T, INPUT_SIZE * NC, NP * OUTPUT_SIZE> {

  auto solver_factor = make_DenseMatrix<INPUT_SIZE * NC, NP * OUTPUT_SIZE>(
      static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0),
      static_cast<T>(-0.61712582), static_cast<T>(0.0),
      static_cast<T>(-1.37253004), static_cast<T>(0.01234246),
      static_cast<T>(-2.10681113), static_cast<T>(0.05151826),
      static_cast<T>(-2.68560539), static_cast<T>(0.13087129),
      static_cast<T>(-3.00344895), static_cast<T>(0.25996854),
      static_cast<T>(-2.99176747), static_cast<T>(0.44267964),
      static_cast<T>(-2.62637253), static_cast<T>(0.67609028),
      static_cast<T>(-1.93147658), static_cast<T>(0.95035764),
      static_cast<T>(-0.97846721), static_cast<T>(1.24954104),
      static_cast<T>(0.12123854), static_cast<T>(1.55333459),
      static_cast<T>(1.2289357), static_cast<T>(1.83951378),
      static_cast<T>(2.1957415), static_cast<T>(2.08680549),
      static_cast<T>(2.88211328), static_cast<T>(2.27781806),
      static_cast<T>(3.1773287), static_cast<T>(2.40163807),
      static_cast<T>(3.01631341), static_cast<T>(2.45572085),
      static_cast<T>(2.39138187), static_cast<T>(2.44677353),
      static_cast<T>(1.35698262), static_cast<T>(2.39044617),
      static_cast<T>(0.02635839), static_cast<T>(2.30979565),
      static_cast<T>(-1.43994868), static_cast<T>(0.0), static_cast<T>(0.0),
      static_cast<T>(0.0), static_cast<T>(1.64113254e-01), static_cast<T>(0.0),
      static_cast<T>(-2.17342891e-01), static_cast<T>(-3.28224888e-03),
      static_cast<T>(-8.89763918e-01), static_cast<T>(-2.05354976e-03),
      static_cast<T>(-1.65948420e+00), static_cast<T>(1.69089053e-02),
      static_cast<T>(-2.36675556e+00), static_cast<T>(6.81127627e-02),
      static_cast<T>(-2.88035416e+00), static_cast<T>(1.64091318e-01),
      static_cast<T>(-3.10225715e+00), static_cast<T>(3.12877768e-01),
      static_cast<T>(-2.97542263e+00), static_cast<T>(5.16269769e-01),
      static_cast<T>(-2.49025719e+00), static_cast<T>(7.69000379e-01),
      static_cast<T>(-1.68699270e+00), static_cast<T>(1.05889942e+00),
      static_cast<T>(-6.52458525e-01), static_cast<T>(1.36804327e+00),
      static_cast<T>(4.89151155e-01), static_cast<T>(1.67477910e+00),
      static_cast<T>(1.59089796e+00), static_cast<T>(1.95639525e+00),
      static_cast<T>(2.50160503e+00), static_cast<T>(2.19211286e+00),
      static_cast<T>(3.08606322e+00), static_cast<T>(2.36601279e+00),
      static_cast<T>(3.24435143e+00), static_cast<T>(2.46949680e+00),
      static_cast<T>(2.92762204e+00), static_cast<T>(2.50291993e+00),
      static_cast<T>(2.14800028e+00), static_cast<T>(2.47611977e+00),
      static_cast<T>(9.80885766e-01));

  return solver_factor;
}

template <typename T>
class Parameter {
public:
  double Lshaft = static_cast<T>(1.0);
  double dshaft = static_cast<T>(0.02);
  double shaftrho = static_cast<T>(7850.0);
  double G = static_cast<T>(81500000000.0);
  double Mmotor = static_cast<T>(100.0);
  double Rmotor = static_cast<T>(0.1);
  double Bmotor = static_cast<T>(0.1);
  double R = static_cast<T>(20.0);
  double Kt = static_cast<T>(10.0);
  double Bload = static_cast<T>(25.0);
};

template <typename T>
using Parameter_Type = Parameter<T>;

namespace mpc_state_space_updater {

template <typename T, typename A_Updater_Output_Type>
class A_Updater {
public:
  static inline auto update(double Lshaft, double dshaft, double shaftrho,
                            double G, double Mmotor, double Rmotor,
                            double Bmotor, double R, double Kt, double Bload)
      -> A_Updater_Output_Type {

    return A_Updater::sympy_function(shaftrho, G, Bload, Kt, Bmotor, Rmotor,
                                     Mmotor, dshaft, Lshaft, R);
  }

  static inline auto sympy_function(double shaftrho, double G, double Bload,
                                    double Kt, double Bmotor, double Rmotor,
                                    double Mmotor, double dshaft, double Lshaft,
                                    double R) -> A_Updater_Output_Type {
    A_Updater_Output_Type result;

    double x0 = Rmotor * Rmotor;

    double x1 = Mmotor * x0;

    double x2 = dshaft * dshaft * dshaft * dshaft;

    double x3 = Lshaft * shaftrho * x2;

    double x4 = 1 / (25.0 * x1 + 0.098174770424681 * x3);

    double x5 = G * x2 / Lshaft;

    double x6 = 0.00490873852123405 * x5;

    double x7 = 1 / (Mmotor * x0);

    double x8 = x5 * x7;

    double x9 = 0.1 * x7;

    result.template set<0, 0>(static_cast<T>(1));
    result.template set<0, 1>(static_cast<T>(0.05));
    result.template set<0, 2>(static_cast<T>(0));
    result.template set<0, 3>(static_cast<T>(0));
    result.template set<1, 0>(static_cast<T>(-x4 * x6));
    result.template set<1, 1>(static_cast<T>(-0.05 * Bload * x4 + 1));
    result.template set<1, 2>(
        static_cast<T>(x6 / (500.0 * x1 + 1.96349540849362 * x3)));
    result.template set<1, 3>(static_cast<T>(0));
    result.template set<2, 0>(static_cast<T>(0));
    result.template set<2, 1>(static_cast<T>(0));
    result.template set<2, 2>(static_cast<T>(1));
    result.template set<2, 3>(static_cast<T>(0.05));
    result.template set<3, 0>(static_cast<T>(0.000490873852123405 * x8));
    result.template set<3, 1>(static_cast<T>(0));
    result.template set<3, 2>(static_cast<T>(-2.45436926061703e-05 * x8));
    result.template set<3, 3>(
        static_cast<T>(-Bmotor * x9 - Kt * Kt * x9 / R + 1));

    return result;
  }
};

template <typename T, typename B_Updater_Output_Type>
class B_Updater {
public:
  static inline auto update(double Lshaft, double dshaft, double shaftrho,
                            double G, double Mmotor, double Rmotor,
                            double Bmotor, double R, double Kt, double Bload)
      -> B_Updater_Output_Type {
    static_cast<void>(Lshaft);
    static_cast<void>(dshaft);
    static_cast<void>(shaftrho);
    static_cast<void>(G);
    static_cast<void>(Bmotor);
    static_cast<void>(Bload);

    return B_Updater::sympy_function(Kt, Rmotor, R, Mmotor);
  }

  static inline auto sympy_function(double Kt, double Rmotor, double R,
                                    double Mmotor) -> B_Updater_Output_Type {
    B_Updater_Output_Type result;

    result.template set<0, 0>(static_cast<T>(0));
    result.template set<1, 0>(static_cast<T>(0));
    result.template set<2, 0>(static_cast<T>(0));
    result.template set<3, 0>(
        static_cast<T>(0.1 * Kt / (Mmotor * R * (Rmotor * Rmotor))));

    return result;
  }
};

template <typename T, typename C_Updater_Output_Type>
class C_Updater {
public:
  static inline auto update(double Lshaft, double dshaft, double shaftrho,
                            double G, double Mmotor, double Rmotor,
                            double Bmotor, double R, double Kt, double Bload)
      -> C_Updater_Output_Type {
    static_cast<void>(shaftrho);
    static_cast<void>(Mmotor);
    static_cast<void>(Rmotor);
    static_cast<void>(Bmotor);
    static_cast<void>(R);
    static_cast<void>(Kt);
    static_cast<void>(Bload);

    return C_Updater::sympy_function(G, dshaft, Lshaft);
  }

  static inline auto sympy_function(double G, double dshaft, double Lshaft)
      -> C_Updater_Output_Type {
    C_Updater_Output_Type result;

    double x0 = G * (dshaft * dshaft * dshaft * dshaft) / Lshaft;

    result.template set<0, 0>(static_cast<T>(1.0));
    result.template set<0, 1>(static_cast<T>(0.0));
    result.template set<0, 2>(static_cast<T>(0.0));
    result.template set<0, 3>(static_cast<T>(0.0));
    result.template set<1, 0>(static_cast<T>(0.098174770424681 * x0));
    result.template set<1, 1>(static_cast<T>(0.0));
    result.template set<1, 2>(static_cast<T>(-0.00490873852123405 * x0));
    result.template set<1, 3>(static_cast<T>(0.0));

    return result;
  }
};

class MPC_StateSpace_Updater {
public:
  template <typename Parameter_Type,
            typename MPC_StateSpace_Updater_Output_Type>
  static inline void update(const Parameter_Type &parameter,
                            MPC_StateSpace_Updater_Output_Type &output) {
    double Lshaft = parameter.Lshaft;
    double dshaft = parameter.dshaft;
    double shaftrho = parameter.shaftrho;
    double G = parameter.G;
    double Mmotor = parameter.Mmotor;
    double Rmotor = parameter.Rmotor;
    double Bmotor = parameter.Bmotor;
    double R = parameter.R;
    double Kt = parameter.Kt;
    double Bload = parameter.Bload;

    using T = typename MPC_StateSpace_Updater_Output_Type::A_Type::Value_Type;

    auto A =
        A_Updater<T, typename MPC_StateSpace_Updater_Output_Type::A_Type>::update(
            Lshaft, dshaft, shaftrho, G, Mmotor, Rmotor, Bmotor, R, Kt, Bload);

    auto B =
        B_Updater<T, typename MPC_StateSpace_Updater_Output_Type::B_Type>::update(
            Lshaft, dshaft, shaftrho, G, Mmotor, Rmotor, Bmotor, R, Kt, Bload);

    auto C =
        C_Updater<T, typename MPC_StateSpace_Updater_Output_Type::C_Type>::update(
            Lshaft, dshaft, shaftrho, G, Mmotor, Rmotor, Bmotor, R, Kt, Bload);

    output.A = A;
    output.B = B;
    output.C = C;
  }
};

} // namespace mpc_state_space_updater

namespace mpc_embedded_integrator_state_space_updater {

    template <typename T, typename A_Updater_Output_Type>
    class A_Updater {
    public:
        static inline auto update(double Lshaft, double dshaft, double shaftrho,
            double G, double Mmotor, double Rmotor,
            double Bmotor, double R, double Kt, double Bload)
            -> A_Updater_Output_Type {

            return A_Updater::sympy_function(Bmotor, shaftrho, G, Lshaft, R, Rmotor,
                dshaft, Bload, Kt, Mmotor);
        }

        static inline auto sympy_function(double Bmotor, double shaftrho, double G,
            double Lshaft, double R, double Rmotor,
            double dshaft, double Bload, double Kt,
            double Mmotor) -> A_Updater_Output_Type {
            A_Updater_Output_Type result;

            double x0 = Rmotor * Rmotor;

            double x1 = Mmotor * x0;

            double x2 = dshaft * dshaft * dshaft * dshaft;

            double x3 = 0.098174770424681 * x2;

            double x4 = Lshaft * shaftrho;

            double x5 = 1 / (25.0 * x1 + x3 * x4);

            double x6 = G / Lshaft;

            double x7 = x2 * x6;

            double x8 = 0.00490873852123405 * x7;

            double x9 = 1 / (Mmotor * x0);

            double x10 = x7 * x9;

            double x11 = 0.1 * x9;

            result.template set<0, 0>(static_cast<T>(1));
            result.template set<0, 1>(static_cast<T>(0.05));
            result.template set<0, 2>(static_cast<T>(0));
            result.template set<0, 3>(static_cast<T>(0));
            result.template set<0, 4>(static_cast<T>(0.0));
            result.template set<0, 5>(static_cast<T>(0.0));
            result.template set<1, 0>(static_cast<T>(-x5 * x8));
            result.template set<1, 1>(static_cast<T>(-0.05 * Bload * x5 + 1));
            result.template set<1, 2>(
                static_cast<T>(x8 / (500.0 * x1 + 1.96349540849362 * x2 * x4)));
            result.template set<1, 3>(static_cast<T>(0));
            result.template set<1, 4>(static_cast<T>(0.0));
            result.template set<1, 5>(static_cast<T>(0.0));
            result.template set<2, 0>(static_cast<T>(0));
            result.template set<2, 1>(static_cast<T>(0));
            result.template set<2, 2>(static_cast<T>(1));
            result.template set<2, 3>(static_cast<T>(0.05));
            result.template set<2, 4>(static_cast<T>(0.0));
            result.template set<2, 5>(static_cast<T>(0.0));
            result.template set<3, 0>(static_cast<T>(0.000490873852123405 * x10));
            result.template set<3, 1>(static_cast<T>(0));
            result.template set<3, 2>(static_cast<T>(-2.45436926061703e-05 * x10));
            result.template set<3, 3>(
                static_cast<T>(-Bmotor * x11 - Kt * Kt * x11 / R + 1));
            result.template set<3, 4>(static_cast<T>(0.0));
            result.template set<3, 5>(static_cast<T>(0.0));
            result.template set<4, 0>(static_cast<T>(1.0));
            result.template set<4, 1>(static_cast<T>(0.05));
            result.template set<4, 2>(static_cast<T>(0));
            result.template set<4, 3>(static_cast<T>(0));
            result.template set<4, 4>(static_cast<T>(1));
            result.template set<4, 5>(static_cast<T>(0));
            result.template set<5, 0>(static_cast<T>(x3 * x6));
            result.template set<5, 1>(static_cast<T>(x8));
            result.template set<5, 2>(static_cast<T>(-x8));
            result.template set<5, 3>(static_cast<T>(-0.000245436926061703 * x7));
            result.template set<5, 4>(static_cast<T>(0));
            result.template set<5, 5>(static_cast<T>(1));

            return result;
        }
    };

    template <typename T, typename B_Updater_Output_Type>
    class B_Updater {
    public:
        static inline auto update(double Lshaft, double dshaft, double shaftrho,
            double G, double Mmotor, double Rmotor,
            double Bmotor, double R, double Kt, double Bload)
            -> B_Updater_Output_Type {
            static_cast<void>(Lshaft);
            static_cast<void>(dshaft);
            static_cast<void>(shaftrho);
            static_cast<void>(G);
            static_cast<void>(Bmotor);
            static_cast<void>(Bload);

            return B_Updater::sympy_function(Rmotor, R, Mmotor, Kt);
        }

        static inline auto sympy_function(double Rmotor, double R, double Mmotor,
            double Kt) -> B_Updater_Output_Type {
            B_Updater_Output_Type result;

            result.template set<0, 0>(static_cast<T>(0));
            result.template set<1, 0>(static_cast<T>(0));
            result.template set<2, 0>(static_cast<T>(0));
            result.template set<3, 0>(
                static_cast<T>(0.1 * Kt / (Mmotor * R * (Rmotor * Rmotor))));
            result.template set<4, 0>(static_cast<T>(0));
            result.template set<5, 0>(static_cast<T>(0));

            return result;
        }
    };

    template <typename T, typename C_Updater_Output_Type>
    class C_Updater {
    public:
        static inline auto update(double Lshaft, double dshaft, double shaftrho,
            double G, double Mmotor, double Rmotor,
            double Bmotor, double R, double Kt, double Bload)
            -> C_Updater_Output_Type {
            static_cast<void>(Lshaft);
            static_cast<void>(dshaft);
            static_cast<void>(shaftrho);
            static_cast<void>(G);
            static_cast<void>(Mmotor);
            static_cast<void>(Rmotor);
            static_cast<void>(Bmotor);
            static_cast<void>(R);
            static_cast<void>(Kt);
            static_cast<void>(Bload);

            return C_Updater::sympy_function();
        }

        static inline auto sympy_function() -> C_Updater_Output_Type {
            C_Updater_Output_Type result;

            result.template set<0, 0>(static_cast<T>(0));
            result.template set<0, 1>(static_cast<T>(0));
            result.template set<0, 2>(static_cast<T>(0));
            result.template set<0, 3>(static_cast<T>(0));
            result.template set<0, 4>(static_cast<T>(1.0));
            result.template set<0, 5>(static_cast<T>(0));
            result.template set<1, 0>(static_cast<T>(0));
            result.template set<1, 1>(static_cast<T>(0));
            result.template set<1, 2>(static_cast<T>(0));
            result.template set<1, 3>(static_cast<T>(0));
            result.template set<1, 4>(static_cast<T>(0));
            result.template set<1, 5>(static_cast<T>(0.005));

            return result;
        }
    };

    class EmbeddedIntegrator_Updater {
    public:
        template <typename Parameter_Type,
            typename EmbeddedIntegrator_Updater_Output_Type>
        static inline void update(const Parameter_Type& parameter,
            EmbeddedIntegrator_Updater_Output_Type& output) {
            double Lshaft = parameter.Lshaft;
            double dshaft = parameter.dshaft;
            double shaftrho = parameter.shaftrho;
            double G = parameter.G;
            double Mmotor = parameter.Mmotor;
            double Rmotor = parameter.Rmotor;
            double Bmotor = parameter.Bmotor;
            double R = parameter.R;
            double Kt = parameter.Kt;
            double Bload = parameter.Bload;

            using T = typename EmbeddedIntegrator_Updater_Output_Type::A_Type::Value_Type;

            auto A = A_Updater<T, typename EmbeddedIntegrator_Updater_Output_Type::A_Type>::update(
                Lshaft, dshaft, shaftrho, G, Mmotor, Rmotor, Bmotor, R, Kt, Bload);

            auto B = B_Updater<T, typename EmbeddedIntegrator_Updater_Output_Type::B_Type>::update(
                Lshaft, dshaft, shaftrho, G, Mmotor, Rmotor, Bmotor, R, Kt, Bload);

            auto C = C_Updater<T, typename EmbeddedIntegrator_Updater_Output_Type::C_Type>::update(
                Lshaft, dshaft, shaftrho, G, Mmotor, Rmotor, Bmotor, R, Kt, Bload);

            output.A = A;
            output.B = B;
            output.C = C;
        }
    };

} // namespace mpc_embedded_integrator_state_space_updater

namespace prediction_matrices_phi_f_updater {

    class PredictionMatricesPhiF_Updater {
    public:
        template <typename A_Type, typename B_Type, typename C_Type,
            typename Phi_Type, typename F_Type>
        static inline void update(const A_Type& A, const B_Type& B, const C_Type& C,
            Phi_Type& Phi, F_Type& F) {
            auto C_A_1 = C * A;
            auto C_A_2 = C_A_1 * A;
            auto C_A_3 = C_A_2 * A;
            auto C_A_4 = C_A_3 * A;
            auto C_A_5 = C_A_4 * A;
            auto C_A_6 = C_A_5 * A;
            auto C_A_7 = C_A_6 * A;
            auto C_A_8 = C_A_7 * A;
            auto C_A_9 = C_A_8 * A;
            auto C_A_10 = C_A_9 * A;
            auto C_A_11 = C_A_10 * A;
            auto C_A_12 = C_A_11 * A;
            auto C_A_13 = C_A_12 * A;
            auto C_A_14 = C_A_13 * A;
            auto C_A_15 = C_A_14 * A;
            auto C_A_16 = C_A_15 * A;
            auto C_A_17 = C_A_16 * A;
            auto C_A_18 = C_A_17 * A;
            auto C_A_19 = C_A_18 * A;
            auto C_A_20 = C_A_19 * A;
            auto C_A_0_B = C * B;
            auto C_A_1_B = C_A_1 * B;
            auto C_A_2_B = C_A_2 * B;
            auto C_A_3_B = C_A_3 * B;
            auto C_A_4_B = C_A_4 * B;
            auto C_A_5_B = C_A_5 * B;
            auto C_A_6_B = C_A_6 * B;
            auto C_A_7_B = C_A_7 * B;
            auto C_A_8_B = C_A_8 * B;
            auto C_A_9_B = C_A_9 * B;
            auto C_A_10_B = C_A_10 * B;
            auto C_A_11_B = C_A_11 * B;
            auto C_A_12_B = C_A_12 * B;
            auto C_A_13_B = C_A_13 * B;
            auto C_A_14_B = C_A_14 * B;
            auto C_A_15_B = C_A_15 * B;
            auto C_A_16_B = C_A_16 * B;
            auto C_A_17_B = C_A_17 * B;
            auto C_A_18_B = C_A_18 * B;
            auto C_A_19_B = C_A_19 * B;
            Phi.template set<0, 0>(C_A_0_B.template get<0, 0>());
            Phi.template set<1, 0>(C_A_0_B.template get<1, 0>());
            Phi.template set<2, 0>(C_A_1_B.template get<0, 0>());
            Phi.template set<3, 0>(C_A_1_B.template get<1, 0>());
            Phi.template set<2, 1>(C_A_0_B.template get<0, 0>());
            Phi.template set<3, 1>(C_A_0_B.template get<1, 0>());
            Phi.template set<4, 0>(C_A_2_B.template get<0, 0>());
            Phi.template set<5, 0>(C_A_2_B.template get<1, 0>());
            Phi.template set<4, 1>(C_A_1_B.template get<0, 0>());
            Phi.template set<5, 1>(C_A_1_B.template get<1, 0>());
            Phi.template set<6, 0>(C_A_3_B.template get<0, 0>());
            Phi.template set<7, 0>(C_A_3_B.template get<1, 0>());
            Phi.template set<6, 1>(C_A_2_B.template get<0, 0>());
            Phi.template set<7, 1>(C_A_2_B.template get<1, 0>());
            Phi.template set<8, 0>(C_A_4_B.template get<0, 0>());
            Phi.template set<9, 0>(C_A_4_B.template get<1, 0>());
            Phi.template set<8, 1>(C_A_3_B.template get<0, 0>());
            Phi.template set<9, 1>(C_A_3_B.template get<1, 0>());
            Phi.template set<10, 0>(C_A_5_B.template get<0, 0>());
            Phi.template set<11, 0>(C_A_5_B.template get<1, 0>());
            Phi.template set<10, 1>(C_A_4_B.template get<0, 0>());
            Phi.template set<11, 1>(C_A_4_B.template get<1, 0>());
            Phi.template set<12, 0>(C_A_6_B.template get<0, 0>());
            Phi.template set<13, 0>(C_A_6_B.template get<1, 0>());
            Phi.template set<12, 1>(C_A_5_B.template get<0, 0>());
            Phi.template set<13, 1>(C_A_5_B.template get<1, 0>());
            Phi.template set<14, 0>(C_A_7_B.template get<0, 0>());
            Phi.template set<15, 0>(C_A_7_B.template get<1, 0>());
            Phi.template set<14, 1>(C_A_6_B.template get<0, 0>());
            Phi.template set<15, 1>(C_A_6_B.template get<1, 0>());
            Phi.template set<16, 0>(C_A_8_B.template get<0, 0>());
            Phi.template set<17, 0>(C_A_8_B.template get<1, 0>());
            Phi.template set<16, 1>(C_A_7_B.template get<0, 0>());
            Phi.template set<17, 1>(C_A_7_B.template get<1, 0>());
            Phi.template set<18, 0>(C_A_9_B.template get<0, 0>());
            Phi.template set<19, 0>(C_A_9_B.template get<1, 0>());
            Phi.template set<18, 1>(C_A_8_B.template get<0, 0>());
            Phi.template set<19, 1>(C_A_8_B.template get<1, 0>());
            Phi.template set<20, 0>(C_A_10_B.template get<0, 0>());
            Phi.template set<21, 0>(C_A_10_B.template get<1, 0>());
            Phi.template set<20, 1>(C_A_9_B.template get<0, 0>());
            Phi.template set<21, 1>(C_A_9_B.template get<1, 0>());
            Phi.template set<22, 0>(C_A_11_B.template get<0, 0>());
            Phi.template set<23, 0>(C_A_11_B.template get<1, 0>());
            Phi.template set<22, 1>(C_A_10_B.template get<0, 0>());
            Phi.template set<23, 1>(C_A_10_B.template get<1, 0>());
            Phi.template set<24, 0>(C_A_12_B.template get<0, 0>());
            Phi.template set<25, 0>(C_A_12_B.template get<1, 0>());
            Phi.template set<24, 1>(C_A_11_B.template get<0, 0>());
            Phi.template set<25, 1>(C_A_11_B.template get<1, 0>());
            Phi.template set<26, 0>(C_A_13_B.template get<0, 0>());
            Phi.template set<27, 0>(C_A_13_B.template get<1, 0>());
            Phi.template set<26, 1>(C_A_12_B.template get<0, 0>());
            Phi.template set<27, 1>(C_A_12_B.template get<1, 0>());
            Phi.template set<28, 0>(C_A_14_B.template get<0, 0>());
            Phi.template set<29, 0>(C_A_14_B.template get<1, 0>());
            Phi.template set<28, 1>(C_A_13_B.template get<0, 0>());
            Phi.template set<29, 1>(C_A_13_B.template get<1, 0>());
            Phi.template set<30, 0>(C_A_15_B.template get<0, 0>());
            Phi.template set<31, 0>(C_A_15_B.template get<1, 0>());
            Phi.template set<30, 1>(C_A_14_B.template get<0, 0>());
            Phi.template set<31, 1>(C_A_14_B.template get<1, 0>());
            Phi.template set<32, 0>(C_A_16_B.template get<0, 0>());
            Phi.template set<33, 0>(C_A_16_B.template get<1, 0>());
            Phi.template set<32, 1>(C_A_15_B.template get<0, 0>());
            Phi.template set<33, 1>(C_A_15_B.template get<1, 0>());
            Phi.template set<34, 0>(C_A_17_B.template get<0, 0>());
            Phi.template set<35, 0>(C_A_17_B.template get<1, 0>());
            Phi.template set<34, 1>(C_A_16_B.template get<0, 0>());
            Phi.template set<35, 1>(C_A_16_B.template get<1, 0>());
            Phi.template set<36, 0>(C_A_18_B.template get<0, 0>());
            Phi.template set<37, 0>(C_A_18_B.template get<1, 0>());
            Phi.template set<36, 1>(C_A_17_B.template get<0, 0>());
            Phi.template set<37, 1>(C_A_17_B.template get<1, 0>());
            Phi.template set<38, 0>(C_A_19_B.template get<0, 0>());
            Phi.template set<39, 0>(C_A_19_B.template get<1, 0>());
            Phi.template set<38, 1>(C_A_18_B.template get<0, 0>());
            Phi.template set<39, 1>(C_A_18_B.template get<1, 0>());
            F.template set<0, 0>(C_A_1.template get<0, 0>());
            F.template set<0, 1>(C_A_1.template get<0, 1>());
            F.template set<0, 2>(C_A_1.template get<0, 2>());
            F.template set<0, 3>(C_A_1.template get<0, 3>());
            F.template set<0, 4>(C_A_1.template get<0, 4>());
            F.template set<0, 5>(C_A_1.template get<0, 5>());
            F.template set<1, 0>(C_A_1.template get<1, 0>());
            F.template set<1, 1>(C_A_1.template get<1, 1>());
            F.template set<1, 2>(C_A_1.template get<1, 2>());
            F.template set<1, 3>(C_A_1.template get<1, 3>());
            F.template set<1, 4>(C_A_1.template get<1, 4>());
            F.template set<1, 5>(C_A_1.template get<1, 5>());
            F.template set<2, 0>(C_A_2.template get<0, 0>());
            F.template set<2, 1>(C_A_2.template get<0, 1>());
            F.template set<2, 2>(C_A_2.template get<0, 2>());
            F.template set<2, 3>(C_A_2.template get<0, 3>());
            F.template set<2, 4>(C_A_2.template get<0, 4>());
            F.template set<2, 5>(C_A_2.template get<0, 5>());
            F.template set<3, 0>(C_A_2.template get<1, 0>());
            F.template set<3, 1>(C_A_2.template get<1, 1>());
            F.template set<3, 2>(C_A_2.template get<1, 2>());
            F.template set<3, 3>(C_A_2.template get<1, 3>());
            F.template set<3, 4>(C_A_2.template get<1, 4>());
            F.template set<3, 5>(C_A_2.template get<1, 5>());
            F.template set<4, 0>(C_A_3.template get<0, 0>());
            F.template set<4, 1>(C_A_3.template get<0, 1>());
            F.template set<4, 2>(C_A_3.template get<0, 2>());
            F.template set<4, 3>(C_A_3.template get<0, 3>());
            F.template set<4, 4>(C_A_3.template get<0, 4>());
            F.template set<4, 5>(C_A_3.template get<0, 5>());
            F.template set<5, 0>(C_A_3.template get<1, 0>());
            F.template set<5, 1>(C_A_3.template get<1, 1>());
            F.template set<5, 2>(C_A_3.template get<1, 2>());
            F.template set<5, 3>(C_A_3.template get<1, 3>());
            F.template set<5, 4>(C_A_3.template get<1, 4>());
            F.template set<5, 5>(C_A_3.template get<1, 5>());
            F.template set<6, 0>(C_A_4.template get<0, 0>());
            F.template set<6, 1>(C_A_4.template get<0, 1>());
            F.template set<6, 2>(C_A_4.template get<0, 2>());
            F.template set<6, 3>(C_A_4.template get<0, 3>());
            F.template set<6, 4>(C_A_4.template get<0, 4>());
            F.template set<6, 5>(C_A_4.template get<0, 5>());
            F.template set<7, 0>(C_A_4.template get<1, 0>());
            F.template set<7, 1>(C_A_4.template get<1, 1>());
            F.template set<7, 2>(C_A_4.template get<1, 2>());
            F.template set<7, 3>(C_A_4.template get<1, 3>());
            F.template set<7, 4>(C_A_4.template get<1, 4>());
            F.template set<7, 5>(C_A_4.template get<1, 5>());
            F.template set<8, 0>(C_A_5.template get<0, 0>());
            F.template set<8, 1>(C_A_5.template get<0, 1>());
            F.template set<8, 2>(C_A_5.template get<0, 2>());
            F.template set<8, 3>(C_A_5.template get<0, 3>());
            F.template set<8, 4>(C_A_5.template get<0, 4>());
            F.template set<8, 5>(C_A_5.template get<0, 5>());
            F.template set<9, 0>(C_A_5.template get<1, 0>());
            F.template set<9, 1>(C_A_5.template get<1, 1>());
            F.template set<9, 2>(C_A_5.template get<1, 2>());
            F.template set<9, 3>(C_A_5.template get<1, 3>());
            F.template set<9, 4>(C_A_5.template get<1, 4>());
            F.template set<9, 5>(C_A_5.template get<1, 5>());
            F.template set<10, 0>(C_A_6.template get<0, 0>());
            F.template set<10, 1>(C_A_6.template get<0, 1>());
            F.template set<10, 2>(C_A_6.template get<0, 2>());
            F.template set<10, 3>(C_A_6.template get<0, 3>());
            F.template set<10, 4>(C_A_6.template get<0, 4>());
            F.template set<10, 5>(C_A_6.template get<0, 5>());
            F.template set<11, 0>(C_A_6.template get<1, 0>());
            F.template set<11, 1>(C_A_6.template get<1, 1>());
            F.template set<11, 2>(C_A_6.template get<1, 2>());
            F.template set<11, 3>(C_A_6.template get<1, 3>());
            F.template set<11, 4>(C_A_6.template get<1, 4>());
            F.template set<11, 5>(C_A_6.template get<1, 5>());
            F.template set<12, 0>(C_A_7.template get<0, 0>());
            F.template set<12, 1>(C_A_7.template get<0, 1>());
            F.template set<12, 2>(C_A_7.template get<0, 2>());
            F.template set<12, 3>(C_A_7.template get<0, 3>());
            F.template set<12, 4>(C_A_7.template get<0, 4>());
            F.template set<12, 5>(C_A_7.template get<0, 5>());
            F.template set<13, 0>(C_A_7.template get<1, 0>());
            F.template set<13, 1>(C_A_7.template get<1, 1>());
            F.template set<13, 2>(C_A_7.template get<1, 2>());
            F.template set<13, 3>(C_A_7.template get<1, 3>());
            F.template set<13, 4>(C_A_7.template get<1, 4>());
            F.template set<13, 5>(C_A_7.template get<1, 5>());
            F.template set<14, 0>(C_A_8.template get<0, 0>());
            F.template set<14, 1>(C_A_8.template get<0, 1>());
            F.template set<14, 2>(C_A_8.template get<0, 2>());
            F.template set<14, 3>(C_A_8.template get<0, 3>());
            F.template set<14, 4>(C_A_8.template get<0, 4>());
            F.template set<14, 5>(C_A_8.template get<0, 5>());
            F.template set<15, 0>(C_A_8.template get<1, 0>());
            F.template set<15, 1>(C_A_8.template get<1, 1>());
            F.template set<15, 2>(C_A_8.template get<1, 2>());
            F.template set<15, 3>(C_A_8.template get<1, 3>());
            F.template set<15, 4>(C_A_8.template get<1, 4>());
            F.template set<15, 5>(C_A_8.template get<1, 5>());
            F.template set<16, 0>(C_A_9.template get<0, 0>());
            F.template set<16, 1>(C_A_9.template get<0, 1>());
            F.template set<16, 2>(C_A_9.template get<0, 2>());
            F.template set<16, 3>(C_A_9.template get<0, 3>());
            F.template set<16, 4>(C_A_9.template get<0, 4>());
            F.template set<16, 5>(C_A_9.template get<0, 5>());
            F.template set<17, 0>(C_A_9.template get<1, 0>());
            F.template set<17, 1>(C_A_9.template get<1, 1>());
            F.template set<17, 2>(C_A_9.template get<1, 2>());
            F.template set<17, 3>(C_A_9.template get<1, 3>());
            F.template set<17, 4>(C_A_9.template get<1, 4>());
            F.template set<17, 5>(C_A_9.template get<1, 5>());
            F.template set<18, 0>(C_A_10.template get<0, 0>());
            F.template set<18, 1>(C_A_10.template get<0, 1>());
            F.template set<18, 2>(C_A_10.template get<0, 2>());
            F.template set<18, 3>(C_A_10.template get<0, 3>());
            F.template set<18, 4>(C_A_10.template get<0, 4>());
            F.template set<18, 5>(C_A_10.template get<0, 5>());
            F.template set<19, 0>(C_A_10.template get<1, 0>());
            F.template set<19, 1>(C_A_10.template get<1, 1>());
            F.template set<19, 2>(C_A_10.template get<1, 2>());
            F.template set<19, 3>(C_A_10.template get<1, 3>());
            F.template set<19, 4>(C_A_10.template get<1, 4>());
            F.template set<19, 5>(C_A_10.template get<1, 5>());
            F.template set<20, 0>(C_A_11.template get<0, 0>());
            F.template set<20, 1>(C_A_11.template get<0, 1>());
            F.template set<20, 2>(C_A_11.template get<0, 2>());
            F.template set<20, 3>(C_A_11.template get<0, 3>());
            F.template set<20, 4>(C_A_11.template get<0, 4>());
            F.template set<20, 5>(C_A_11.template get<0, 5>());
            F.template set<21, 0>(C_A_11.template get<1, 0>());
            F.template set<21, 1>(C_A_11.template get<1, 1>());
            F.template set<21, 2>(C_A_11.template get<1, 2>());
            F.template set<21, 3>(C_A_11.template get<1, 3>());
            F.template set<21, 4>(C_A_11.template get<1, 4>());
            F.template set<21, 5>(C_A_11.template get<1, 5>());
            F.template set<22, 0>(C_A_12.template get<0, 0>());
            F.template set<22, 1>(C_A_12.template get<0, 1>());
            F.template set<22, 2>(C_A_12.template get<0, 2>());
            F.template set<22, 3>(C_A_12.template get<0, 3>());
            F.template set<22, 4>(C_A_12.template get<0, 4>());
            F.template set<22, 5>(C_A_12.template get<0, 5>());
            F.template set<23, 0>(C_A_12.template get<1, 0>());
            F.template set<23, 1>(C_A_12.template get<1, 1>());
            F.template set<23, 2>(C_A_12.template get<1, 2>());
            F.template set<23, 3>(C_A_12.template get<1, 3>());
            F.template set<23, 4>(C_A_12.template get<1, 4>());
            F.template set<23, 5>(C_A_12.template get<1, 5>());
            F.template set<24, 0>(C_A_13.template get<0, 0>());
            F.template set<24, 1>(C_A_13.template get<0, 1>());
            F.template set<24, 2>(C_A_13.template get<0, 2>());
            F.template set<24, 3>(C_A_13.template get<0, 3>());
            F.template set<24, 4>(C_A_13.template get<0, 4>());
            F.template set<24, 5>(C_A_13.template get<0, 5>());
            F.template set<25, 0>(C_A_13.template get<1, 0>());
            F.template set<25, 1>(C_A_13.template get<1, 1>());
            F.template set<25, 2>(C_A_13.template get<1, 2>());
            F.template set<25, 3>(C_A_13.template get<1, 3>());
            F.template set<25, 4>(C_A_13.template get<1, 4>());
            F.template set<25, 5>(C_A_13.template get<1, 5>());
            F.template set<26, 0>(C_A_14.template get<0, 0>());
            F.template set<26, 1>(C_A_14.template get<0, 1>());
            F.template set<26, 2>(C_A_14.template get<0, 2>());
            F.template set<26, 3>(C_A_14.template get<0, 3>());
            F.template set<26, 4>(C_A_14.template get<0, 4>());
            F.template set<26, 5>(C_A_14.template get<0, 5>());
            F.template set<27, 0>(C_A_14.template get<1, 0>());
            F.template set<27, 1>(C_A_14.template get<1, 1>());
            F.template set<27, 2>(C_A_14.template get<1, 2>());
            F.template set<27, 3>(C_A_14.template get<1, 3>());
            F.template set<27, 4>(C_A_14.template get<1, 4>());
            F.template set<27, 5>(C_A_14.template get<1, 5>());
            F.template set<28, 0>(C_A_15.template get<0, 0>());
            F.template set<28, 1>(C_A_15.template get<0, 1>());
            F.template set<28, 2>(C_A_15.template get<0, 2>());
            F.template set<28, 3>(C_A_15.template get<0, 3>());
            F.template set<28, 4>(C_A_15.template get<0, 4>());
            F.template set<28, 5>(C_A_15.template get<0, 5>());
            F.template set<29, 0>(C_A_15.template get<1, 0>());
            F.template set<29, 1>(C_A_15.template get<1, 1>());
            F.template set<29, 2>(C_A_15.template get<1, 2>());
            F.template set<29, 3>(C_A_15.template get<1, 3>());
            F.template set<29, 4>(C_A_15.template get<1, 4>());
            F.template set<29, 5>(C_A_15.template get<1, 5>());
            F.template set<30, 0>(C_A_16.template get<0, 0>());
            F.template set<30, 1>(C_A_16.template get<0, 1>());
            F.template set<30, 2>(C_A_16.template get<0, 2>());
            F.template set<30, 3>(C_A_16.template get<0, 3>());
            F.template set<30, 4>(C_A_16.template get<0, 4>());
            F.template set<30, 5>(C_A_16.template get<0, 5>());
            F.template set<31, 0>(C_A_16.template get<1, 0>());
            F.template set<31, 1>(C_A_16.template get<1, 1>());
            F.template set<31, 2>(C_A_16.template get<1, 2>());
            F.template set<31, 3>(C_A_16.template get<1, 3>());
            F.template set<31, 4>(C_A_16.template get<1, 4>());
            F.template set<31, 5>(C_A_16.template get<1, 5>());
            F.template set<32, 0>(C_A_17.template get<0, 0>());
            F.template set<32, 1>(C_A_17.template get<0, 1>());
            F.template set<32, 2>(C_A_17.template get<0, 2>());
            F.template set<32, 3>(C_A_17.template get<0, 3>());
            F.template set<32, 4>(C_A_17.template get<0, 4>());
            F.template set<32, 5>(C_A_17.template get<0, 5>());
            F.template set<33, 0>(C_A_17.template get<1, 0>());
            F.template set<33, 1>(C_A_17.template get<1, 1>());
            F.template set<33, 2>(C_A_17.template get<1, 2>());
            F.template set<33, 3>(C_A_17.template get<1, 3>());
            F.template set<33, 4>(C_A_17.template get<1, 4>());
            F.template set<33, 5>(C_A_17.template get<1, 5>());
            F.template set<34, 0>(C_A_18.template get<0, 0>());
            F.template set<34, 1>(C_A_18.template get<0, 1>());
            F.template set<34, 2>(C_A_18.template get<0, 2>());
            F.template set<34, 3>(C_A_18.template get<0, 3>());
            F.template set<34, 4>(C_A_18.template get<0, 4>());
            F.template set<34, 5>(C_A_18.template get<0, 5>());
            F.template set<35, 0>(C_A_18.template get<1, 0>());
            F.template set<35, 1>(C_A_18.template get<1, 1>());
            F.template set<35, 2>(C_A_18.template get<1, 2>());
            F.template set<35, 3>(C_A_18.template get<1, 3>());
            F.template set<35, 4>(C_A_18.template get<1, 4>());
            F.template set<35, 5>(C_A_18.template get<1, 5>());
            F.template set<36, 0>(C_A_19.template get<0, 0>());
            F.template set<36, 1>(C_A_19.template get<0, 1>());
            F.template set<36, 2>(C_A_19.template get<0, 2>());
            F.template set<36, 3>(C_A_19.template get<0, 3>());
            F.template set<36, 4>(C_A_19.template get<0, 4>());
            F.template set<36, 5>(C_A_19.template get<0, 5>());
            F.template set<37, 0>(C_A_19.template get<1, 0>());
            F.template set<37, 1>(C_A_19.template get<1, 1>());
            F.template set<37, 2>(C_A_19.template get<1, 2>());
            F.template set<37, 3>(C_A_19.template get<1, 3>());
            F.template set<37, 4>(C_A_19.template get<1, 4>());
            F.template set<37, 5>(C_A_19.template get<1, 5>());
            F.template set<38, 0>(C_A_20.template get<0, 0>());
            F.template set<38, 1>(C_A_20.template get<0, 1>());
            F.template set<38, 2>(C_A_20.template get<0, 2>());
            F.template set<38, 3>(C_A_20.template get<0, 3>());
            F.template set<38, 4>(C_A_20.template get<0, 4>());
            F.template set<38, 5>(C_A_20.template get<0, 5>());
            F.template set<39, 0>(C_A_20.template get<1, 0>());
            F.template set<39, 1>(C_A_20.template get<1, 1>());
            F.template set<39, 2>(C_A_20.template get<1, 2>());
            F.template set<39, 3>(C_A_20.template get<1, 3>());
            F.template set<39, 4>(C_A_20.template get<1, 4>());
            F.template set<39, 5>(C_A_20.template get<1, 5>());
        }
    };

} // namespace prediction_matrices_phi_f_updater

namespace ltv_mpc_phi_f_updater {

    using namespace mpc_embedded_integrator_state_space_updater;
    using namespace prediction_matrices_phi_f_updater;

    class LTV_MPC_Phi_F_Updater {
    public:
        template <typename StateSpace_Type, typename Parameter_Type,
            typename Phi_Type, typename F_Type>
        static inline void update(const Parameter_Type& parameter, Phi_Type& Phi,
            F_Type& F) {

            StateSpace_Type state_space;
            EmbeddedIntegrator_Updater::update(parameter, state_space);

            PredictionMatricesPhiF_Updater::update(state_space.A, state_space.B,
                state_space.C, Phi, F);
        }
    };

} // namespace ltv_mpc_phi_f_updater

} // namespace PythonMPC_ServoMotorData

#endif // __TEST_MPC_SERVO_MOTOR_DATA_HPP__
