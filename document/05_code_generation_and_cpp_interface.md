# 5. コード生成と Python/C++ インターフェース対応

この章は `python_mpc/*.py` と `mpc_utility/*.py` の Deploy クラス、および `sample/*.cpp` の使い方対応をまとめます。

## 生成API一覧

| Python API | 用途 |
|---|---|
| `LinearMPC_Deploy.generate_LTI_MPC_NC_cpp_code(...)` | LTI 制約なし |
| `LinearMPC_Deploy.generate_LTI_MPC_cpp_code(...)` | LTI 制約あり |
| `LinearMPC_Deploy.generate_LTV_MPC_NC_cpp_code(...)` | LTV 制約なし |
| `LinearMPC_Deploy.generate_LTV_MPC_cpp_code(...)` | LTV 制約あり |
| `AdaptiveMPC_Deploy.generate_Adaptive_MPC_NC_cpp_code(...)` | Adaptive 制約なし |
| `AdaptiveMPC_Deploy.generate_Adaptive_MPC_cpp_code(...)` | Adaptive 制約あり |
| `NonlinearMPC_TwiceDifferentiableDeploy.generate_Nonlinear_MPC_cpp_code(...)` | Nonlinear SQP |
| `NonlinearMPC_OptimizationEngineDeploy.generate_Nonlinear_MPC_cpp_code(...)` | Nonlinear OpEn |

## 生成ファイルの考え方

Deployを呼ぶと、1ファイルだけでなく複数ヘッダが生成されます。

- EKF / LKF
- 予測行列（`F`, `Phi`）
- 制約・重み
- パラメータ構造体
- アップデータ（LTV/Adaptive）
- 最終統合ヘッダ（`xxx_lti_mpc.hpp` など）

最終統合ヘッダを C++ サンプルが `#include` して `::make()` を呼ぶ構成です。

## C++ 側の共通パターン

```cpp
#include "servo_motor_lti_mpc.hpp"
auto lti_mpc = servo_motor_lti_mpc::make();
servo_motor_lti_mpc::Reference_Type reference;
U = lti_mpc.update(reference, sys.get_Y());
```

### よく出る型

| 型 | 意味 |
|---|---|
| `xxx::type` | MPC本体型 |
| `xxx::Reference_Type` | 1時刻参照値 |
| `xxx::ReferenceTrajectory_Type` | ホライズン参照軌道（主に非線形） |
| `xxx::Parameter_Type` | パラメータ構造体 |
| `STATE_SIZE / INPUT_SIZE / OUTPUT_SIZE` | 次元定数 |

## Python実装とC++実装の対応表

| Python実装例 | C++実装例 |
|---|---|
| `sample/lti_no_constraints/servo_motor.py` | `sample/lti_no_constraints/servo_motor.cpp` |
| `sample/lti/servo_motor_constraints.py` | `sample/lti/servo_motor_constraints.cpp` |
| `sample/ltv_no_constraints/servo_motor_ltv.py` | `sample/ltv_no_constraints/servo_motor_ltv.cpp` |
| `sample/ltv/servo_motor_ltv_constraints.py` | `sample/ltv/servo_motor_ltv_constraints.cpp` |
| `sample/adaptive_no_constraints/two_wheel_vehicle_model.py` | `sample/adaptive_no_constraints/two_wheel_vehicle_model.cpp` |
| `sample/adaptive/two_wheel_vehicle_model_constraints.py` | `sample/adaptive/two_wheel_vehicle_model_constraints.cpp` |
| `sample/nonlinear/simple_pendulum.py` | `sample/nonlinear/simple_pendulum.cpp` |
| `sample/nonlinear/kinematic_bicycle_model.py` | `sample/nonlinear/kinematic_bicycle_model.cpp` |
| `sample/nonlinear/kinematic_bicycle_model_op_en.py` | `sample/nonlinear/kinematic_bicycle_model_op_en.cpp` |

## 実運用での注意

- C++サンプル先頭コメントの通り、**先に対応するPythonサンプルを実行**してヘッダを生成する
- LTV/Adaptive/Nonlinearは、必要に応じてパラメータや参照軌道CSVを同時に用意する
- 生成ヘッダはサンプルと同ディレクトリ運用が前提の実装が多い（`#include "xxx.hpp"`）
