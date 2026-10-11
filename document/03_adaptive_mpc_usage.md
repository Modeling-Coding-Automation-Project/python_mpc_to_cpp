# 3. 適応MPCの使い方（Adaptive MPC）

対象サンプル:

- `sample/adaptive_no_constraints/two_wheel_vehicle_model.py`
- `sample/adaptive/two_wheel_vehicle_model_constraints.py`
- `sample/sqrt_2_2_model/sqrt_2_2_model.py`

## 基本構成

Adaptive MPC では、線形モデル行列を直接渡す代わりに、次を渡します。

- 状態方程式 `fxu`
- 観測方程式 `hx`
- それぞれのヤコビアン
- パラメータ構造体（dataclass）

```python
ada_mpc = AdaptiveMPC_NoConstraints(
    delta_time=sim_delta_time,
    X=X, U=U, X_initial=X_initial,
    fxu=fxu, fxu_jacobian_X=fxu_jacobian_X, fxu_jacobian_U=fxu_jacobian_U,
    hx=hx, hx_jacobian=hx_jacobian,
    parameters_struct=controller_parameters,
    Np=Np, Nc=Nc,
    Weight_U=Weight_U, Weight_Y=Weight_Y,
    Q_kf=Q_ekf, R_kf=R_ekf,
    Number_of_Delay=Number_of_Delay
)
```

制約あり版は `AdaptiveMPC` を使い、`U_min` / `U_max` などを追加します。

## 制御ループ

サンプル共通で、制御入力更新は次です。

```python
u_from_mpc = ada_mpc.update_manipulation(reference, y_measured)
```

前段で `fxu_function` / `hx_function` を使ってプラントを更新し、計測値を作っています。

## `mpc_utility` が担う部分

Adaptiveのコード生成では `mpc_utility/adaptive_matrices_deploy.py` が使われ、
主に以下を生成します。

- Parameter構造体ヘッダ
- Embedded Integrator 更新器
- 予測行列更新器
- Adaptive MPC 用の Phi/F 更新器

これを `AdaptiveMPC_Deploy` が統合して最終MPCヘッダを作る構成です。

## Python/C++ の対応（Adaptive）

| Python | C++ |
|---|---|
| `AdaptiveMPC_Deploy.generate_Adaptive_MPC_NC_cpp_code(ada_mpc)` | `#include "xxx_ada_mpc.hpp"` |
| `u_from_mpc = ada_mpc.update_manipulation(reference, y)` | `U = ada_mpc_nc.update_manipulation(reference, Y)` |
| `parameters_struct` dataclass | `xxx_ekf_parameter::Parameter_Type` |
| 状態方程式関数呼び出し | `xxx_ekf_state_equation::function(X, U, parameters)` |

実装例は `sample/adaptive_no_constraints/two_wheel_vehicle_model.cpp` と
`sample/adaptive/two_wheel_vehicle_model_constraints.cpp` です。
