# 4. 非線形MPCの使い方（SQP / OpEn）

対象サンプル:

- `sample/nonlinear/simple_pendulum.py`
- `sample/nonlinear/kinematic_bicycle_model.py`
- `sample/nonlinear/kinematic_bicycle_model_op_en.py`

## 2種類の非線形MPC

| 種類 | Pythonクラス | 主なサンプル |
|---|---|---|
| SQPベース | `NonlinearMPC_TwiceDifferentiable` | `simple_pendulum.py`, `kinematic_bicycle_model.py` |
| OpEn（PANOC/ALM）ベース | `NonlinearMPC_OptimizationEngine` | `kinematic_bicycle_model_op_en.py` |

## 共通の作業

1. SymPyで `fxu` / `hx` / `X` / `U` を定義
2. 重み・制約・EKF共分散を設定
3. `update_manipulation(reference, y_measured)` をループ実行
4. C++コード生成を呼ぶ

```python
deployed_file_names = NonlinearMPC_TwiceDifferentiableDeploy.generate_Nonlinear_MPC_cpp_code(
    nonlinear_mpc
)
```

または

```python
deployed_file_names = NonlinearMPC_OptimizationEngineDeploy.generate_Nonlinear_MPC_cpp_code(
    nonlinear_mpc
)
```

## 参照軌道の扱い

`kinematic_bicycle_model*.py` では、`reference_path.csv` を出力し、
将来ホライズン分の `reference_path (OUTPUT_SIZE x Np)` を毎ステップ構築しています。

このため C++側でも同じCSVを読み、`ReferenceTrajectory_Type` へ詰める流れになります。

## 反復回数設定・監視

非線形では反復回数の設定・監視を使っています。

- SQP版: `set_solver_max_iteration(5)` / `get_solver_step_iterated_number()`
- OpEn版: `set_solver_max_iteration(outer_max_iterations, inner_max_iterations)` /
  `get_solver_step_iterated_number()`（outer, inner の2値）

## Python/C++ の対応（Nonlinear）

| Python | C++ |
|---|---|
| `nonlinear_mpc = ...` | `auto nonlinear_mpc = xxx_nonlinear_mpc::make();` |
| `reference_path` ndarray | `xxx_nonlinear_mpc::ReferenceTrajectory_Type reference_trajectory;` |
| `update_manipulation(...)` | `U = nonlinear_mpc.update_manipulation(reference_trajectory, Y)` |
| 反復回数取得 | `nonlinear_mpc.get_solver_step_iterated_number()` |

`sample/nonlinear/simple_pendulum.cpp`、
`sample/nonlinear/kinematic_bicycle_model.cpp`、
`sample/nonlinear/kinematic_bicycle_model_op_en.cpp` に具体例があります。
