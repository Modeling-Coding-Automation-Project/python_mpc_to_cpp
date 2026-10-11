# 2. 線形MPCの使い方（LTI/LTV）

この章は、主に以下サンプルをもとにしています。

- `sample/lti_no_constraints/state_space_SISO.py`
- `sample/lti_no_constraints/servo_motor.py`
- `sample/lti/servo_motor_constraints.py`
- `sample/ltv_no_constraints/servo_motor_ltv.py`
- `sample/ltv/servo_motor_ltv_constraints.py`

## LTI（制約なし）の最小パターン

1. 離散状態空間を作成
2. `LTI_MPC_NoConstraints` を生成
3. `update(reference, y_measured)` を繰り返し呼ぶ
4. 必要なら C++ コードを生成

```python
ideal_plant_model = SymbolicStateSpace(A, B, C, delta_time=dt, Number_of_Delay=5)
lti_mpc = LTI_MPC_NoConstraints(
    ideal_plant_model, Np=10, Nc=2,
    Weight_U=Weight_U, Weight_Y=Weight_Y,
)

deployed_file_names = LinearMPC_Deploy.generate_LTI_MPC_NC_cpp_code(lti_mpc)
U = lti_mpc.update(reference, y_measured)
```

## LTI（制約あり）で追加するもの

`sample/lti/servo_motor_constraints.py` のように、制約ありでは上下限を渡します。

- `delta_U_min / delta_U_max`
- `U_min / U_max`
- `Y_min / Y_max`

```python
lti_mpc = LTI_MPC(
    ideal_plant_model, Np=Np, Nc=Nc,
    Weight_U=Weight_U, Weight_Y=Weight_Y,
    delta_U_min=delta_U_min, delta_U_max=delta_U_max,
    U_min=U_min, U_max=U_max,
    Y_min=Y_min, Y_max=Y_max,
)
```

## LTVのポイント

`sample/ltv_no_constraints/servo_motor_ltv.py` / `sample/ltv/servo_motor_ltv_constraints.py` では:

- SymPyでパラメータ依存の `A, B, C, D` を作成
- `parameters_struct` をMPC生成時に渡す
- シミュレーション中に `update_parameters(...)` で更新可能

```python
ltv_mpc = LTV_MPC_NoConstraints(
    state_space=ideal_plant_model,
    parameters_struct=controller_parameters,
    Np=Np, Nc=Nc, Weight_U=Weight_U, Weight_Y=Weight_Y,
)

ltv_mpc.update_parameters(controller_parameters)
U = ltv_mpc.update_manipulation(reference, y_measured)
```

## 参照信号の与え方

`Reference` は出力次元に合わせて渡します。

- 1出力なら `(1,1)` または `(1,Np)`（軌道参照時）
- 多出力なら `[[y1_ref], [y2_ref], ...]`

## PythonからC++へ移る時の対応

| Python側 | C++側（生成後） |
|---|---|
| `lti_mpc = ...` | `auto lti_mpc = xxx_lti_mpc::make();` |
| `reference` ndarray | `xxx_lti_mpc::Reference_Type reference;` |
| `update(...)` | `lti_mpc.update(reference, sys.get_Y())` |
| `LTV update_parameters(...)` | `ltv_mpc.update_parameters(controller_parameters)` |

`sample/lti_no_constraints/state_space_SISO.cpp` と `sample/ltv_no_constraints/servo_motor_ltv.cpp` がそのまま実装例です。
