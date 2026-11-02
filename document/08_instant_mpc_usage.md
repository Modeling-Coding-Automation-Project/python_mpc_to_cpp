# 8. Instant MPC（iMPC）の使い方

Instant MPC は、Python で計算した固定行列を用い、制御周期ごとに所定回数の主変数・双対変数の更新を行う LTI コントローラです。オンラインで QP を収束まで解く方式とは更新方法が異なります。

この章は以下の実装とサンプルをもとにしています。

- [Python サンプル](../sample/lti/servo_motor_constraints_instant_MPC.py)
- [C++ サンプル](../sample/lti/servo_motor_constraints_instant_MPC.cpp)
- [Python コントローラ](../external_libraries/MCAP_python_mpc/python_mpc/linear_mpc_instant.py)
- [C++ コード生成](../python_mpc/linear_mpc_instant_deploy.py) / [C++ コントローラ](../python_mpc/python_linear_mpc_instant.hpp)
- [Python/C++ の SIL 比較](../test_sil/lti_mpc/servo_motor_constraints_instant_MPC_SIL.py)

## Python で設計・生成する

1. 離散時間の LTI モデルを作成する
2. 重み、予測区間、入力制約などを指定して `InstantMPC_LTI` を生成する
3. シミュレーション前に C++ ヘッダを生成する
4. `update(reference, y_measured)` を制御周期ごとに呼ぶ

次はサーボモータサンプルと同じ設定です。リポジトリ直下から実行し、Python の変数 `impc` にコントローラを保持します。

```python
import numpy as np

from sample.lti.servo_motor_constraints_instant_MPC import create_plant_model
from external_libraries.MCAP_python_mpc.mpc_utility.state_space_utility import SymbolicStateSpace
from external_libraries.MCAP_python_mpc.python_mpc.linear_mpc_instant import InstantMPC_LTI
from python_mpc.linear_mpc_instant_deploy import InstantMPC_LTI_Deploy

dt = 0.05
sys_d = create_plant_model().sample(Ts=dt, method="euler")
model = SymbolicStateSpace(
    sys_d.A, sys_d.B, sys_d.C, delta_time=dt, Number_of_Delay=0)

impc = InstantMPC_LTI(
    model, Np=20, Nc=2,
    Weight_U=np.diag([0.0005]),
    Weight_Y=np.diag([1.0, 0.005]),
    U_min=np.array([[-180.0]]), U_max=np.array([[180.0]]),
    zeta=50.0,
    near_zero_relative_limit=1.0e-5,
    is_reference_trajectory=False,
)

files = InstantMPC_LTI_Deploy.generate_InstantMPC_LTI_cpp_code(
    impc, file_name="motor")
print(files[-1])  # motor_impc.hpp
```

生成先は実行時のカレントディレクトリです。`file_name` は拡張子なしのファイル名・名前空間の接頭辞であり、出力ディレクトリではありません。省略時は呼出元スクリプト名が使われます。生成名には呼出元の変数名も入るため、上記のように名前付き変数を渡してください。名前には C++ の識別子として使える文字列を使用します。

メインヘッダは戻り値の最後の要素です。依存する行列、制約、カルマンフィルタのヘッダも生成されるため、一式を C++ のインクルード先に置きます。戻り値のカルマンフィルタ依存ファイル部分は入れ子のリストです。

## 主な設計パラメータ

`Nx` を状態数、`Nu` を入力数、`Ny` を出力数とします。

| 引数 | 意味・形状 |
|---|---|
| `state_space` | 離散モデル `SymbolicStateSpace`。サンプル周期と観測遅延を含む |
| `Np` | 予測区間のステップ数 |
| `Nc` | 制御区間のステップ数。`Nc <= Np` とし、以降の入力増分を等式制約でゼロにする |
| `Weight_U` | 入力増分の重み。例では `Nu × Nu` の対角行列 |
| `Weight_Y` | 出力誤差の重み。例では `Ny × Ny` の対角行列。目的関数では実効的に `Weight_Y.T @ Weight_Y` が使われる |
| `Q_kf`, `R_kf` | カルマンフィルタ用の共分散行列。省略時はそれぞれ状態数・出力数の単位行列 |
| `delta_U_min`, `delta_U_max` | 入力増分の下限・上限。各 `Nu × 1`、省略または `None` でその制約群を無効にする |
| `U_min`, `U_max` | 入力の下限・上限。各 `Nu × 1`、省略または `None` でその制約群を無効にする |
| `is_reference_trajectory` | `False` は一定目標値、`True` は予測区間の参照軌道 |
| `zeta` | 主変数・双対変数の更新速度に関わる係数。サンプルでは `50.0` |
| `near_zero_relative_limit` | オフライン行列中の微小係数をゼロ化する相対しきい値。疎構造と数値近似に影響する |

数値がゼロの制約も有効です。例えば `U_max=np.array([[0.0]])` は上限ゼロを指定し、`U_max=None` は上限制約を外します。配列の一部だけを `None` にする使い方ではありません。

内部更新回数は現在の Python 実装では `N_sub = max(1, ceil(dt * zeta))`、内部刻みは `dtzeta_sub = dt * zeta / N_sub` です。サンプルではそれぞれ `3` と約 `0.8333` になります。この分割だけで任意のモデル・重みに対する安定性が保証されるわけではありません。

## 一定目標値と参照軌道

一定目標値では `reference` は `Ny × 1` です。

```python
reference = np.array([[1.0], [0.0]])
U = impc.update(reference, y_measured)  # 戻り値は Nu × 1
```

参照軌道モードでは `Ny × Np` とし、列 `k` に予測ステップ `k` の目標値を入れます。最初の列を基準目標値とし、後続列との差分が目的関数の補正に使われます。

```python
reference = np.zeros((2, impc.Np))
for k in range(impc.Np):
    reference[0, k] = np.sin(2.0 * np.pi * 0.1 * (time + k * dt))
U = impc.update(reference, y_measured)
```

Python の軌道モードは `Ny × 1` の目標値を区間全体へ展開することもできます。C++ の `Reference_Type` は生成時に `Ny × 1` または `Ny × Np` に固定されるため、軌道モードでは全列を設定します。モードを変える際は Python コントローラを作り直して再生成します。

## C++ で使う

上の `file_name="motor"` の例からは次のインターフェースが生成されます。

```cpp
#include "motor_impc.hpp"

auto impc = motor_impc::make();
motor_impc::Reference_Type reference;
reference(0, 0) = 1.0;
reference(1, 0) = 0.0;

// y_measured は OUTPUT_SIZE × 1 の観測ベクトル。
auto U = impc.update(reference, y_measured);
```

`InstantMPC_LTI` のデフォルトコンストラクタだけでは設計済みの係数が入りません。通常は生成された `make()` を使用します。

| C++ API | 内容 |
|---|---|
| `motor_impc::type` | 生成されたコントローラ型 |
| `motor_impc::make()` | オフライン係数を設定し、内部履歴を初期化したコントローラを返す |
| `motor_impc::Reference_Type` | 参照ベクトルまたは参照軌道の型。`ReferenceTrajectory_Type` も同じ型 |
| `NP`, `NC` | 生成時の予測区間・制御区間 |
| `INPUT_SIZE`, `STATE_SIZE`, `OUTPUT_SIZE` | 入力数、状態数、出力数 |
| `NUMBER_OF_DELAY` | 生成時の観測遅延ステップ数 |
| `N_SUB`, `DTZETA_SUB` | 内部更新回数と内部刻み |
| `impc.update(reference, y_measured)` | 内部推定・最適化状態を進め、次に適用する入力を返す |
| `impc.get_X_inner_model()` | 遅延補償後の最新推定状態をコピーして返す |
| `impc.get_U_latest()` | 直近に算出した入力をコピーして返す |
| `impc.get_Y_store()` | 遅延補償用出力履歴をコピーして返す。遅延ゼロでは更新されない |

`make()` は実行中の Python オブジェクトの状態を再現するチェックポイントではありません。Python と C++ の比較は両方を同じ初期状態から始めます。C++ コントローラのコピーはその時点の内部状態もコピーし、初期化し直す場合は `impc = motor_impc::make();` とします。

## 観測と入力の更新タイミング

サーボモータサンプルは、**前回の入力を適用 → 状態を更新 → 更新後の出力を観測 → 次の入力を算出**という順序です。遅延ゼロ・直達項ゼロの Python 側では次の処理になります。

```python
X = sys_d.A @ X + sys_d.B @ U
y_measured = sys_d.C @ X
U = impc.update(reference, y_measured)
```

対応する C++ の順序は次のとおりです。

```cpp
sys.update(U);
const auto y_measured = sys.output_equation(sys.get_X(), U);
U = impc.update(reference, y_measured);
```

`DiscreteStateSpace::update()` は出力を計算してから状態を進めるため、この位置の `sys.get_Y()` は状態更新前の観測です。上記サンプルでは更新後の状態から出力を明示的に計算します。異なる順序を使う場合も、Python、C++、実機の観測時刻と前回入力の対応をそろえてください。

`update()` は内部で前回の入力を使い、固定ゲインのカルマンフィルタを予測・更新します。観測遅延ありの場合は設定した入力履歴で状態を外挿し、その状態から出力誤差を作ります。サーボモータサンプルと後述の SIL は遅延ゼロです。

最初の参照列と、直近のリセット時に記録した参照値（初期値はゼロ）との最大絶対差が `1e-6` を超えると、主変数・双対変数がリセットされます。推定状態、前回入力、遅延履歴は継続します。また、主変数の初期値は毎回オフラインの無制約初期化マップから作り直され、双対変数は参照変更時を除いて引き継がれます。

## オフライン計算と生成行列

拡大状態の次元は `Ns = Nx + Ny`、決定変数の次元は `Nw = Np * (Nu + Ns)` です。決定変数には `Np * Nu` 個の入力増分と、`Np` ステップ分の拡大状態を並べます。`Nc` を小さくしてもこの次元は減らず、区間後半の入力増分をゼロにする等式制約が追加されます。

次の式は微小係数のゼロ化前の関係を表します。`Ceq * w + Deq * xt = 0` を等式制約、`P` を目的関数の二次項、`xt = [推定状態の差分; 推定出力 - 最初の参照列]` とします。

| 生成行列 | 役割 |
|---|---|
| `K`, `L` | 等式制約への射影 `wproj = K*w + L*xt`。`K = I - Ceq⁺*Ceq`、`L = -Ceq⁺*Deq` |
| `q_K_matrix` | `K*P*L` から作る目的関数のオフセット用マップ |
| `w_unc_map` | 正則化した等式制約付き KKT 系から求める、毎回の主変数初期値のマップ |
| `w_unc_traj_map` | 同じ KKT 系から求める参照軌道補正用マップ。一定目標値モードではゼロ |
| `M_sub_inv` | `I + dtzeta_sub * P_K` の逆行列。`P_K` は `K*P*K` から生成。オンラインでは行列積のみを行う |
| `Ag`, `bg` | 入力制約 `Ag*wproj - bg <= 0`。入力増分上限・下限、入力上限・下限の順に有効な群を並べる |
| `AgKT` | `(Ag*K).T` から作る、双対変数と主変数を結ぶマップ |
| `WCzT_Qk` | 参照列と最初の参照列の差を、拡大状態部分の目的関数補正へ変換するマップ |

入力上限・下限の `bg` は、毎回 `U_max - U_latest`、`U_latest - U_min` に更新されます。入力増分の制約部分は固定です。制約がすべて無効な場合は、生成行列には形状を保つダミー行が入りますが、C++ の制約更新分岐は無効になります。

モデル、重み、区間、制約、サンプル周期、`zeta` を変更する場合は、Python の設計から全ヘッダを再生成してください。`N_SUB` や `DTZETA_SUB` だけを編集すると、事前計算した `M_sub_inv` などとの対応が崩れます。

## 制約と利用上の制限

- 対応する不等式制約は入力と入力増分の上下限です。このクラスには `Y_min` / `Y_max` による出力制約の引数はありません。
- 有限回の内部更新で入力を算出し、最後に上下限へクリップする処理はありません。制約違反が残ることがあり、サンプル設定でも一時的な入力上限超過が生じます。
- QP の最適解への収束判定や、実行時の制約充足・安定性を証明するステータスは返しません。モデル誤差、丸め、重み、更新刻みを含めて閉ループ挙動を確認します。
- コントローラは返した入力が次の周期に適用される前提で推定を進めます。外部で入力を飽和させたり変更したりする場合、内部の前回入力との不一致にも対応が必要です。
- オンラインでモデルや重みを差し替える LTV 用の更新 API はありません。参照と観測を変更して使う、固定設計の LTI コントローラです。
- 内部では `Nw × Nw` の行列も保持します。予測区間を延ばすとオフライン設計、生成物、C++ ビルド、実行時の計算量・メモリが増えるため、対象環境で測定してください。

## サンプルを実行する

リポジトリ直下で Python サンプルを実行すると、ヘッダ生成と 400 ステップのシミュレーションを行います。`CI=1` はプロットの HTML 出力とブラウザ表示を抑止します。

```bash
CI=1 /opt/venv_py_MCAP/bin/python sample/lti/servo_motor_constraints_instant_MPC.py
```

既定の `PATH_FOLLOWING=False` は 20 秒周期・デューティ比 50% の目標値で、最初の 10 秒は第 1 出力を `1`、次の 10 秒は `0` とします。`True` にすると 0.1 Hz の正弦波を先読みする参照軌道になり、第 2 出力の目標値はいずれもゼロです。モード変更後は生成からやり直します。

C++ サンプルは生成済みの `servo_motor_constraints_instant_MPC_impc.hpp` を読み込みます。C++11 以降で、リポジトリ直下と各ライブラリのヘッダディレクトリをインクルードパスに指定します。VS Code の [.vscode/tasks.json](../.vscode/tasks.json) に設定例があり、対象 C++ ファイルを開いてビルドできます。
