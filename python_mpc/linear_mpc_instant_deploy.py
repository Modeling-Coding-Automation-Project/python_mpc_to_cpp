
"""
File: linear_mpc_instant_deploy.py

Export an offline-designed InstantMPC_LTI controller as C++11 model headers.

The Python controller constructor prepares equality-projection matrices,
input-constraint matrices, regularized unconstrained initialization maps,
a semi-implicit inverse, and a fixed Kalman gain. This module serializes those
existing coefficients; it does not redesign the controller or solve a QP.
The generated namespace provides ``type``, ``Reference_Type``, horizon/size
constants and ``make()`` for ``python_linear_mpc_instant.hpp``.

Bounds are encoded by their presence, independently of their numeric value:
``None`` produces a structurally empty bound type, while a supplied zero bound
remains active. Point and trajectory references select different C++ types.
All matrices and substep parameters must be regenerated together when the
model, weights, horizon, constraints, sample period or integration gain change.

Generated ``make()`` initializes fresh controller history. This is a model
export, not a checkpoint of the Python object's current primal/dual state,
state estimate or previous input. Generate before the simulation loop when
comparing the initial Python and C++ controllers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(os.path.join(
    os.getcwd(), "external_libraries", "python_control_to_cpp"))

import inspect
import numpy as np

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import KalmanFilterDeploy

from python_mpc.common_mpc_deploy import convert_SparseAvailable_for_deploy
from external_libraries.python_optimization_to_cpp.optimization_utility.common_optimization_deploy import MinMaxCodeGenerator

from external_libraries.MCAP_python_mpc.python_mpc.linear_mpc_instant import InstantMPC_LTI


class InstantMPC_LTI_Deploy:
    """
    Serialize a prepared InstantMPC_LTI instance and its matrix dependencies.

    The public entry point is ``generate_InstantMPC_LTI_cpp_code``. It emits
    Kalman-filter/state-space headers, fixed-size matrix headers, bound types
    and a final controller factory. Files are written in the current working
    directory, following the same naming convention as the other MPC deployers.

    The generated factory selects active-bound and trajectory branches through
    types. No active-set QP solver, runtime inverse or iteration-until-converged
    loop is generated; C++ performs the controller's configured substeps.
    """

    @staticmethod
    def generate_InstantMPC_LTI_cpp_code(
            impc: InstantMPC_LTI, file_name=None):
        """
        Generate the complete header set for one offline controller design.

        Args:
            impc (InstantMPC_LTI): Initialized controller with prepared matrices
                and a fixed Kalman gain. Bind it to a named variable in the
                calling function: its local variable name forms part of every
                generated namespace and file name.
            file_name (str, optional): File/namespace prefix without extension.
                If None, use the caller's script basename. Choose a prefix
                valid in a C++ identifier; it is not an output-directory option.

        Returns:
            list: Generated dependencies followed by the main controller header
                as the last entry. The Kalman-filter dependency entry is itself
                a list, matching the other deployment APIs.

        Side Effects:
            Writes or replaces generated headers in the working directory.
            For ``impc`` in ``motor.py``, the main header and namespace are
            ``motor_impc.hpp`` and ``motor_impc`` respectively.

        Notes:
            Scalar type follows the Kalman state matrix dtype and is checked by
            ControlDeploy/NumpyDeploy. Matrix sparsity is derived from the
            already prepared coefficients, including Python's near-zero
            rounding. N_SUB and DTZETA_SUB are exported together with
            M_sub_inv; changing either constant alone is not a redesign.
            This function does not serialize running controller history or
            certify constraint satisfaction/stability of the chosen design.
        """
        number_of_delay = impc.Number_of_Delay

        deployed_file_names = []

        data_type = impc._Kalman_filter.A.dtype
        ControlDeploy.restrict_data_type(data_type.name)

        type_name = NumpyDeploy.check_dtype(impc._Kalman_filter.A)

        # %% inspect arguments
        # Match the caller's object by identity so generated C++ names retain
        # the Python variable name. The caller must keep a named local binding.
        frame = inspect.currentframe().f_back
        caller_locals = frame.f_locals
        variable_name = None
        for name, value in caller_locals.items():
            if value is impc:
                variable_name = name
                break
        if file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_no_extension = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_no_extension = file_name

        # %% code generation
        code_file_name = caller_file_name_no_extension + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # --- 1. Kalman filter code ---
        # Export A/B/C, delay metadata, Q/R and the prepared P/G. The online
        # iMPC path uses predict_and_update_with_fixed_G, not gain convergence.
        locals_map = {
            f"{variable_name}_lkf": impc._Kalman_filter,
            "caller_file_name_no_extension": caller_file_name_no_extension,
            "number_of_delay": number_of_delay
        }
        lkf_file_names = eval(
            f"KalmanFilterDeploy.generate_LKF_cpp_code({variable_name}_lkf, "
            "caller_file_name_no_extension, number_of_delay=number_of_delay)",
            globals(),
            locals_map
        )

        deployed_file_names.append(lkf_file_names)
        lkf_file_name = lkf_file_names[-1]
        lkf_file_name_no_extension = lkf_file_name.split(".")[0]

        # --- 2. Trajectory gradient map WCzT_Qk (Ns x Ny) ---
        # WCz = Weight_Y * [0, I]; WCz.T * Weight_Y maps a reference deviation
        # to its state-gradient block. It is zero in point-reference mode.
        # NumpyDeploy chooses the actual storage type from the coefficients.
        locals_map = {
            f"{variable_name}_WCzT_Qk": np.array(impc._WCzT_Qk, dtype=data_type),
            "caller_file_name_no_extension": caller_file_name_no_extension
        }
        WCzT_Qk_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_WCzT_Qk, "
            "file_name=caller_file_name_no_extension)",
            globals(),
            locals_map
        )
        deployed_file_names.append(WCzT_Qk_file_name)
        WCzT_Qk_file_name_no_extension = WCzT_Qk_file_name.split(".")[0]

        # --- 3. Bound matrices (delta_U_min, delta_U_max, U_min, U_max) ---
        # Encode whether an entire bound vector is enabled, not whether its
        # numerical entries are zero. Disabled bounds use SparseMatrixEmpty;
        # enabled bounds receive their values in the final generated make().
        U_size = impc.Nu

        def _make_bound_generator(attr_name, bound_flag, min_max_name):
            """Build a per-input activity mask from the Python bound flag.

            ``attr_name`` selects the numeric array, ``bound_flag`` controls
            whether all channels of this bound group are active, and
            ``min_max_name`` supplies its generated field/file suffix.
            """
            arr = getattr(impc, attr_name, None)
            generator = MinMaxCodeGenerator(
                min_max_array=arr if bound_flag else None,
                min_max_name=min_max_name,
                size=U_size
            )
            generator.generate_active_set(
                is_active_array=(
                    np.ones(U_size, dtype=bool) if bound_flag else
                    np.zeros(U_size, dtype=bool)
                )
            )
            return generator

        delta_U_min_gen = _make_bound_generator(
            "delta_U_min", impc._use_du_lb, "delta_U_min")
        delta_U_max_gen = _make_bound_generator(
            "delta_U_max", impc._use_du_ub, "delta_U_max")
        U_min_gen = _make_bound_generator("U_min", impc._use_u_lb, "U_min")
        U_max_gen = _make_bound_generator("U_max", impc._use_u_ub, "U_max")

        delta_U_min_file_name, delta_U_min_file_name_no_extension = \
            delta_U_min_gen.create_limits_code(data_type, variable_name,
                                               caller_file_name_no_extension)
        deployed_file_names.append(delta_U_min_file_name)

        delta_U_max_file_name, delta_U_max_file_name_no_extension = \
            delta_U_max_gen.create_limits_code(data_type, variable_name,
                                               caller_file_name_no_extension)
        deployed_file_names.append(delta_U_max_file_name)

        U_min_file_name, U_min_file_name_no_extension = \
            U_min_gen.create_limits_code(data_type, variable_name,
                                         caller_file_name_no_extension)
        deployed_file_names.append(U_min_file_name)

        U_max_file_name, U_max_file_name_no_extension = \
            U_max_gen.create_limits_code(data_type, variable_name,
                                         caller_file_name_no_extension)
        deployed_file_names.append(U_max_file_name)

        # --- 4. Inequality coefficients Ag (Nmu x Nw) ---
        # g = Ag * wproj - bg. Groups are delta-U upper, delta-U lower,
        # U upper, U lower, each tiled across Np; absent groups are omitted.
        locals_map = {
            f"{variable_name}_Ag": np.array(impc._Ag, dtype=data_type),
            "caller_file_name_no_extension": caller_file_name_no_extension
        }
        Ag_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_Ag, "
            "file_name=caller_file_name_no_extension)",
            globals(),
            locals_map
        )
        deployed_file_names.append(Ag_file_name)
        Ag_file_name_no_extension = Ag_file_name.split(".")[0]

        # --- 5. Inequality bounds bg (Nmu x 1) ---
        # Absolute-U groups are refreshed in C++ from U_max-U_latest and
        # U_latest-U_min. Delta-U groups retain these exported constant values.
        bg_cpp = np.array(impc._bg, dtype=data_type)
        locals_map = {
            f"{variable_name}_bg": bg_cpp,
            "caller_file_name_no_extension": caller_file_name_no_extension
        }
        bg_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_bg, "
            "file_name=caller_file_name_no_extension)",
            globals(),
            locals_map
        )
        deployed_file_names.append(bg_file_name)
        bg_file_name_no_extension = bg_file_name.split(".")[0]

        # --- 6. Equality null-space projection K (Nw x Nw) ---
        # Nominally I - C_eq^+ * C_eq; export the rounded Python matrix as-is.
        # Nc is already encoded by equality rows for later zero increments.
        locals_map = {
            f"{variable_name}_K": np.array(impc._K, dtype=data_type),
            "caller_file_name_no_extension": caller_file_name_no_extension
        }
        K_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_K, "
            "file_name=caller_file_name_no_extension)",
            globals(),
            locals_map
        )
        deployed_file_names.append(K_file_name)
        K_file_name_no_extension = K_file_name.split(".")[0]

        # --- 7. Equality particular-solution map L (Nw x Ns) ---
        # Nominally -C_eq^+ * D_eq. K*w + L*xt is the affine projection,
        # where xt = [estimated delta_x; estimated y - base reference].
        L_cpp = np.array(impc._L, dtype=data_type)
        locals_map = {
            f"{variable_name}_L": L_cpp,
            "caller_file_name_no_extension": caller_file_name_no_extension
        }
        L_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_L, "
            "file_name=caller_file_name_no_extension)",
            globals(),
            locals_map
        )
        deployed_file_names.append(L_file_name)
        L_file_name_no_extension = L_file_name.split(".")[0]

        # --- 8. Projected objective offset q_K_matrix (Nw x Ns) ---
        # Prepared from K*P*L and applied once per control step to xt.
        q_K_matrix_cpp = np.array(impc._q_K_matrix, dtype=data_type)
        locals_map = {
            f"{variable_name}_q_K_matrix": q_K_matrix_cpp,
            "caller_file_name_no_extension": caller_file_name_no_extension
        }
        q_K_matrix_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_q_K_matrix, "
            "file_name=caller_file_name_no_extension)",
            globals(),
            locals_map
        )
        deployed_file_names.append(q_K_matrix_file_name)
        q_K_matrix_file_name_no_extension = q_K_matrix_file_name.split(".")[0]

        # --- 9. Unconstrained initialization map w_unc_map (Nw x Ns) ---
        # Obtained by a regularized equality-constrained KKT solve in Python.
        # C++ rebuilds its primal guess with this map on every update.
        w_unc_map_cpp = np.array(impc._w_unc_map, dtype=data_type)
        locals_map = {
            f"{variable_name}_w_unc_map": w_unc_map_cpp,
            "caller_file_name_no_extension": caller_file_name_no_extension
        }
        w_unc_map_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_w_unc_map, "
            "file_name=caller_file_name_no_extension)",
            globals(),
            locals_map
        )
        deployed_file_names.append(w_unc_map_file_name)
        w_unc_map_file_name_no_extension = w_unc_map_file_name.split(".")[0]

        # --- 10. Trajectory initialization correction (Nw x Nw) ---
        # A second KKT right-hand side maps q_trajectory into the primal guess.
        # Python provides an all-zero map when trajectory mode is disabled.
        locals_map = {
            f"{variable_name}_w_unc_traj_map": np.array(impc._w_unc_traj_map, dtype=data_type),
            "caller_file_name_no_extension": caller_file_name_no_extension
        }
        w_unc_traj_map_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_w_unc_traj_map, "
            "file_name=caller_file_name_no_extension)",
            globals(),
            locals_map
        )
        deployed_file_names.append(w_unc_traj_map_file_name)
        w_unc_traj_map_file_name_no_extension = w_unc_traj_map_file_name.split(".")[
            0]

        # --- 11. Semi-implicit inverse M_sub_inv (Nw x Nw) ---
        # Prepared from I + DTZETA_SUB * P_K, with P_K derived from K*P*K.
        # The online loop only multiplies by this matrix; it does not invert it.
        locals_map = {
            f"{variable_name}_M_sub_inv": np.array(impc._M_sub_inv, dtype=data_type),
            "caller_file_name_no_extension": caller_file_name_no_extension
        }
        M_sub_inv_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_M_sub_inv, "
            "file_name=caller_file_name_no_extension)",
            globals(),
            locals_map
        )
        deployed_file_names.append(M_sub_inv_file_name)
        M_sub_inv_file_name_no_extension = M_sub_inv_file_name.split(".")[0]

        # --- 12. Projected inequality coupling AgKT (Nw x Nmu) ---
        # Rounded (Ag*K).T multiplies the previous dual vector times eta.
        AgKT_cpp = np.array(impc._AgKT, dtype=data_type)
        locals_map = {
            f"{variable_name}_AgKT": AgKT_cpp,
            "caller_file_name_no_extension": caller_file_name_no_extension
        }
        AgKT_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_AgKT, "
            "file_name=caller_file_name_no_extension)",
            globals(),
            locals_map
        )
        deployed_file_names.append(AgKT_file_name)
        AgKT_file_name_no_extension = AgKT_file_name.split(".")[0]

        # %% main code generation
        code_text = ""

        file_header_macro_name = code_file_name.upper() + "_HPP_"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{lkf_file_name}\"\n"
        code_text += f"#include \"{WCzT_Qk_file_name}\"\n"
        code_text += f"#include \"{delta_U_min_file_name}\"\n"
        code_text += f"#include \"{delta_U_max_file_name}\"\n"
        code_text += f"#include \"{U_min_file_name}\"\n"
        code_text += f"#include \"{U_max_file_name}\"\n"
        code_text += f"#include \"{Ag_file_name}\"\n"
        code_text += f"#include \"{bg_file_name}\"\n"
        code_text += f"#include \"{K_file_name}\"\n"
        code_text += f"#include \"{L_file_name}\"\n"
        code_text += f"#include \"{q_K_matrix_file_name}\"\n"
        code_text += f"#include \"{w_unc_map_file_name}\"\n"
        code_text += f"#include \"{w_unc_traj_map_file_name}\"\n"
        code_text += f"#include \"{M_sub_inv_file_name}\"\n"
        code_text += f"#include \"{AgKT_file_name}\"\n\n"
        code_text += "#include \"python_mpc.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n"
        code_text += "using namespace PythonMPC;\n\n"

        code_text += f"constexpr std::size_t NP = {impc.Np};\n"
        code_text += f"constexpr std::size_t NC = {impc.Nc};\n\n"

        code_text += f"constexpr std::size_t INPUT_SIZE = {lkf_file_name_no_extension}::INPUT_SIZE;\n"
        code_text += f"constexpr std::size_t STATE_SIZE = {lkf_file_name_no_extension}::STATE_SIZE;\n"
        code_text += f"constexpr std::size_t OUTPUT_SIZE = {lkf_file_name_no_extension}::OUTPUT_SIZE;\n\n"

        code_text += f"constexpr std::size_t NUMBER_OF_DELAY = {lkf_file_name_no_extension}::NUMBER_OF_DELAY;\n\n"

        # Substeps selected by the Python design: h = dt*zeta/N_sub.
        # Keep these constants paired with the exported semi-implicit inverse.
        code_text += f"constexpr std::size_t N_SUB = {impc._N_sub};\n"
        code_text += f"constexpr {type_name} DTZETA_SUB = " \
            f"static_cast<{type_name}>({impc._dtzeta_sub});\n\n"

        code_text += f"using LKF_Type = {lkf_file_name_no_extension}::type;\n\n"

        code_text += f"using WCzT_Qk_Type = {WCzT_Qk_file_name_no_extension}::type;\n\n"

        code_text += f"using Delta_U_Min_Type = {delta_U_min_file_name_no_extension}::type;\n\n"
        code_text += f"using Delta_U_Max_Type = {delta_U_max_file_name_no_extension}::type;\n\n"
        code_text += f"using U_Min_Type = {U_min_file_name_no_extension}::type;\n\n"
        code_text += f"using U_Max_Type = {U_max_file_name_no_extension}::type;\n\n"

        code_text += f"using Ag_Type = {Ag_file_name_no_extension}::type;\n\n"
        code_text += f"using Bg_Type = {bg_file_name_no_extension}::type;\n\n"

        code_text += f"using K_Type = {K_file_name_no_extension}::type;\n\n"
        code_text += f"using L_Type = {L_file_name_no_extension}::type;\n\n"
        code_text += f"using q_K_matrix_Type = {q_K_matrix_file_name_no_extension}::type;\n\n"
        code_text += f"using W_Unc_Map_Type = {w_unc_map_file_name_no_extension}::type;\n\n"
        code_text += f"using W_Unc_Traj_Map_Type = {w_unc_traj_map_file_name_no_extension}::type;\n\n"
        code_text += f"using M_Sub_Inv_Type = {M_sub_inv_file_name_no_extension}::type;\n\n"
        code_text += f"using AgKT_Type = {AgKT_file_name_no_extension}::type;\n\n"

        # Each trajectory column is one prediction-step reference. Point mode
        # keeps a single Ny-by-1 column; it is a separate generated C++ type.
        ref_row_size_text = "1"
        if impc.is_reference_trajectory:
            ref_row_size_text = "NP"

        code_text += f"using Reference_Type = DenseMatrix_Type<{type_name}, OUTPUT_SIZE, " \
            + ref_row_size_text + ">;\n\n"

        code_text += "using ReferenceTrajectory_Type = Reference_Type;\n\n"

        code_text += "using type = InstantMPC_LTI_Type<\n" \
                     "  NP, NC, LKF_Type, ReferenceTrajectory_Type,\n" \
                     "  Delta_U_Min_Type, Delta_U_Max_Type,\n" \
                     "  U_Min_Type, U_Max_Type,\n" \
                     "  WCzT_Qk_Type, Ag_Type, Bg_Type,\n" \
                     "  K_Type, L_Type, q_K_matrix_Type,\n" \
                     "  W_Unc_Map_Type, W_Unc_Traj_Map_Type, M_Sub_Inv_Type, AgKT_Type>;\n\n"

        code_text += "inline auto make(void) -> type {\n\n"

        code_text += f"  auto kalman_filter = {lkf_file_name_no_extension}::make();\n\n"
        code_text += f"  auto WCzT_Qk = {WCzT_Qk_file_name_no_extension}::make();\n\n"

        code_text = delta_U_min_gen.write_limits_code(code_text, type_name)
        code_text = delta_U_max_gen.write_limits_code(code_text, type_name)
        code_text = U_min_gen.write_limits_code(code_text, type_name)
        code_text = U_max_gen.write_limits_code(code_text, type_name)

        code_text += f"  auto Ag = {Ag_file_name_no_extension}::make();\n\n"
        code_text += f"  auto bg = {bg_file_name_no_extension}::make();\n\n"
        code_text += f"  constexpr std::size_t N_sub = N_SUB;\n"
        code_text += f"  constexpr {type_name} dtzeta_sub = DTZETA_SUB;\n\n"
        code_text += f"  auto K = {K_file_name_no_extension}::make();\n\n"
        code_text += f"  auto L = {L_file_name_no_extension}::make();\n\n"
        code_text += f"  auto q_K_matrix = {q_K_matrix_file_name_no_extension}::make();\n\n"
        code_text += f"  auto w_unc_map = {w_unc_map_file_name_no_extension}::make();\n\n"
        code_text += f"  auto w_unc_traj_map = {w_unc_traj_map_file_name_no_extension}::make();\n\n"
        code_text += f"  auto M_sub_inv = {M_sub_inv_file_name_no_extension}::make();\n\n"
        code_text += f"  auto AgKT = {AgKT_file_name_no_extension}::make();\n\n"

        code_text += f"  auto {variable_name} = make_InstantMPC_LTI<NP, NC, LKF_Type," \
            "ReferenceTrajectory_Type,\n" \
            "    Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type, U_Max_Type,\n" \
            "    WCzT_Qk_Type, Ag_Type, Bg_Type,\n" \
            "    K_Type, L_Type, q_K_matrix_Type,\n" \
            "    W_Unc_Map_Type, W_Unc_Traj_Map_Type, M_Sub_Inv_Type, AgKT_Type>(\n" \
            "    kalman_filter, WCzT_Qk,\n" \
            "    delta_U_min, delta_U_max, U_min, U_max,\n" \
            "    Ag, bg, N_sub, dtzeta_sub,\n" \
            "    K, L, q_K_matrix, w_unc_map, w_unc_traj_map, M_sub_inv, AgKT);\n\n"

        code_text += f"  return {variable_name};\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
