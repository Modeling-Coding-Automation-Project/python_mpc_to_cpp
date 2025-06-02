from dataclasses import dataclass
import numpy as np
import sympy as sp

from external_libraries.python_optimization_to_cpp.python_optimization.qp_active_set import QP_ActiveSetSolver


@dataclass
class UXY_Limits:
    delta_U_min: np.ndarray = None
    delta_U_max: np.ndarray = None
    U_min: np.ndarray = None
    U_max: np.ndarray = None
    Y_min: np.ndarray = None
    Y_max: np.ndarray = None


class LTI_MPC_QP_Solver:
    def __init__(self, number_of_variables: int, x: np.ndarray = None,
                 delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                 U_min: np.ndarray = None, U_max: np.ndarray = None,
                 Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        self.number_of_variables = number_of_variables
        self.number_of_constraints = None

        self.UXY_Limits = UXY_Limits(
            delta_U_min=delta_U_min,
            delta_U_max=delta_U_max,
            U_min=U_min,
            U_max=U_max,
            Y_min=Y_min,
            Y_max=Y_max
        )

        self.number_of_constraints = self.count_constraints(self,
                                                            UXY_Limits=self.UXY_Limits
                                                            )

        self.active_set_solver = QP_ActiveSetSolver(
            number_of_variables=number_of_variables,
            number_of_constraints=self.number_of_constraints,
            x=x)

        self.M = np.zeros(
            (self.number_of_constraints, self.number_of_variables))
        self.gamma = np.zeros((self.number_of_constraints, 1))

    def check_min_max_compatibility(self, UXY_Limits: UXY_Limits):

        if UXY_Limits.delta_U_min is not None or UXY_Limits.delta_U_min.shape[1] > 1:
            raise ValueError("delta_U_min must be a (n, 1) vector.")
        if UXY_Limits.delta_U_max is not None or UXY_Limits.delta_U_max.shape[1] > 1:
            raise ValueError("delta_U_max must be a (n, 1) vector.")

        if UXY_Limits.U_min is not None or UXY_Limits.U_min.shape[1] > 1:
            raise ValueError("U_min must be a (n, 1) vector.")
        if UXY_Limits.U_max is not None or UXY_Limits.U_max.shape[1] > 1:
            raise ValueError("U_max must be a (n, 1) vector.")

        if UXY_Limits.Y_min is not None or UXY_Limits.Y_min.shape[1] > 1:
            raise ValueError("Y_min must be a (n, 1) vector.")
        if UXY_Limits.Y_max is not None or UXY_Limits.Y_max.shape[1] > 1:
            raise ValueError("Y_max must be a (n, 1) vector.")

    def count_constraints(self, UXY_Limits: UXY_Limits):

        self.check_min_max_compatibility(self,
                                         UXY_Limits=UXY_Limits
                                         )

        count = 0

        if UXY_Limits.delta_U_min is not None:
            for i in range(UXY_Limits.delta_U_min.shape[0]):
                if np.isfinite(UXY_Limits.delta_U_min[i]):
                    count += 1
        if UXY_Limits.delta_U_max is not None:
            for i in range(UXY_Limits.delta_U_max.shape[0]):
                if np.isfinite(UXY_Limits.delta_U_max[i]):
                    count += 1

        if UXY_Limits.U_min is not None:
            for i in range(UXY_Limits.U_min.shape[0]):
                if np.isfinite(UXY_Limits.U_min[i]):
                    count += 1
        if UXY_Limits.U_max is not None:
            for i in range(UXY_Limits.U_max.shape[0]):
                if np.isfinite(UXY_Limits.U_max[i]):
                    count += 1

        if UXY_Limits.Y_min is not None:
            for i in range(UXY_Limits.Y_min.shape[0]):
                if np.isfinite(UXY_Limits.Y_min[i]):
                    count += 1
        if UXY_Limits.Y_max is not None:
            for i in range(UXY_Limits.Y_max.shape[0]):
                if np.isfinite(UXY_Limits.Y_max[i]):
                    count += 1

        if self.number_of_constraints is not None and \
                (self.number_of_constraints != count):
            raise ValueError(
                "The number of constraints does not match the expected count.")

        return count

    def update_constraints(self, U: np.ndarray, X: np.ndarray, Phi: np.ndarray):

        if 0 == self.number_of_constraints:
            raise ValueError(
                "No constraints defined. Please set constraints before updating.")

        M = np.zeros((self.number_of_constraints, self.number_of_variables))
        gamma = np.zeros((self.number_of_constraints, 1))

        return M, gamma
