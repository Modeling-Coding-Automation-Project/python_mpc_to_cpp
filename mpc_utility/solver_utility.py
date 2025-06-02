import numpy as np
import sympy as sp

from external_libraries.python_optimization_to_cpp.python_optimization.qp_active_set import QP_ActiveSetSolver


class LTI_MPC_QP_Solver:
    def __init__(self, number_of_variables: int, x: np.ndarray = None,
                 delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                 U_min: np.ndarray = None, U_max: np.ndarray = None,
                 Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        self.number_of_variables = number_of_variables
        self.number_of_constraints = None

        self.number_of_constraints = self.count_constraints(self,
                                                            delta_U_min=delta_U_min,
                                                            delta_U_max=delta_U_max,
                                                            U_min=U_min,
                                                            U_max=U_max,
                                                            Y_min=Y_min,
                                                            Y_max=Y_max
                                                            )

        self.active_set_solver = QP_ActiveSetSolver(
            number_of_variables=number_of_variables,
            number_of_constraints=self.number_of_constraints,
            x=x)

        self.M, self.gamma = self.initialize_constraints(
            delta_U_min=delta_U_min,
            delta_U_max=delta_U_max,
            U_min=U_min,
            U_max=U_max,
            Y_min=Y_min,
            Y_max=Y_max
        )

    def check_min_max_compatibility(self, delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                                    U_min: np.ndarray = None, U_max: np.ndarray = None,
                                    Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        if delta_U_min is not None or delta_U_min.shape[1] > 1:
            raise ValueError("delta_U_min must be a (n, 1) vector.")
        if delta_U_max is not None or delta_U_max.shape[1] > 1:
            raise ValueError("delta_U_max must be a (n, 1) vector.")

        if U_min is not None or U_min.shape[1] > 1:
            raise ValueError("U_min must be a (n, 1) vector.")
        if U_max is not None or U_max.shape[1] > 1:
            raise ValueError("U_max must be a (n, 1) vector.")

        if Y_min is not None or Y_min.shape[1] > 1:
            raise ValueError("Y_min must be a (n, 1) vector.")
        if Y_max is not None or Y_max.shape[1] > 1:
            raise ValueError("Y_max must be a (n, 1) vector.")

    def count_constraints(self, delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                          U_min: np.ndarray = None, U_max: np.ndarray = None,
                          Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        self.check_min_max_compatibility(self,
                                         delta_U_min=delta_U_min,
                                         delta_U_max=delta_U_max,
                                         U_min=U_min,
                                         U_max=U_max,
                                         Y_min=Y_min,
                                         Y_max=Y_max
                                         )

        count = 0

        if delta_U_min is not None:
            for i in range(delta_U_min.shape[0]):
                if np.isfinite(delta_U_min[i]):
                    count += 1
        if delta_U_max is not None:
            for i in range(delta_U_max.shape[0]):
                if np.isfinite(delta_U_max[i]):
                    count += 1

        if U_min is not None:
            for i in range(U_min.shape[0]):
                if np.isfinite(U_min[i]):
                    count += 1
        if U_max is not None:
            for i in range(U_max.shape[0]):
                if np.isfinite(U_max[i]):
                    count += 1

        if Y_min is not None:
            for i in range(Y_min.shape[0]):
                if np.isfinite(Y_min[i]):
                    count += 1
        if Y_max is not None:
            for i in range(Y_max.shape[0]):
                if np.isfinite(Y_max[i]):
                    count += 1

        if self.number_of_constraints is not None and \
                (self.number_of_constraints != count):
            raise ValueError(
                "The number of constraints does not match the expected count.")

        return count

    def initialize_constraints(self, delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                               U_min: np.ndarray = None, U_max: np.ndarray = None,
                               Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        self.count_constraints(self,
                               delta_U_min=delta_U_min,
                               delta_U_max=delta_U_max,
                               U_min=U_min,
                               U_max=U_max,
                               Y_min=Y_min,
                               Y_max=Y_max
                               )

        M = np.zeros((self.number_of_constraints, self.number_of_variables))
        gamma = np.zeros((self.number_of_constraints, 1))

        M_delta_U = np.vstack((-np.eye(self.number_of_variables),
                               np.eye(self.number_of_variables)))

        gamma_delta_U = np.vstack((-delta_U_min, delta_U_max))
