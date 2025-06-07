import numpy as np

MAX_ITERATION_DEFAULT = 100
TOL_DEFAULT = 1e-8


class ActiveSet:
    """
    A class that manages an active set of constraints with a
    fixed number of constraints, ensuring safe management of the active set information.

    Attributes
    ----------
    active_flags : np.ndarray of bool
        An array indicating whether each constraint is active (length: number of constraints).
    active_indices : np.ndarray of int
        An array storing the indices of active constraints
        (length: number of constraints, unused parts are set to 0, etc.).
    number_of_active : int
        The current number of active constraints.
    """

    def __init__(self, number_of_constraints: int):
        self.number_of_constraints = number_of_constraints

        self._active_flags = np.zeros(number_of_constraints, dtype=bool)
        self._active_indices = np.zeros(number_of_constraints, dtype=int)
        self._number_of_active = 0

    def push_active(self, index: int):
        if not self._active_flags[index]:
            self._active_flags[index] = True
            self._active_indices[self._number_of_active] = index
            self._number_of_active += 1

    def push_inactive(self, index: int):
        if self._active_flags[index]:
            self._active_flags[index] = False
            found = False
            for i in range(self._number_of_active):
                if not found and self._active_indices[i] == index:
                    found = True
                if found and i < self._number_of_active - 1:
                    self._active_indices[i] = self._active_indices[i + 1]
            if found:
                self._active_indices[self._number_of_active - 1] = 0
                self._number_of_active -= 1

    def get_active(self, index: int):
        if index < 0 or index >= self._number_of_active:
            raise IndexError("Index out of bounds for active set.")
        return self._active_indices[index]

    def get_active_indices(self):
        return self._active_indices

    def get_number_of_active(self):
        return self._number_of_active

    def is_active(self, index: int):
        return self._active_flags[index]


class QP_ActiveSetSolver:
    """
    Quadratic Programming (QP) solver using the Active Set method.

    Problem: minimize (1/2) X^T E X - X^T L  subject to  M X <= gamma.

    E, L, M, gamma : Parameters of the above QP (numpy arrays)
    max_iteration: Maximum number of iterations (limit)
    tol     : Tolerance for numerical errors (used for constraint violation and negative lambda checks)
    X        : Solution vector estimated as optimal
    active_set   : List of indices of constraints that were active at the end
    """

    def __init__(self, number_of_variables, number_of_constraints,
                 X: np.ndarray = None, active_set: ActiveSet = None,
                 max_iteration=MAX_ITERATION_DEFAULT, tol=TOL_DEFAULT):

        self.number_of_variables = number_of_variables
        self.number_of_constraints = number_of_constraints

        self.X = X
        self.active_set = active_set
        self.max_iteration = max_iteration
        self.tol = tol
        self.iteration_count = 0

        self.KKT = np.zeros((number_of_variables + number_of_constraints,
                             number_of_variables + number_of_constraints))
        self.rhs = np.zeros((number_of_variables + number_of_constraints, 1))

    def update_E(self, E: np.ndarray):
        m = self.number_of_variables

        self.KKT[:m, :m] = E

    def update_L(self, L: np.ndarray):
        m = self.number_of_variables

        self.rhs[:m, 0] = L.flatten()

    def _set_KKT(self, E: np.ndarray = None, M: np.ndarray = None):

        if E is None and M is None:
            return

        m = self.number_of_variables

        if E is not None:
            self.update_E(E)

        for i in range(self.active_set.get_number_of_active()):
            index = self.active_set.get_active(i)
            self.KKT[:m, m + i] = M[index, :].T
            self.KKT[m + i, :m] = M[index, :]

    def _set_rhs(self, L: np.ndarray = None, gamma: np.ndarray = None):

        if L is None and gamma is None:
            return

        m = self.number_of_variables

        if L is not None:
            self.update_L(L)

        k = self.active_set.get_number_of_active()

        for i in range(k):
            index = self.active_set.get_active(i)
            self.rhs[m + i, 0] = gamma[index, 0]

    def _solve_KKT_inv(self, k) -> np.ndarray:
        m = self.number_of_variables

        KKT = self.KKT[:(m + k), :(m + k)]
        rhs = self.rhs[:(m + k), :]

        sol = np.linalg.solve(KKT, rhs)

        return sol

    def solve_no_constrained_X(self,
                               E: np.ndarray = None,
                               L: np.ndarray = None):

        m = self.number_of_variables

        if E is None and L is None:
            return self.X

        if E is None:
            E_matrix = self.KKT[:m, :m]
        else:
            E_matrix = E

        if L is None:
            L_matrix = self.rhs[:m, 0]
        else:
            L_matrix = L

        return np.linalg.solve(E_matrix, L_matrix)

    def initialize_X(self, E: np.ndarray, L: np.ndarray,
                     M: np.ndarray, gamma: np.ndarray):

        n = E.shape[0]

        if 0 == self.active_set.get_number_of_active():
            # Use the unconstrained optimal solution as the initial point
            try:
                self.X = self.solve_no_constrained_X(E, L)
            except np.linalg.LinAlgError:
                self.X = np.zeros(self.number_of_variables)
        else:
            # If initial active constraints are specified, initialize the solution
            k = self.active_set.get_number_of_active()

            self._set_KKT(E, M)
            self._set_rhs(L, gamma)

            sol = self._solve_KKT_inv(k)
            self.X = sol[:n]

    def solve(self,
              E: np.ndarray = None, L: np.ndarray = None,
              M: np.ndarray = None, gamma: np.ndarray = None) -> np.ndarray:
        # check compatibility
        if E is not None:
            if E.shape[0] != self.number_of_variables or \
                    E.shape[1] != self.number_of_variables:
                raise ValueError(
                    "E must be a square matrix of size (n, n) where n is the number of variables.")

        if L is not None:
            if L.shape[0] != self.number_of_variables or L.shape[1] != 1:
                raise ValueError(
                    "L must be a column vector of size (n, 1) where n is the number of variables.")

        if M is None or gamma is None:
            raise ValueError(
                "M and gamma must be provided for solving QP.")

        if M.shape[1] != self.number_of_variables or M.shape[0] != self.number_of_constraints:
            raise ValueError(
                "M must be a matrix of size (m, n) where m is the number of constraints and n is the number of variables.")

        if gamma.shape[0] != self.number_of_constraints or gamma.shape[1] != 1:
            raise ValueError(
                "gamma must be a column vector of size (m, 1) where m is the number of constraints.")

        # Initialize
        if self.active_set is None:
            self.active_set = ActiveSet(self.number_of_constraints)

        if self.X is None:
            self.initialize_X(E, L, M, gamma)

        # Main iterative loop
        lambda_values = np.zeros(self.number_of_constraints)
        lambda_candidate_exists = False

        for iteration_count in range(self.max_iteration):
            # Build and solve the KKT system for the active constraints A
            k = self.active_set.get_number_of_active()
            if k == 0:
                # If there are no active constraints, simply solve E X = L
                X_candidate = self.solve_no_constrained_X(E, L)
                lambda_candidate_exists = False

            else:
                m = self.number_of_variables

                self._set_KKT(E, M)
                self._set_rhs(L, gamma)

                sol = self._solve_KKT_inv(k)
                X_candidate = sol[:m]
                lambda_candidate = sol[m:]
                lambda_candidate_exists = True

            # (1) Check constraint violations for the candidate solution
            violation_index = -1
            max_violation = 0.0
            M_X = M @ X_candidate

            for j in range(self.number_of_constraints):
                gamma_tol = gamma[j] + self.tol
                if M_X[j, 0] > gamma_tol:

                    M_X_gamma = M_X[j, 0] - gamma[j]
                    if M_X_gamma > max_violation:
                        max_violation = M_X_gamma
                        violation_index = j

            if violation_index >= 0:
                self.active_set.push_active(violation_index)

                # Since a constraint was added, re-optimize in the next loop
                self.X = X_candidate
                continue

            # (2) All constraints are satisfied -> Check lambda
            if self.active_set.get_number_of_active() > 0:
                # Find negative lambda among the active constraints
                min_lambda_index = -1
                min_lambda_val = 0.0

                if lambda_candidate_exists:
                    for index_local, lam in enumerate(lambda_candidate):
                        if lam < -self.tol and lam < min_lambda_val:
                            min_lambda_val = lam
                            min_lambda_index = index_local

                if min_lambda_index >= 0:
                    self.active_set.push_inactive(min_lambda_index)
                    # Since a constraint was removed, re-optimize
                    self.X = X_candidate
                    continue

            # If there are no constraint violations and all lambda are non-negative,
            # consider as optimal solution
            self.X = X_candidate

            for j in range(self.number_of_constraints):
                if self.active_set.is_active(j):
                    self.active_set.push_inactive(j)

            break

        self.iteration_count = iteration_count + 1

        return self.X
