import numpy as np
import pandas as pd

class LinearProgrammingProblem:
    def __init__(self, num_vars, num_cons, is_min, obj_coeffs, constraint_matrix, constraint_rhs, constraint_signs, variable_signs):
        self.num_vars = num_vars
        self.num_cons = num_cons
        self.is_min = is_min
        self.obj_coeffs = obj_coeffs
        self.constraint_matrix = constraint_matrix
        self.constraint_rhs = constraint_rhs
        self.constraint_signs = constraint_signs
        self.variable_signs = variable_signs
        self.num_new_vars = 0

    def solve(self):
        self.convert_to_standard_form()
        tableau = np.zeros((self.num_cons + 1, self.num_vars + self.num_cons + 1))
        tableau = self.initialize_tableau(tableau)
        check = 0
        algorithm_choice = self.choose_algorithm()
        tableau_list = []
        algo_name = ""

        if algorithm_choice == 0:
            tableau, check, tableau_list = self.dantzig_method(tableau, tableau_list)
            algo_name = "Dantzig method"
        elif algorithm_choice == 1:
            tableau, check, tableau_list = self.bland_method(tableau, tableau_list)
            algo_name = "Bland method"
        else:
            tableau, check, tableau_list = self.two_phase_method(tableau, tableau_list)
            algo_name = "Two-phase method"

        # Build a textual log of iterations
        output = ""
        output += "Method: " + algo_name + "\n\n"

        for i, arr in enumerate(tableau_list):
            output += f"Iteration {i+1}:\n"
            for row in arr:
                output += "  " + str(row) + "\n"
            output += "\n"

        result_text = self.process_output(tableau, check)
        output += result_text + "\n"

        print("Tableau history:")
        for t in tableau_list:
            print(t)
        return output

    def display(self):
        objective_function = ""
        if self.is_min:
            objective_function = "Min z = "
        else:
            objective_function = "Max z = "

        for j in range(self.num_vars):
            coefficient = self.obj_coeffs[j]
            if coefficient >= 0 and j != 0:
                objective_function += " + " + str(coefficient) + "x" + str(j + 1)
            else:
                objective_function += str(coefficient) + "x" + str(j + 1)
        print(objective_function)

        for i in range(self.num_cons):
            constraint = ""
            for j in range(self.num_vars):
                coefficient = self.constraint_matrix[i, j]
                if coefficient >= 0 and j != 0:
                    constraint += " + " + str(coefficient) + "x" + str(j + 1)
                else:
                    constraint += str(coefficient) + "x" + str(j + 1)
            if self.constraint_signs[i] == 1:
                constraint += " >= "
            elif self.constraint_signs[i] == 0:
                constraint += " = "
            else:
                constraint += " <= "
            constraint += str(self.constraint_rhs[i])
            print(constraint)

        for j in range(self.num_vars):
            variable = "x" + str(j + 1)
            if self.variable_signs[j] == 1:
                variable += " >= 0"
            elif self.variable_signs[j] == -1:
                variable += " <= 0"
            else:
                break
            print(variable)

    def convert_to_standard_form(self):
        # Objective Function: convert max to min by negation
        if not self.is_min:
            self.obj_coeffs = -self.obj_coeffs

        # Handle variable signs
        for i in range(self.num_vars - self.num_new_vars):
            if self.variable_signs[i] == -1:
                self.obj_coeffs[i] = -self.obj_coeffs[i]
                self.constraint_matrix[:, i] = -self.constraint_matrix[:, i]
            elif self.variable_signs[i] == 0:
                # free variable: replace x_i with (x_i' - x_i''), both >= 0
                self.num_vars += 1
                self.num_new_vars += 1
                self.variable_signs = np.append(self.variable_signs, 0)
                self.obj_coeffs = np.append(self.obj_coeffs, -self.obj_coeffs[i])
                self.constraint_matrix = np.concatenate(
                    (self.constraint_matrix, -np.array([self.constraint_matrix[:, i]]).T),
                    axis=1
                )

        # Handle constraint signs
        for i in range(self.num_cons):
            if self.constraint_signs[i] == 1:  # '>='
                self.constraint_matrix[i] = -self.constraint_matrix[i]
                self.constraint_rhs[i] = -self.constraint_rhs[i]
                self.constraint_signs[i] = -1  # now '<='
            elif self.constraint_signs[i] == 0:  # '='
                # add a duplicate with reverse sign to enforce equality
                self.num_cons += 1
                self.constraint_signs[i] = -1
                self.constraint_signs = np.append(self.constraint_signs, -1)
                self.constraint_matrix = np.concatenate(
                    (self.constraint_matrix, -np.array([self.constraint_matrix[i]])),
                    axis=0
                )
                self.constraint_rhs = np.append(self.constraint_rhs, -self.constraint_rhs[i])

    def print_table(self, tableau):
        print(tableau)

    def choose_pivot_dantzig(self, tableau, xPivot, yPivot, phase1):
        minC = 0
        yPivot = -1
        for i in range(tableau.shape[1] - 1):
            if (tableau[0, i] < 0) and (tableau[0, i] < minC):
                minC = tableau[0, i]
                yPivot = i
        if yPivot == -1:
            return xPivot, yPivot, 0
        xPivot = self.find_arg_min_ratio(tableau, yPivot, phase1)
        if xPivot == -1:
            return xPivot, yPivot, -1
        return xPivot, yPivot, 1

    def dantzig_method(self, tableau, tableau_list, phase1=False):
        xPivot, yPivot = -1, -1
        while True:
            # (1) Tìm pivot column & row
            xPivot, yPivot, check = self.choose_pivot_dantzig(tableau, xPivot, yPivot, phase1)
            if check != 1:
                # Nếu không còn pivot hợp lệ, append bảng cuối cùng và return
                tableau_list.append(np.copy(tableau))
                return tableau, -check, tableau_list

            # (2) Thực hiện pivot (chỉ log 1 snapshot sau khi quay xong)
            tableau, xPivot, yPivot = self.rotate_pivot(tableau, xPivot, yPivot)
            tableau_list.append(np.copy(tableau))

    def choose_pivot_bland(self, tableau, xPivot, yPivot):
        yPivot = -1
        for i in range(tableau.shape[1] - 1):
            if tableau[0, i] < 0:
                yPivot = i
                break
        if yPivot == -1:
            return xPivot, yPivot, 0
        xPivot = self.find_arg_min_ratio(tableau, yPivot, False)
        if xPivot == -1:
            return xPivot, yPivot, -1
        return xPivot, yPivot, 1

    def bland_method(self, tableau, tableau_list):
        xPivot, yPivot = -1, -1
        while True:
            # (1) Tìm pivot column & row theo Bland
            xPivot, yPivot, check = self.choose_pivot_bland(tableau, xPivot, yPivot)
            if check != 1:
                tableau_list.append(np.copy(tableau))
                return tableau, -check, tableau_list

            # (2) Thực hiện pivot, chỉ append sau khi xoay xong
            tableau, xPivot, yPivot = self.rotate_pivot(tableau, xPivot, yPivot)
            tableau_list.append(np.copy(tableau))

    def find_pivot_column(self, tableau, col):
        xPivot = -1
        flag = False
        for i in range(1, tableau.shape[0]):
            if tableau[i, col] == 0:
                continue
            if tableau[i, col] == 1:
                if not flag:
                    xPivot = i
                    flag = True
                else:
                    return -1
            else:
                return -1
        return xPivot

    def two_phase_method(self, tableau, tableau_list):
        # Phase 1: Introduce artificial variable x0
        tableauP1 = np.zeros((tableau.shape[0], tableau.shape[1] + 1))
        tableauP1[0, -2] = 1
        tableauP1[1:, -2] = -np.ones((tableau.shape[0] - 1, 1)).ravel()
        tableauP1[1:, :tableau.shape[1] - 1] = tableau[1:, :tableau.shape[1] - 1]
        tableauP1[1:, -1] = tableau[1:, -1]

        xPivot, yPivot = -1, tableauP1.shape[1] - 2
        minB = 0
        for i in range(tableauP1.shape[0]):
            if tableauP1[i, yPivot] < minB:
                minB = tableauP1[i, yPivot]
                xPivot = i

        tableauP1, xPivot, yPivot = self.rotate_pivot(tableauP1, xPivot, yPivot)
        tableau_list.append(np.copy(tableauP1))
        tableauP1, check, tableau_list = self.dantzig_method(tableauP1, tableau_list, phase1=True)

        # If Phase 1 objective != 0, no feasible solution
        for j in range(tableauP1.shape[1] - 2):
            if tableauP1[0, j] != 0:
                tableau_list.append(np.copy(tableau))
                return tableau, -1, tableau_list

        # Phase 2: Remove artificial variable, restore original tableau structure
        tableau[1:, :tableau.shape[1] - 1] = tableauP1[1:, :tableau.shape[1] - 1]
        tableau[1:, -1] = tableauP1[1:, -1]

        for j in range(tableau.shape[1]):
            xPivot = self.find_pivot_column(tableau, j)
            if xPivot == -1:
                continue
            tableau, xPivot, j = self.rotate_pivot(tableau, xPivot, j)
            tableau_list.append(np.copy(tableau))

        tableau, check, tableau_list = self.dantzig_method(tableau, tableau_list)
        tableau_list.append(np.copy(tableau))
        return tableau, check, tableau_list

    def check_unique_solution(self, tableau, pivots):
        for i in range(tableau.shape[1] - 1):
            if (i >= self.num_vars - self.num_new_vars) and (i < self.num_vars):
                continue
            if ((pivots[i] == -1) and (abs(tableau[0, i]) < 1e-6)) and (self.variable_signs[i] != 0):
                return False
        return True

    def find_variable_name(self, tableau, index):
        if index < self.num_vars - self.num_new_vars:
            return 1, "x" + str(index + 1)
        elif (index + 1 > self.num_vars) and (index + 1 < tableau.shape[1]):
            return 1, "w" + str(index + 1 - self.num_vars)
        return 0, ""

    def initialize_tableau(self, tableau):
        tableau[0, :self.num_vars] = self.obj_coeffs
        tableau[0, -1] = 0
        tableau[1:, :self.num_vars] = self.constraint_matrix
        tableau[1:, self.num_vars:-1] = np.identity(self.num_cons)
        tableau[1:, -1] = self.constraint_rhs

        self.variable_signs = np.append(self.variable_signs, np.ones(self.num_cons))
        return tableau

    def process_output(self, tableau, result):
        print("Tableau:", tableau)
        print("Result code:", result)

        output = "===== RESULT =====\n\n"
        if result == 1:
            if self.is_min:
                output += "=> The problem is UNBOUNDED. MIN z = -infinity\n"
            else:
                output += "=> The problem is UNBOUNDED. MAX z = +infinity\n"
        elif result == 0:
            if self.is_min:
                output += f"MIN z = {-tableau[0, -1]}\n"
            else:
                output += f"MAX z = {tableau[0, -1]}\n"

            pivots = np.array([self.find_pivot_column(tableau, i) for i in range(tableau.shape[1] - 1)])
            if self.check_unique_solution(tableau, pivots):
                output += "=> UNIQUE SOLUTION. The optimal solution is:\n"
                for j in range(self.num_vars - self.num_new_vars):
                    if tableau[0, j] != 0:
                        output += f"x{j + 1} = 0\n"
                        continue
                    count = 0
                    index = 0
                    for i in range(1, tableau.shape[0]):
                        if tableau[i, j] != 0:
                            count += 1
                            index = i
                    if self.variable_signs[j] == -1:
                        output += f"x{j + 1} = {-tableau[index, -1]}\n"
                    else:
                        output += f"x{j + 1} = {tableau[index, -1]}\n"
            else:
                output += "=> MULTIPLE SOLUTIONS. The optimal solution set is:\n"
                sign = np.array([
                    -1 if ((self.variable_signs[i] < 0) and (i < self.num_vars - self.num_new_vars)) else 1
                    for i in range(tableau.shape[1] - 1)
                ])
                for i in range(self.num_vars - self.num_new_vars):
                    if pivots[i] == -1:
                        if abs(tableau[0, i]) > 1e-4:
                            output += f"x{i + 1} = 0\n"
                        else:
                            if self.variable_signs[i] == 0:
                                output += f"x{i + 1} is free\n"
                            elif self.variable_signs[i] == 1:
                                output += f"x{i + 1} >= 0\n"
                            else:
                                output += f"x{i + 1} <= 0\n"
                    else:
                        root = f"x{i + 1} = {sign[i] * tableau[pivots[i], -1]}"
                        for j in range(tableau.shape[1] - 1):
                            if (abs(tableau[0, j]) > 1e-4) or (pivots[j] != -1) or (j == i):
                                continue
                            check_root, name = self.find_variable_name(tableau, j)
                            if check_root == 0:
                                continue
                            coeff = -sign[i] * sign[j] * tableau[pivots[i], j]
                            if coeff == 0:
                                continue
                            if coeff > 0:
                                root += f" + {coeff}{name}"
                            else:
                                root += f" {coeff}{name}"
                        output += root + "\n"
                output += "Subject to:\n"
                for i in range(tableau.shape[1] - 1):
                    if (i >= self.num_vars - self.num_new_vars) and (i < self.num_vars):
                        continue
                    if (i < self.num_vars - self.num_new_vars) and (self.variable_signs[i] == 0):
                        continue
                    if pivots[i] == -1:
                        if i < self.num_vars - self.num_new_vars:
                            continue
                        if abs(tableau[0, i]) < 1e-4:
                            check_root, name = self.find_variable_name(tableau, i)
                            if check_root == 1:
                                if i >= self.num_vars - self.num_new_vars:
                                    output += f"{name} >= 0\n"
                                else:
                                    if self.variable_signs[i] == 0:
                                        output += f"{name} is free\n"
                                    elif self.variable_signs[i] < 0:
                                        output += f"{name} <= 0\n"
                                    else:
                                        output += f"{name} >= 0\n"
                    else:
                        root = f"{sign[i] * tableau[pivots[i], -1]}"
                        for j in range(tableau.shape[1] - 1):
                            if (abs(tableau[0, j]) > 1e-4) or (pivots[j] != -1):
                                continue
                            check_root, name = self.find_variable_name(tableau, j)
                            if check_root == 0:
                                continue
                            coeff = -sign[i] * sign[j] * tableau[pivots[i], j]
                            if coeff == 0:
                                continue
                            if coeff > 0:
                                root += f" + {coeff}{name}"
                            else:
                                root += f" {coeff}{name}"
                        root += " >= 0"
                        output += root + "\n"
        else:
            output += "NO SOLUTION\n"
        print(output)
        return output

    def choose_algorithm(self):
        flag = 0
        for i in range(self.num_cons):
            if self.constraint_rhs[i] == 0:  # Bland
                flag = 1
            if self.constraint_rhs[i] < 0:  # Two-phase
                return 2
        return flag

    def rotate_pivot(self, tableau, xPivot, yPivot):
        """
        Rút gọn pivot column một lần duy nhất:
         - Chia pivot row để hệ số ở (xPivot,yPivot) = 1.
         - Khử cột yPivot khỏi tất cả các hàng khác.
        Trả về tableau mới và (xPivot,yPivot).
        """
        # (1) Chia pivot row
        pivot_value = tableau[xPivot, yPivot]
        tableau[xPivot, :] = tableau[xPivot, :] / pivot_value

        # (2) Loại pivot column khỏi các hàng khác
        for i in range(tableau.shape[0]):
            if i == xPivot:
                continue
            factor = tableau[i, yPivot]
            tableau[i, :] -= factor * tableau[xPivot, :]

        return tableau, xPivot, yPivot

    def find_arg_min_ratio(self, tableau, yPivot, phase1):
        xPivot = -1
        minRatio = -1
        for i in range(tableau.shape[0]):
            if tableau[i, yPivot] > 0:
                minRatio = tableau[i, -1] / tableau[i, yPivot]
                xPivot = i
                break
        if xPivot == -1:
            return -1
        for i in range(1, tableau.shape[0]):
            if tableau[i, yPivot] > 0:
                ratio = tableau[i, -1] / tableau[i, yPivot]
                if ratio < minRatio:
                    minRatio = ratio
                    xPivot = i
                if phase1 and (ratio == minRatio) and (tableau[i, -2] == 1):
                    xPivot = i
        return xPivot
