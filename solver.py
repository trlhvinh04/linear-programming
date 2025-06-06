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
        output += "## Method: " + algo_name + "\n\n"

        for i, arr in enumerate(tableau_list):
            output += f"### Iteration {i+1}:\n\n"
            
            # Create empty headers for proper markdown table formatting
            num_cols = arr.shape[1]
            
            # Create empty header row
            output += "|" + "|".join([" " for _ in range(num_cols)]) + "|\n"
            
            # Create separator row
            output += "|" + "|".join([":---:" for _ in range(num_cols)]) + "|\n"
            
            # Create data rows
            for row_idx, row in enumerate(arr):
                output += "|"
                for element in row:
                    # Format numbers to avoid very long decimals
                    if isinstance(element, (int, float)):
                        if abs(element) < 1e-10:  # Treat very small numbers as 0
                            formatted_element = "0"
                        elif element == int(element):  # If it's effectively an integer
                            formatted_element = str(int(element))
                        else:
                            formatted_element = f"{element:.3f}"  # 3 decimal places for better readability
                    else:
                        formatted_element = str(element)
                    output += f" {formatted_element} |"
                output += "\n"
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
            objective_function = "$\\min z = "
        else:
            objective_function = "$\\max z = "

        for j in range(self.num_vars):
            coefficient = self.obj_coeffs[j]
            if j == 0:
                if coefficient == 1:
                    objective_function += f"x_{{{j + 1}}}"
                elif coefficient == -1:
                    objective_function += f"-x_{{{j + 1}}}"
                else:
                    objective_function += f"{coefficient}x_{{{j + 1}}}"
            else:
                if coefficient > 0:
                    if coefficient == 1:
                        objective_function += f" + x_{{{j + 1}}}"
                    else:
                        objective_function += f" + {coefficient}x_{{{j + 1}}}"
                elif coefficient < 0:
                    if coefficient == -1:
                        objective_function += f" - x_{{{j + 1}}}"
                    else:
                        objective_function += f" {coefficient}x_{{{j + 1}}}"
        objective_function += "$"
        print(objective_function)

        print("Subject to:")
        for i in range(self.num_cons):
            constraint = "$"
            for j in range(self.num_vars):
                coefficient = self.constraint_matrix[i, j]
                if j == 0:
                    if coefficient == 1:
                        constraint += f"x_{{{j + 1}}}"
                    elif coefficient == -1:
                        constraint += f"-x_{{{j + 1}}}"
                    elif coefficient != 0:
                        constraint += f"{coefficient}x_{{{j + 1}}}"
                else:
                    if coefficient > 0:
                        if coefficient == 1:
                            constraint += f" + x_{{{j + 1}}}"
                        else:
                            constraint += f" + {coefficient}x_{{{j + 1}}}"
                    elif coefficient < 0:
                        if coefficient == -1:
                            constraint += f" - x_{{{j + 1}}}"
                        else:
                            constraint += f" {coefficient}x_{{{j + 1}}}"
            
            if self.constraint_signs[i] == 1:
                constraint += f" \\geq "
            elif self.constraint_signs[i] == 0:
                constraint += f" = "
            else:
                constraint += f" \\leq "
            constraint += f"{self.constraint_rhs[i]}$"
            print(constraint)

        for j in range(self.num_vars):
            variable = f"$x_{{{j + 1}}}"
            if self.variable_signs[j] == 1:
                variable += f" \\geq 0$"
            elif self.variable_signs[j] == -1:
                variable += f" \\leq 0$"
            else:
                variable += "$ is free"
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
        # -----------------------
        # 1) PHASE 1 (Artificial)
        # -----------------------
        phase1_history = []

        tableauP1 = np.zeros((tableau.shape[0], tableau.shape[1] + 1))
        tableauP1[0, -2] = 1.0
        tableauP1[1:, -2] = -np.ones((tableau.shape[0] - 1,))
        tableauP1[1:, : tableau.shape[1] - 1] = tableau[1:, : tableau.shape[1] - 1]
        tableauP1[1:, -1] = tableau[1:, -1]

        xPivot, yPivot = -1, tableauP1.shape[1] - 2
        minB = 0.0
        for i in range(tableauP1.shape[0]):
            if tableauP1[i, yPivot] < minB:
                minB = tableauP1[i, yPivot]
                xPivot = i

        if xPivot != -1:
            tableauP1, xPivot, yPivot = self.rotate_pivot(tableauP1, xPivot, yPivot)
            if not phase1_history or not np.allclose(phase1_history[-1], tableauP1):
                phase1_history.append(np.copy(tableauP1))

            tableauP1, check1, phase1_history = self.dantzig_method(
                tableauP1, phase1_history, phase1=True
            )

            for j in range(tableauP1.shape[1] - 2):
                if abs(tableauP1[0, j]) > 1e-8:
                    phase1_history.append(np.copy(tableau))
                    tableau_list.extend(phase1_history)
                    return tableau, -1, tableau_list

        # Nếu xPivot == -1, Phase 1 đã khả thi ngay (không cần pivot ban đầu).

        # -----------------------------
        # 2) PHASE 2 (Pivot bình thường)
        # -----------------------------
        phase2_history = []

        # Gán từ tableauP1 trở về tableau (giữ nguyên shape)
        tableau[1:, : tableau.shape[1] - 1] = tableauP1[1:, : tableau.shape[1] - 1]
        tableau[1:, -1] = tableauP1[1:, -1]

        # Phase 2 snapshot đầu tiên
        if not phase2_history or not np.allclose(phase2_history[-1], tableau):
            phase2_history.append(np.copy(tableau))

        # Pivot qua cột 0..(shape[1]-2)
        for j in range(tableau.shape[1] - 1):
            xPivot = self.find_pivot_column(tableau, j)
            if xPivot == -1:
                continue
            tableau, xPivot, j = self.rotate_pivot(tableau, xPivot, j)
            if not phase2_history or not np.allclose(phase2_history[-1], tableau):
                phase2_history.append(np.copy(tableau))
            else:
                break  # dừng hẳn Phase 2 nếu lặp bản sao

        # Kiểm tra xem có còn pivot Dantzig ở cuối không
        _, _, check2 = self.choose_pivot_dantzig(tableau, -1, -1, phase1=False)
        if check2 == 1:
            tableau, check3, phase2_history = self.dantzig_method(tableau, phase2_history)

        # Bây giờ ta chỉ “làm sạch” Phase 2_history (tất cả đều có shape 4×6)
        phase2_clean = self._remove_consecutive_duplicates(phase2_history)

        # Cuối cùng ghép Phase 1 + Phase 2 đã “làm sạch” vào tableau_list
        phase1_clean = self._remove_consecutive_duplicates(phase1_history)

        tableau_list.extend(phase1_clean)
        tableau_list.extend(phase2_clean)

        return tableau, (check3 if 'check3' in locals() else 0), tableau_list


    def _remove_consecutive_duplicates(self, history_list):
        """
        Trả về một danh sách tương tự history_list nhưng loại bỏ
        tất cả các phần tử mà nó giống hệt phần tử ngay trước đó.
        """
        if not history_list:
            return []

        cleaned = [history_list[0]]
        for tbl in history_list[1:]:
            if not np.allclose(cleaned[-1], tbl):
                cleaned.append(tbl)
        return cleaned
    
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
        output = "## RESULT\n\n"
        if result == 1:
            if self.is_min:
                output += "=> The problem is UNBOUNDED. $\\min z = -\\infty$\n"
            else:
                output += "=> The problem is UNBOUNDED. $\\max z = +\\infty$\n"
        elif result == 0:
            if self.is_min:
                output += f"$\\min z = {-tableau[0, -1]}$\n"
            else:
                output += f"$\\max z = {tableau[0, -1]}$\n"

            pivots = np.array([self.find_pivot_column(tableau, i) for i in range(tableau.shape[1] - 1)])
            if self.check_unique_solution(tableau, pivots):
                output += "=> UNIQUE SOLUTION. The optimal solution is:\n"
                for j in range(self.num_vars - self.num_new_vars):
                    if tableau[0, j] != 0:
                        output += f"$x_{{{j + 1}}} = 0$\n"
                        continue
                    count = 0
                    index = 0
                    for i in range(1, tableau.shape[0]):
                        if tableau[i, j] != 0:
                            count += 1
                            index = i
                    if self.variable_signs[j] == -1:
                        output += f"$x_{{{j + 1}}} = {-tableau[index, -1]}$\n"
                    else:
                        output += f"$x_{{{j + 1}}} = {tableau[index, -1]}$\n"
            else:
                output += "=> MULTIPLE SOLUTIONS. The optimal solution set is:\n"
                sign = np.array([
                    -1 if ((self.variable_signs[i] < 0) and (i < self.num_vars - self.num_new_vars)) else 1
                    for i in range(tableau.shape[1] - 1)
                ])
                for i in range(self.num_vars - self.num_new_vars):
                    if pivots[i] == -1:
                        if abs(tableau[0, i]) > 1e-4:
                            output += f"$x_{{{i + 1}}} = 0$\n"
                        else:
                            if self.variable_signs[i] == 0:
                                output += f"$x_{{{i + 1}}}$ is free\n"
                            elif self.variable_signs[i] == 1:
                                output += f"$x_{{{i + 1}}} \\geq 0$\n"
                            else:
                                output += f"$x_{{{i + 1}}} \\leq 0$\n"
                    else:
                        root = f"$x_{{{i + 1}}} = {sign[i] * tableau[pivots[i], -1]}"
                        for j in range(tableau.shape[1] - 1):
                            if (abs(tableau[0, j]) > 1e-4) or (pivots[j] != -1) or (j == i):
                                continue
                            check_root, name = self.find_variable_name(tableau, j)
                            if check_root == 0:
                                continue
                            coeff = -sign[i] * sign[j] * tableau[pivots[i], j]
                            if coeff == 0:
                                continue
                            # Convert variable name to LaTeX format
                            latex_name = name.replace("x", "x_").replace("w", "w_")
                            if len(latex_name) > 2:  # Has subscript
                                latex_name = latex_name.replace("_", "_{") + "}"
                            if coeff > 0:
                                if coeff == 1:
                                    root += f" + {latex_name}"
                                else:
                                    root += f" + {coeff}{latex_name}"
                            else:
                                if coeff == -1:
                                    root += f" - {latex_name}"
                                else:
                                    root += f" {coeff}{latex_name}"
                        output += root + "$\n"
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
                                # Convert variable name to LaTeX format
                                latex_name = name.replace("x", "x_").replace("w", "w_")
                                if len(latex_name) > 2:  # Has subscript
                                    latex_name = latex_name.replace("_", "_{") + "}"
                                if i >= self.num_vars - self.num_new_vars:
                                    output += f"${latex_name} \\geq 0$\n"
                                else:
                                    if self.variable_signs[i] == 0:
                                        output += f"${latex_name}$ is free\n"
                                    elif self.variable_signs[i] < 0:
                                        output += f"${latex_name} \\leq 0$\n"
                                    else:
                                        output += f"${latex_name} \\geq 0$\n"
                    else:
                        root = f"${sign[i] * tableau[pivots[i], -1]}"
                        for j in range(tableau.shape[1] - 1):
                            if (abs(tableau[0, j]) > 1e-4) or (pivots[j] != -1):
                                continue
                            check_root, name = self.find_variable_name(tableau, j)
                            if check_root == 0:
                                continue
                            coeff = -sign[i] * sign[j] * tableau[pivots[i], j]
                            if coeff == 0:
                                continue
                            # Convert variable name to LaTeX format
                            latex_name = name.replace("x", "x_").replace("w", "w_")
                            if len(latex_name) > 2:  # Has subscript
                                latex_name = latex_name.replace("_", "_{") + "}"
                            if coeff > 0:
                                if coeff == 1:
                                    root += f" + {latex_name}"
                                else:
                                    root += f" + {coeff}{latex_name}"
                            else:
                                if coeff == -1:
                                    root += f" - {latex_name}"
                                else:
                                    root += f" {coeff}{latex_name}"
                        root += " \\geq 0$"
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
