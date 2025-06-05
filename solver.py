import numpy as np

class LinearProgrammingProblem:
    def __init__(self, num_vars, num_cons, is_min,
                 obj_coeffs, constraint_matrix,
                 constraint_rhs, constraint_signs,
                 variable_signs):
        self.num_vars = num_vars
        self.num_cons = num_cons
        self.is_min = is_min
        self.obj_coeffs = obj_coeffs.astype(float)
        self.constraint_matrix = constraint_matrix.astype(float)
        self.constraint_rhs = constraint_rhs.astype(float)
        self.constraint_signs = constraint_signs.astype(int)
        self.variable_signs = variable_signs.astype(int)
        self.num_new_vars = 0

    def solve(self):
        # Chuyển sang dạng chuẩn
        self.convert_to_standard_form()

        # Khởi tạo tableau (hàng 0: objective; các hàng sau: constraints)
        tableau = np.zeros((self.num_cons + 1,
                            self.num_vars + self.num_cons + 1),
                           dtype=float)
        tableau = self.initialize_tableau(tableau)

        # Chọn thuật toán: 0 = Dantzig, 1 = Bland, 2 = Two-phase
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

        # Biên dịch kết quả thu được thành chuỗi plain-text
        output = []
        output.append(f"===== SIMPLEX ({algo_name}) =====\n")

        for i, arr in enumerate(tableau_list):
            output.append(f"Iteration {i+1}:")
            for row in arr:
                output.append("  " + np.array2string(row, precision=6, separator=", "))
            output.append("")  # dòng trống giữa các iteration

        output.append("Final Tableau:")
        output.append(np.array2string(tableau, precision=6, separator=", "))
        output.append("")

        # Xử lý nghiệm
        result_text = self.process_output(tableau, check)
        output.append(result_text)

        # Kết hợp tất cả thành một chuỗi
        return "\n".join(output)

    def convert_to_standard_form(self):
        # 1) Nếu là Maximize, đổi sang Minimize bằng cách nhân -1 cho obj_coeffs
        if not self.is_min:
            self.obj_coeffs = -self.obj_coeffs

        # 2) Xử lý dấu của biến:
        #    Nếu variable_signs[i] = -1 (x_i <= 0), nhân -1 cột tương ứng.
        #    Nếu variable_signs[i] = 0 (free), thêm biến phụ x'_i để chuyển free thành nonnegative.
        for i in range(self.num_vars - self.num_new_vars):
            sign = self.variable_signs[i]
            if sign == -1:
                self.obj_coeffs[i] = -self.obj_coeffs[i]
                self.constraint_matrix[:, i] = -self.constraint_matrix[:, i]
            elif sign == 0:
                # đối với biến free: x_i = x_i' - x_i''  (thêm cột x_i'')
                self.num_vars += 1
                self.num_new_vars += 1
                self.variable_signs = np.append(self.variable_signs, 0)
                self.obj_coeffs = np.append(self.obj_coeffs, -self.obj_coeffs[i])
                # Thêm một cột mới bằng -cột i
                new_col = -self.constraint_matrix[:, i].reshape(-1, 1)
                self.constraint_matrix = np.hstack((self.constraint_matrix, new_col))

        # 3) Xử lý dấu của từng constraint:
        #    - Nếu dấu = (constraint_signs[i] == 0), nhân thêm hàng đối xứng
        #    - Nếu dấu ≥ (constraint_signs[i] == 1), chuyển thành ≤ bằng cách nhân -1
        for i in range(self.num_cons):
            sign = self.constraint_signs[i]
            if sign == 1:  # ≥
                self.constraint_matrix[i, :] = -self.constraint_matrix[i, :]
                self.constraint_rhs[i] = -self.constraint_rhs[i]
                self.constraint_signs[i] = -1  # bây giờ thành ≤
            elif sign == 0:  # =  (thêm một hàng đối xứng)
                # Đổi hàng i thành ≤
                self.constraint_signs[i] = -1
                # Thêm một hàng mới tương ứng với ≥ (nhân -1 để thành ≤)
                new_matrix_row = -self.constraint_matrix[i, :].reshape(1, -1)
                new_rhs = -self.constraint_rhs[i]
                self.constraint_matrix = np.vstack((self.constraint_matrix, new_matrix_row))
                self.constraint_rhs = np.append(self.constraint_rhs, new_rhs)
                self.constraint_signs = np.append(self.constraint_signs, -1)
                self.num_cons += 1

    def initialize_tableau(self, tableau):
        # Hàng 0: objective function (c^T x → đưa vào các c ở cột 0..num_vars-1)
        tableau[0, :self.num_vars] = self.obj_coeffs
        tableau[0, -1] = 0.0

        # Các hàng sau: constraint_matrix và identity slack
        tableau[1:, :self.num_vars] = self.constraint_matrix
        tableau[1:, self.num_vars:self.num_vars + self.num_cons] = np.identity(self.num_cons)
        tableau[1:, -1] = self.constraint_rhs

        # Cập nhật variable_signs để bao gồm slack variables (slack ≥ 0)
        self.variable_signs = np.append(self.variable_signs,
                                        np.ones(self.num_cons, dtype=int))
        return tableau

    def choose_algorithm(self):
        # Nếu có bất kỳ RHS nào < 0 → Two-phase
        for rhs in self.constraint_rhs:
            if rhs < 0:
                return 2
        # Nếu có RHS = 0 → Bland, nếu không → Dantzig
        for rhs in self.constraint_rhs:
            if abs(rhs) < 1e-9:
                return 1
        return 0

    def dantzig_method(self, tableau, tableau_list, phase1=False):
        while True:
            xPivot, yPivot, check = self.choose_pivot_dantzig(tableau, phase1)
            tableau_list.append(np.copy(tableau))
            if check == 1:
                tableau, xPivot, yPivot = self.rotate_pivot(tableau, xPivot, yPivot, tableau_list)
            else:
                return tableau, -check, tableau_list

    def choose_pivot_dantzig(self, tableau, phase1):
        minC = 0.0
        yPivot = -1
        # Tìm cột có giá trị < 0 (nhỏ nhất)
        for j in range(tableau.shape[1] - 1):
            if tableau[0, j] < minC:
                minC = tableau[0, j]
                yPivot = j
        if yPivot == -1:
            return -1, -1, 0  # optimal
        # Tìm hàng (tối thiểu tỉ số RHS / coefficient > 0)
        xPivot = self.find_arg_min_ratio(tableau, yPivot, phase1)
        if xPivot == -1:
            return -1, yPivot, -1  # unbounded
        return xPivot, yPivot, 1

    def bland_method(self, tableau, tableau_list):
        while True:
            xPivot, yPivot, check = self.choose_pivot_bland(tableau)
            tableau_list.append(np.copy(tableau))
            if check == 1:
                tableau, xPivot, yPivot = self.rotate_pivot(tableau, xPivot, yPivot, tableau_list)
            else:
                return tableau, -check, tableau_list

    def choose_pivot_bland(self, tableau):
        yPivot = -1
        for j in range(tableau.shape[1] - 1):
            if tableau[0, j] < 0:
                yPivot = j
                break
        if yPivot == -1:
            return -1, -1, 0  # optimal
        xPivot = self.find_arg_min_ratio(tableau, yPivot, False)
        if xPivot == -1:
            return -1, yPivot, -1  # unbounded
        return xPivot, yPivot, 1

    def find_arg_min_ratio(self, tableau, yPivot, phase1):
        xPivot = -1
        minRatio = np.inf
        for i in range(1, tableau.shape[0]):
            coeff = tableau[i, yPivot]
            if coeff > 0:
                ratio = tableau[i, -1] / coeff
                if ratio < minRatio - 1e-12 or (
                   phase1 and abs(ratio - minRatio) < 1e-12
                   and tableau[i, -2] == 1):
                    minRatio = ratio
                    xPivot = i
        return xPivot

    def rotate_pivot(self, tableau, xPivot, yPivot, tableau_list):
        pivot_val = tableau[xPivot, yPivot]
        # Chuẩn hóa hàng pivot
        tableau[xPivot, :] = tableau[xPivot, :] / pivot_val
        tableau_list.append(np.copy(tableau))
        # Cân bằng các hàng khác
        for i in range(tableau.shape[0]):
            if i != xPivot:
                factor = tableau[i, yPivot]
                tableau[i, :] -= factor * tableau[xPivot, :]
                tableau_list.append(np.copy(tableau))
        return tableau, xPivot, yPivot

    def two_phase_method(self, tableau, tableau_list):
        # Tạo một cột phụ x0 để đưa các RHS < 0 về ≥ 0
        rows, cols = tableau.shape
        tableauP1 = np.zeros((rows, cols + 1))
        # Dòng 0 (objective cho phase1)
        tableauP1[0, -2] = 1.0
        # Các hàng sau: 
        tableauP1[1:, -2] = -1.0
        tableauP1[1:, :cols-1] = tableau[1:, :cols-1]
        tableauP1[1:, -1] = tableau[1:, -1]

        # Tìm pivot ban đầu (x0) để làm sạch các hàng có RHS < 0
        xPivot = -1
        for i in range(1, tableauP1.shape[0]):
            if tableauP1[i, -2] < 0:
                xPivot = i
                break
        if xPivot != -1:
            tableauP1, xPivot, _ = self.rotate_pivot(tableauP1, xPivot,
                                                     tableauP1.shape[1] - 2,
                                                     tableau_list)
        # Chạy Dantzig trên phase1
        tableauP1, check_p1, tableau_list = self.dantzig_method(tableauP1, tableau_list, phase1=True)
        # Nếu phase1 không optimal (còn giá trị ≠ 0 trên hàng 0), vô nghiệm
        for j in range(tableauP1.shape[1] - 2):
            if abs(tableauP1[0, j]) > 1e-9:
                return tableau, -1, tableau_list  # infeasible

        # Phase2: đưa kết quả về tableau gốc
        tableau[1:, :cols-1] = tableauP1[1:, :cols-1]
        tableau[1:, -1] = tableauP1[1:, -1]
        # Sửa lại các pivot cơ bản tương ứng
        for j in range(tableau.shape[1] - 1):
            x_col = self.find_pivot_column(tableau, j)
            if x_col != -1:
                tableau, x_col, j = self.rotate_pivot(tableau, x_col, j, tableau_list)

        # Chạy Dantzig trên phase2
        tableau, check_p2, tableau_list = self.dantzig_method(tableau, tableau_list)
        return tableau, check_p2, tableau_list

    def find_pivot_column(self, tableau, col):
        """
        Trả về dòng pivot duy nhất có giá trị 1 trên cột này, hoặc -1 nếu không tồn tại hoặc không phải cơ bản.
        """
        xPivot = -1
        flag = False
        for i in range(1, tableau.shape[0]):
            if abs(tableau[i, col] - 1.0) < 1e-9:
                if not flag:
                    xPivot = i
                    flag = True
                else:
                    return -1
            elif abs(tableau[i, col]) > 1e-9:
                return -1
        return xPivot

    def process_output(self, tableau, result):
        """
        Xử lý kết quả sau khi chạy simplex (kết quả result):
         - result = 1 → unbounded
         - result = 0 → optimal
         - result = -1 → infeasible
        Trả về một chuỗi plain-text chứa nghiệm hoặc thông báo tương ứng.
        """
        lines = []
        if result == 1:
            # Unbounded
            if self.is_min:
                lines.append("Kết luận: Bài toán UNBOUNDED. (MIN z = -∞)")
            else:
                lines.append("Kết luận: Bài toán UNBOUNDED. (MAX z = +∞)")
        elif result == 0:
            # Optimal
            if self.is_min:
                z_val = -tableau[0, -1]
                lines.append(f"MIN z = {z_val:.6f}")
            else:
                z_val = tableau[0, -1]
                lines.append(f"MAX z = {z_val:.6f}")

            # Tìm vị trí biến cơ bản:
            pivots = np.array([self.find_pivot_column(tableau, j)
                               for j in range(tableau.shape[1] - 1)])

            # Kiểm tra nghiệm duy nhất hay đa nghiệm
            if self.check_unique_solution(tableau, pivots):
                lines.append("=> UNIQUE SOLUTION:")
                for j in range(self.num_vars - self.num_new_vars):
                    if abs(tableau[0, j]) > 1e-9:
                        # Hệ số c_j ≠ 0 trên hàm mục tiêu nghĩa là biến không cơ bản = 0
                        lines.append(f"  x{j+1} = 0")
                    else:
                        # Tìm hàng cơ bản
                        idx = pivots[j]
                        value = tableau[idx, -1]
                        if self.variable_signs[j] == -1:
                            value = -value
                        lines.append(f"  x{j+1} = {value:.6f}")
            else:
                lines.append("=> MULTIPLE SOLUTIONS:")
                # Liệt kê nghiệm tham số hóa (nếu cần)
                # Cách đơn giản: nếu biến không cơ bản (pivots[j] == -1) và c_j = 0, biến đó là free
                m = tableau.shape[1] - 1
                sign = np.array([-1 if (self.variable_signs[i] < 0 and i < self.num_vars - self.num_new_vars)
                                 else 1 for i in range(m)])
                for j in range(self.num_vars - self.num_new_vars):
                    if pivots[j] == -1:
                        if abs(tableau[0, j]) > 1e-9:
                            lines.append(f"  x{j+1} = 0")
                        else:
                            if self.variable_signs[j] == 0:
                                lines.append(f"  x{j+1} is free")
                            elif self.variable_signs[j] > 0:
                                lines.append(f"  x{j+1} ≥ 0")
                            else:
                                lines.append(f"  x{j+1} ≤ 0")
                    else:
                        base_val = sign[j] * tableau[pivots[j], -1]
                        expr = f"x{j+1} = {base_val:.6f}"
                        # Thêm phần phiếu tự do
                        for k in range(m):
                            if k == j: continue
                            if pivots[k] != -1 or abs(tableau[0, k]) > 1e-9: 
                                continue
                            # Biến tự do x_k
                            check_root, var_name = self.find_variable_name(tableau, k)
                            if check_root == 0: 
                                continue
                            coef = -sign[j] * sign[k] * tableau[pivots[j], k]
                            if abs(coef) < 1e-12:
                                continue
                            if coef > 0:
                                expr += f" + {coef:.6f}{var_name}"
                            else:
                                expr += f" - {abs(coef):.6f}{var_name}"
                        lines.append("  " + expr)
                # Ràng buộc bổ sung
                lines.append("With:")
                for i in range(m):
                    if (i >= self.num_vars - self.num_new_vars) and (i < self.num_vars):
                        continue
                    if (i < self.num_vars - self.num_new_vars) and (self.variable_signs[i] == 0):
                        continue
                    if pivots[i] == -1:
                        if i < self.num_vars - self.num_new_vars:
                            continue
                        if abs(tableau[0, i]) < 1e-9:
                            check_root, name = self.find_variable_name(tableau, i)
                            if check_root == 1:
                                if i >= self.num_vars - self.num_new_vars:
                                    lines.append(f"  {name} ≥ 0")
                                else:
                                    if self.variable_signs[i] == 0:
                                        lines.append(f"  {name} is free")
                                    elif self.variable_signs[i] < 0:
                                        lines.append(f"  {name} ≤ 0")
                                    else:
                                        lines.append(f"  {name} ≥ 0")
                    else:
                        # Biến cơ bản
                        base_val = sign[i] * tableau[pivots[i], -1]
                        expr = f"{base_val:.6f}"
                        for j in range(m):
                            if (abs(tableau[0, j]) > 1e-9) or (pivots[j] != -1):
                                continue
                            check_root, name = self.find_variable_name(tableau, j)
                            if check_root == 0:
                                continue
                            coef = -sign[i] * sign[j] * tableau[pivots[i], j]
                            if abs(coef) < 1e-12:
                                continue
                            if coef > 0:
                                expr += f" + {coef:.6f}{name}"
                            else:
                                expr += f" - {abs(coef):.6f}{name}"
                        expr += " ≥ 0"
                        lines.append("  " + expr)

        else:
            lines.append("=> INFEASIBLE: Bài toán vô nghiệm.")

        return "\n".join(lines)

    def check_unique_solution(self, tableau, pivots):
        """
        Kiểm tra nghiệm có duy nhất hay không:
        Nếu tồn tại một cột j mà c_j = 0, pivots[j] = -1 và biến đó không phải free,
        thì tồn tại nhiều nghiệm.
        """
        m = tableau.shape[1] - 1
        for j in range(m):
            if (j < self.num_vars - self.num_new_vars) and (self.variable_signs[j] == 0):
                # biến free → luôn đa nghiệm
                return False
            if pivots[j] == -1 and abs(tableau[0, j]) < 1e-9 and self.variable_signs[j] != 0:
                return False
        return True

    def find_variable_name(self, tableau, index):
        """
        Với index < num_vars - num_new_vars → biến gốc, gọi x(index+1)
        Với index >= num_vars → biến slack (w), tên w(index+1-num_vars)
        Trả về (1, tên biến) nếu tìm được, ngược lại (0, "").
        """
        name = ""
        if index < self.num_vars - self.num_new_vars:
            name = f"x{index+1}"
            return 1, name
        elif (index + 1 > self.num_vars) and (index + 1 < tableau.shape[1]):
            name = f"w{index+1-self.num_vars}"
            return 1, name
        return 0, name
