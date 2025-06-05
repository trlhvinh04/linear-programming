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

    def solve(self, method: str = None):
        """
        Giải LP bằng Simplex. 
        method: "dantzig", "bland", "two-phase" hoặc None để tự chọn theo RHS.
        """
        # 1) Chuyển về standard form
        self.convert_to_standard_form()

        # 2) Khởi tạo tableau
        tableau = np.zeros((self.num_cons + 1,
                            self.num_vars + self.num_cons + 1),
                           dtype=float)
        tableau = self.initialize_tableau(tableau)

        # 3) Mảng lưu các bảng con sau mỗi pivot
        tableau_list = []

        # 3.1) Append bảng khởi tạo vào đầu (Iteration 0)
        tableau_list.append(np.copy(tableau))

        # 4) Xác định thuật toán
        if method is None:
            algorithm_choice = self.choose_algorithm()
        else:
            m = method.lower()
            if m == "dantzig":
                algorithm_choice = 0
            elif m == "bland":
                algorithm_choice = 1
            elif m in ("two-phase", "two phase", "two_phase"):
                algorithm_choice = 2
            else:
                raise ValueError(f"Unknown method '{method}'. "
                                 f"Chỉ chấp nhận: 'dantzig', 'bland', 'two-phase'.")

        if algorithm_choice == 0:
            algo_name = "Dantzig method"
            tableau, check, tableau_list = self.dantzig_method(tableau, tableau_list)
        elif algorithm_choice == 1:
            algo_name = "Bland method"
            tableau, check, tableau_list = self.bland_method(tableau, tableau_list)
        else:
            algo_name = "Two-phase method"
            tableau, check, tableau_list = self.two_phase_method(tableau, tableau_list)

        # 5) Biên dịch kết quả ra plain-text
        output = []
        output.append(f"==== {algo_name} =====\n")

        for i, arr in enumerate(tableau_list):
            output.append(f"Iteration {i}:")
            for row in arr:
                output.append("  " + np.array2string(row, precision=6, separator=", "))
            output.append("")

        output.append("Final Tableau:")
        output.append(np.array2string(tableau, precision=6, separator=", "))
        output.append("")

        # 6) Xử lý nghiệm
        result_text = self.process_output(tableau, check)
        output.append(result_text)

        return "\n".join(output)

    def convert_to_standard_form(self):
        # 1) Nếu Maximize → Minimize
        if not self.is_min:
            self.obj_coeffs = -self.obj_coeffs

        # 2) Xử lý biến theo variable_signs
        for i in range(self.num_vars - self.num_new_vars):
            sign = self.variable_signs[i]
            if sign == -1:
                # x_i ≤ 0  <=>  y_i = -x_i ≥ 0
                self.obj_coeffs[i] = -self.obj_coeffs[i]
                self.constraint_matrix[:, i] = -self.constraint_matrix[:, i]
                # giờ vẫn giữ variable_signs[i] = -1 để khi đọc nghiệm ta đổi lại
            elif sign == 0:
                # x_i free  <=>  x_i = x_i' - x_i''   với x_i', x_i'' ≥ 0
                self.num_vars += 1
                self.num_new_vars += 1
                self.variable_signs = np.append(self.variable_signs, 0)
                self.obj_coeffs = np.append(self.obj_coeffs, -self.obj_coeffs[i])
                new_col = -self.constraint_matrix[:, i].reshape(-1, 1)
                self.constraint_matrix = np.hstack((self.constraint_matrix, new_col))

        # 3) Xử lý ràng buộc theo constraint_signs
        for i in range(self.num_cons):
            sign = self.constraint_signs[i]
            if sign == 1:    # ≥
                # Nhân cả hàng với -1 để chuyển thành ≤
                self.constraint_matrix[i, :] = -self.constraint_matrix[i, :]
                self.constraint_rhs[i] = -self.constraint_rhs[i]
                self.constraint_signs[i] = -1
            elif sign == 0:  # = 
                # Đổi hàng i thành ≤ 
                self.constraint_signs[i] = -1
                # Thêm một hàng mới “- (hàng i)” cũng là ≤
                new_matrix_row = -self.constraint_matrix[i, :].reshape(1, -1)
                new_rhs = -self.constraint_rhs[i]
                self.constraint_matrix = np.vstack((self.constraint_matrix, new_matrix_row))
                self.constraint_rhs = np.append(self.constraint_rhs, new_rhs)
                self.constraint_signs = np.append(self.constraint_signs, -1)
                self.num_cons += 1

        # Cập nhật lại num_cons cho đúng với số hàng thực trước khi khởi tạo bảng
        self.num_cons = self.constraint_matrix.shape[0]

    def initialize_tableau(self, tableau):
        """
        Điền giá trị vào tableau ban đầu:
        - Hàng 0: hệ số hàm mục tiêu (đã là Min) và cột RHS = 0
        - Hàng 1..: ma trận A (constraint_matrix), tiếp theo là ma trận đơn vị I (slack variables), cuối cùng cột RHS
        """
        # 1) Hàng 0
        tableau[0, :self.num_vars] = self.obj_coeffs
        tableau[0, -1] = 0.0

        # 2) Hàng 1..num_cons
        tableau[1:, :self.num_vars] = self.constraint_matrix
        # Thêm ma trận đơn vị I cho các biến slack
        tableau[1:, self.num_vars:self.num_vars + self.num_cons] = np.identity(self.num_cons)
        # Cột RHS
        tableau[1:, -1] = self.constraint_rhs

        # 3) Cập nhật variable_signs cho các slack variables (tất cả ≥ 0)
        self.variable_signs = np.append(self.variable_signs,
                                        np.ones(self.num_cons, dtype=int))
        return tableau

    def choose_algorithm(self):
        # Nếu có RHS < 0 → two-phase
        for rhs in self.constraint_rhs:
            if rhs < 0:
                return 2
        # Nếu có RHS == 0 → Bland
        for rhs in self.constraint_rhs:
            if abs(rhs) < 1e-9:
                return 1
        # Ngược lại → Dantzig
        return 0

    def dantzig_method(self, tableau, tableau_list, phase1=False):
        """
        Vòng lặp Dantzig:
         - Mỗi lần tìm pivot → nếu có (xPivot,yPivot,check=1) → xoay bảng, append 1 bảng sau pivot
         - Nếu check = 0: tối ưu → trả về
         - Nếu check = -1: unbounded → trả về
        """
        while True:
            xPivot, yPivot, check = self.choose_pivot_dantzig(tableau, phase1)
            if check == 1:
                tableau, xPivot, yPivot = self.rotate_pivot(tableau, xPivot, yPivot, tableau_list)
            else:
                return tableau, -check, tableau_list

    def choose_pivot_dantzig(self, tableau, phase1):
        """
        Chọn cột yPivot = chỉ số cột có hệ số âm nhỏ nhất ở hàng 0.
        Sau đó tìm xPivot theo min ratio test (RHS / hệ số tại cột yPivot).
        Nếu không có cột âm → check=0
        Nếu có cột âm nhưng không tìm ra ratio → check=-1 (unbounded)
        Ngược lại → check=1 (pivot).
        """
        minC = 0.0
        yPivot = -1
        for j in range(tableau.shape[1] - 1):
            if tableau[0, j] < minC:
                minC = tableau[0, j]
                yPivot = j
        if yPivot == -1:
            return -1, -1, 0

        xPivot = self.find_arg_min_ratio(tableau, yPivot, phase1)
        if xPivot == -1:
            return -1, yPivot, -1
        return xPivot, yPivot, 1

    def bland_method(self, tableau, tableau_list):
        """
        Vòng lặp Bland:
         - Mỗi lần tìm pivot Bland → nếu check=1 → xoay bảng, append
         - Nếu check=0 → tối ưu
         - Nếu check=-1 → unbounded
        """
        while True:
            xPivot, yPivot, check = self.choose_pivot_bland(tableau)
            if check == 1:
                tableau, xPivot, yPivot = self.rotate_pivot(tableau, xPivot, yPivot, tableau_list)
            else:
                return tableau, -check, tableau_list

    def choose_pivot_bland(self, tableau):
        """
        Chọn cột yPivot = chỉ số cột nhỏ nhất sao cho tableau[0,j]<0.
        Sau đó tìm xPivot theo min ratio test.
        """
        yPivot = -1
        for j in range(tableau.shape[1] - 1):
            if tableau[0, j] < 0:
                yPivot = j
                break
        if yPivot == -1:
            return -1, -1, 0

        xPivot = self.find_arg_min_ratio(tableau, yPivot, False)
        if xPivot == -1:
            return -1, yPivot, -1
        return xPivot, yPivot, 1

    def find_arg_min_ratio(self, tableau, yPivot, phase1):
        """
        Tìm xPivot: min ratio = min{ tableau[i,-1] / tableau[i,yPivot] | tableau[i,yPivot] > 0 }.
        Nếu nhiều hàng cùng ratio và đang trong Phase1 (phase1=True), ưu tiên hàng có tableau[i,-2] = 1.
        Nếu không có hàng hội, trả về -1.
        """
        xPivot = -1
        minRatio = np.inf
        for i in range(1, tableau.shape[0]):
            coeff = tableau[i, yPivot]
            if coeff > 0:
                ratio = tableau[i, -1] / coeff
                if ratio < minRatio - 1e-12 or \
                   (phase1 and abs(ratio - minRatio) < 1e-12 and tableau[i, -2] == 1):
                    minRatio = ratio
                    xPivot = i
        return xPivot

    def rotate_pivot(self, tableau, xPivot, yPivot, tableau_list):
        """
        Thực hiện full row-operation để pivot cột yPivot, hàng xPivot:
         1) Chia hàng xPivot cho hệ số tại (xPivot,yPivot) để thành 1
         2) Với mỗi hàng i != xPivot: tableau[i,:] -= tableau[i,yPivot] * tableau[xPivot,:]
         Sau khi hoàn tất toàn bộ row-operations, append 1 bản copy của tableau.
         -> Đảm bảo chỉ có 1 append / pivot.
        """
        # 1) Chia hàng pivot
        pivot_val = tableau[xPivot, yPivot]
        tableau[xPivot, :] = tableau[xPivot, :] / pivot_val

        # 2) Khử các hàng còn lại
        for i in range(tableau.shape[0]):
            if i != xPivot:
                factor = tableau[i, yPivot]
                tableau[i, :] -= factor * tableau[xPivot, :]

        # 3) Append 1 lần duy nhất sau khi xoay xong full bảng
        tableau_list.append(np.copy(tableau))
        return tableau, xPivot, yPivot

    def two_phase_method(self, tableau, tableau_list):
        """
        Thuật toán hai pha:
         Pha 1: Thêm biến artificial để tìm cơ sở khả thi.
         Pha 2: Loại bỏ artificial, quay lại Dantzig với tableau gốc đã build từ pha 1.
        """
        rows, cols = tableau.shape
        tableauP1 = np.zeros((rows, cols + 1))

        # 1) Khởi tạo tableauP1:
        #    - Hàng 0: hệ số artificial (cột -2) = 1, cột RHS = 0
        #    - Hàng 1..: hệ số cột artificial = -1, copy lại A và RHS từ tableau gốc
        tableauP1[0, -2] = 1.0
        tableauP1[1:, -2] = -1.0
        tableauP1[1:, :cols-1] = tableau[1:, :cols-1]
        tableauP1[1:, -1] = tableau[1:, -1]

        # 2) Xuất phát nếu có RHS âm thì phải pivot đưa artificial vào
        xPivot = -1
        for i in range(1, tableauP1.shape[0]):
            if tableauP1[i, -2] < 0:
                xPivot = i
                break
        if xPivot != -1:
            tableauP1, xPivot, _ = self.rotate_pivot(tableauP1, xPivot,
                                                     tableauP1.shape[1] - 2,
                                                     tableau_list)

        # 3) Chạy Dantzig trên tableauP1 (phase1=True)
        tableauP1, check_p1, tableau_list = self.dantzig_method(tableauP1, tableau_list, phase1=True)

        # 4) Nếu ở hàng 0 còn hệ số != 0 (ngoại trừ cột RHS & artificial) → infeasible
        for j in range(tableauP1.shape[1] - 2):
            if abs(tableauP1[0, j]) > 1e-9:
                return tableau, -1, tableau_list

        # 5) Pha 2: copy lại phần khả thi (A, RHS) về tableau gốc
        tableau[1:, :cols-1] = tableauP1[1:, :cols-1]
        tableau[1:, -1] = tableauP1[1:, -1]

        # 6) Nếu còn artificial trong cơ sở, xoay để loại bỏ nó
        for j in range(tableau.shape[1] - 1):
            x_col = self.find_pivot_column(tableau, j)
            if x_col != -1:
                tableau, x_col, j = self.rotate_pivot(tableau, x_col, j, tableau_list)

        # 7) Chạy Dantzig trên tableau gốc (đã setup xong)
        tableau, check_p2, tableau_list = self.dantzig_method(tableau, tableau_list)
        return tableau, check_p2, tableau_list

    def find_pivot_column(self, tableau, col):
        """
        Kiểm tra xem cột `col` có phải cột cơ bản không. 
        Nếu đúng: duy nhất 1 hàng i trong các hàng 1.. có tableau[i,col] = 1, 
                  còn lại đều 0 → trả về i. 
        Nếu không đúng: trả về -1.
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
        Dựa vào 'result' (1=unbounded, 0=optimal, -1=infeasible) 
        in nghiệm ra plain-text.
        """
        lines = []
        if result == 1:
            if self.is_min:
                lines.append("Kết luận: Bài toán UNBOUNDED. (MIN z = -∞)")
            else:
                lines.append("Kết luận: Bài toán UNBOUNDED. (MAX z = +∞)")
        elif result == 0:
            # Tính giá trị z
            if self.is_min:
                z_val = -tableau[0, -1]
                lines.append(f"MIN z = {z_val:.6f}")
            else:
                z_val = tableau[0, -1]
                lines.append(f"MAX z = {z_val:.6f}")

            # Xác định cột pivot
            pivots = np.array([self.find_pivot_column(tableau, j)
                               for j in range(tableau.shape[1] - 1)])

            # Kiểm tra unique / multiple
            if self.check_unique_solution(tableau, pivots):
                lines.append("=> UNIQUE SOLUTION:")
                for j in range(self.num_vars - self.num_new_vars):
                    idx = pivots[j]
                    if idx == -1:
                        lines.append(f"  x{j+1} = 0")
                    else:
                        val = tableau[idx, -1]
                        if self.variable_signs[j] == -1:
                            val = -val
                        lines.append(f"  x{j+1} = {val:.6f}")
            else:
                lines.append("=> MULTIPLE SOLUTIONS:")
                m = tableau.shape[1] - 1
                sign = np.array([-1 if (self.variable_signs[i] < 0 and i < self.num_vars - self.num_new_vars)
                                 else 1 for i in range(m)])
                for j in range(self.num_vars - self.num_new_vars):
                    idx = pivots[j]
                    if idx == -1:
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
                        base_val = sign[j] * tableau[idx, -1]
                        expr = f"x{j+1} = {base_val:.6f}"
                        for k in range(m):
                            if k == j: 
                                continue
                            if pivots[k] != -1 or abs(tableau[0, k]) > 1e-9:
                                continue
                            check_root, var_name = self.find_variable_name(tableau, k)
                            if check_root == 0:
                                continue
                            coeff = -sign[j] * sign[k] * tableau[idx, k]
                            if abs(coeff) < 1e-12:
                                continue
                            if coeff > 0:
                                expr += f" + {coeff:.6f}{var_name}"
                            else:
                                expr += f" - {abs(coeff):.6f}{var_name}"
                        lines.append("  " + expr)
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
                        base_val = sign[i] * tableau[pivots[i], -1]
                        expr = f"{base_val:.6f}"
                        for j in range(m):
                            if (abs(tableau[0, j]) > 1e-9) or (pivots[j] != -1):
                                continue
                            check_root, name = self.find_variable_name(tableau, j)
                            if check_root == 0:
                                continue
                            coeff = -sign[i] * sign[j] * tableau[pivots[i], j]
                            if abs(coeff) < 1e-12:
                                continue
                            if coeff > 0:
                                expr += f" + {coeff:.6f}{name}"
                            else:
                                expr += f" - {abs(coeff):.6f}{name}"
                        expr += " ≥ 0"
                        lines.append("  " + expr)
        else:
            lines.append("=> INFEASIBLE: Bài toán vô nghiệm.")

        return "\n".join(lines)

    def check_unique_solution(self, tableau, pivots):
        """
        Nếu tồn tại biến cơ bản không duy nhất hoặc biến free có hệ số zero,
        coi đó là multiple solutions.
        """
        m = tableau.shape[1] - 1
        for j in range(m):
            if (j < self.num_vars - self.num_new_vars) and (self.variable_signs[j] == 0):
                return False
            if pivots[j] == -1 and abs(tableau[0, j]) < 1e-9 and self.variable_signs[j] != 0:
                return False
        return True

    def find_variable_name(self, tableau, index):
        """
        Trả về (1, tên) nếu index là biến gốc (x_i) hoặc slack (w_i), ngược lại (0, "").
        """
        if index < self.num_vars - self.num_new_vars:
            return 1, f"x{index+1}"
        elif (index + 1 > self.num_vars) and (index + 1 < tableau.shape[1]):
            return 1, f"w{index+1-self.num_vars}"
        return 0, ""
