import os
import json
import base64
import requests
import numpy as np
from pydantic import BaseModel, Field
from solver import LinearProgrammingProblem  # solver.py đã có sẵn, không cần sửa
    

class Pipeline:
    """
    Pipeline xử lý LP từ hình ảnh. 
    Chỉ đảm nhận việc:
      - Gọi Vision API, parse JSON
      - Chuyển constraint “≥ → ≤”
      - Ghép G và A (nếu có) mà không báo lỗi khi A/b = None hoặc rỗng
      - Ghép mảng variable_signs trực tiếp xuống solver để solver tự xử lý biến.
      - Nếu trong prompt có chứa “dantzig”/“đơn hình” → force dùng Dantzig
        Nếu có chứa “bland” → force dùng Bland
        Nếu có chứa “hai pha”/“2 pha”/“two phase”/“two-phase” → force dùng Two‐phase
    """

    class Valves(BaseModel):
        VISION_API_URL: str = Field(
            default="https://openrouter.ai/api/v1/chat/completions",
            description="URL của Vision API endpoint."
        )
        VISION_API_KEY: str = Field(
            default="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",
            description="API Key cho Vision API (bắt buộc)."
        )
        VISION_MODEL_ID: str = Field(
            default="qwen/qwen2.5-vl-72b-instruct:free",
            description="Model ID cho Vision API."
        )
        MAX_TOKENS_VISION_API: int = Field(
            default=3000,
            description="Số token tối đa cho Vision API."
        )

    def __init__(self):
        self.name = "LP Image Solver Pipeline (Custom)"
        valves_defaults = self.Valves().model_dump()
        self.valves = self.Valves(
            **{
                "VISION_API_URL": os.getenv("VISION_API_URL", valves_defaults["VISION_API_URL"]),
                "VISION_API_KEY": os.getenv("VISION_API_KEY", valves_defaults["VISION_API_KEY"]),
                "VISION_MODEL_ID": os.getenv("VISION_MODEL_ID", valves_defaults["VISION_MODEL_ID"]),
                "MAX_TOKENS_VISION_API": int(os.getenv("MAX_TOKENS_VISION_API", valves_defaults["MAX_TOKENS_VISION_API"])),
            }
        )
        print(f"[{self.name}] Initialized. Vision API Key set: {'Yes' if self.valves.VISION_API_KEY else 'No (REQUIRED!)'}")
        if not self.valves.VISION_API_KEY:
            print(f"[{self.name}] WARNING: VISION_API_KEY is empty! Pipeline sẽ không hoạt động đúng.")

    async def on_startup(self):
        print(f"[{self.name}] Started.")

    async def on_shutdown(self):
        print(f"[{self.name}] Shutting down.")

    def _encode_image_to_base64(self, image_bytes: bytes) -> str:
        try:
            return base64.b64encode(image_bytes).decode("utf-8")
        except:
            return ""

    def _get_image_mime_type(self, image_filename: str) -> str:
        ext = image_filename.split(".")[-1].lower() if "." in image_filename else ""
        if ext == "png":
            return "image/png"
        if ext in ["jpg", "jpeg"]:
            return "image/jpeg"
        if ext == "gif":
            return "image/gif"
        if ext == "webp":
            return "image/webp"
        return "application/octet-stream"

    def _extract_problem_from_image_api(
        self, image_bytes: bytes, image_filename: str, prompt_instruction: str
    ) -> tuple[bool, dict | str]:
        """
        Gọi Vision API, parse JSON LP:
        {
          "problem_type": "LP",
          "objective_type": "maximize"/"minimize",
          "c": [c1, c2, ..., cn],
          "G": [[g11,g12,...], [g21,g22,...], ...],
          "h": [h1, h2, ...],
          # Nếu có constraint đẳng thức:
          "A": [[a11,a12,...], ...],    # list of lists hoặc None
          "b": [b1, b2, ...],           # list hoặc None
          "constraint_signs": [-1,0,+1,...],  # -1=≤,0==, +1=≥
          "variable_signs": [1,0,-1,...]      # 1=x≥0, 0=x free, -1=x≤0
        }
        Nếu thiếu khóa quan trọng hoặc lỗi parse, trả về False + thông báo lỗi.
        """

        if not self.valves.VISION_API_KEY:
            return False, "Lỗi cấu hình: VISION_API_KEY chưa được cấp."

        base64_image = self._encode_image_to_base64(image_bytes)
        if not base64_image:
            return False, "Lỗi: Không thể mã hóa hình ảnh."

        mime_type = self._get_image_mime_type(image_filename)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.VISION_API_KEY}",
        }

        user_prompt = """
        QUAN TRỌNG: Chỉ trả về một đối tượng JSON. Ví dụ (LP):
        {
          "problem_type": "LP",
          "objective_type": "maximize",
          "c": [5.0,3.0],
          "constraint_signs": [-1, -1, -1],
          "G": [[1.0,1.0],[2.0,1.0],[1.0,4.0]],
          "h": [10.0,16.0,32.0],
          "A": [[0.0,1.0]],      # nếu có (đẳng thức)
          "b": [5.0],           # nếu có
          "variable_signs": [1, -1]
        }
        - constraint_signs[i] = -1 nghĩa là “≤”, =0 nghĩa là “=”, =+1 nghĩa là “≥”.
        - variable_signs[j]   =  1 nghĩa là x_j ≥ 0, =0 nghĩa là x_j tự do, =-1 nghĩa là x_j ≤ 0.
        Nếu không trích được, trả về {}.
        """
        if prompt_instruction:
            user_prompt += f"\n\nLưu ý thêm: {prompt_instruction}"

        payload = {
            "model": self.valves.VISION_MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                    ],
                }
            ],
            "max_tokens": self.valves.MAX_TOKENS_VISION_API,
        }

        try:
            resp = requests.post(self.valves.VISION_API_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            api_resp = resp.json()
            content = api_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                return False, "Lỗi: Vision API không trả về nội dung."

            # Nếu output có ```json ... ```, bóc ra
            if content.strip().startswith("```json"):
                content = content.strip()[7:]
                if content.strip().endswith("```"):
                    content = content.strip()[:-3]

            problem_data = json.loads(content.strip())

            required = ["problem_type", "objective_type", "c", "G", "h"]
            missing = [k for k in required if k not in problem_data]
            if missing:
                return False, f"JSON thiếu khóa: {', '.join(missing)}"

            # Đảm bảo có ít nhất hai mảng dấu; nếu không, sẽ tự gán None
            problem_data.setdefault("constraint_signs", None)
            problem_data.setdefault("variable_signs", None)
            # Đảm bảo A, b tồn tại (có thể None)
            problem_data.setdefault("A", None)
            problem_data.setdefault("b", None)

            return True, problem_data

        except requests.exceptions.Timeout:
            return False, f"[{self.name}] Lỗi: Vision API timeout."
        except requests.exceptions.RequestException as e:
            return False, f"[{self.name}] Lỗi kết nối Vision API: {e}"
        except json.JSONDecodeError as e:
            return False, f"[{self.name}] Lỗi parse JSON: {e}. Raw: {content[:200]}"
        except Exception as e:
            return False, f"[{self.name}] Lỗi không xác định: {e}"

    def pipe(
        self, user_message: str, model_id: str, messages: list, body: dict
    ) -> str:
        """
        Xử lý request, chỉ support LP.

        Trả về HTML/Text chứa kết quả (solver trả về).
        """

        print(f"[{self.name}] PIPE called. Stream? {body.get('stream', False)}")

        if not self.valves.VISION_API_KEY:
            return "Lỗi cấu hình: VISION_API_KEY chưa được cấp."

        # --- (1) LẤY ẢNH TỪ REQUEST ---
        image_info = None
        if messages:
            last = messages[-1]
            cont = last.get("content")
            if isinstance(cont, list):
                for item in cont:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            try:
                                header, encoded = url.split(",", 1)
                                img_bytes = base64.b64decode(encoded)
                                mime = header.split(":")[1].split(";")[0] if ":" in header and ";" in header else "image/jpeg"
                                ext = mime.split("/")[-1]
                                image_info = {"bytes": img_bytes, "filename": f"uploaded_image.{ext}"}
                                break
                            except Exception as e:
                                return f"Lỗi giải mã data URL: {e}"

        if not image_info and body.get("files"):
            file0 = body["files"][0]
            try:
                img_bytes = base64.b64decode(file0["content"])
                image_info = {"bytes": img_bytes, "filename": file0.get("name", "attached_image.jpg")}
            except Exception as e:
                return f"Lỗi xử lý file đính kèm: {e}"

        if not image_info:
            return "Lỗi: Không tìm thấy hình ảnh."

        # --- (2) LẤY PROMPT TEXT (nếu user có gắn thêm) ---
        user_text_prompt = ""
        if messages:
            last = messages[-1]
            cont = last.get("content")
            if isinstance(cont, list):
                for item in cont:
                    if isinstance(item, dict) and item.get("type") == "text":
                        user_text_prompt = item.get("text", "")
                        break
            elif isinstance(cont, str):
                user_text_prompt = cont
        if not user_text_prompt and user_message:
            user_text_prompt = user_message

        print(f"[{self.name}] Dùng prompt: '{user_text_prompt}'")

        # --- (3) GỌI VISION API ---
        ok, problem_or_err = self._extract_problem_from_image_api(
            image_info["bytes"], image_info["filename"], user_text_prompt
        )
        if not ok:
            return problem_or_err

        problem_data = problem_or_err
        print(f"[{self.name}] Problem data:\n{json.dumps(problem_data, indent=2, ensure_ascii=False)}")

        # --- (4) PARSE CÁC MẢNG trong JSON ---
        problem_type    = problem_data.get("problem_type", "LP").upper()
        objective_type  = problem_data.get("objective_type", "minimize").lower()
        c_list          = problem_data.get("c")
        G_list          = problem_data.get("G")
        h_list          = problem_data.get("h")
        A_list          = problem_data.get("A")                     # Có thể None hoặc list
        b_list          = problem_data.get("b")                     # Có thể None hoặc list
        signs_list      = problem_data.get("constraint_signs")      # [-1,0,+1] hoặc None
        var_signs_list  = problem_data.get("variable_signs")        # [1,0,-1] hoặc None

        # Kiểm tra problem_type
        if problem_type != "LP":
            return f"Lỗi: Chỉ hỗ trợ LP. Loại bài toán '{problem_type}' chưa hỗ trợ."

        # Kiểm tra c, G, h
        if not (isinstance(c_list, list) and isinstance(G_list, list) and isinstance(h_list, list)):
            return "Lỗi: c, G, h phải là danh sách."

        try:
            # Chuyển c, G, h sang numpy
            c_internal = [float(x) for x in c_list]
            num_vars   = len(c_internal)

            G_np = np.array(G_list, dtype=float)
            h_np = np.array(h_list, dtype=float)

            if G_np.ndim != 2 or G_np.shape[1] != num_vars:
                return "Lỗi: Kích thước ma trận G không khớp với chiều của c."

            num_cons = G_np.shape[0]

            # --- XỬ LÝ A, b (chỉ khi A_list và b_list là danh sách không rỗng) ---
            if isinstance(A_list, list) and isinstance(b_list, list) and len(A_list) > 0 and len(b_list) > 0:
                A_np = np.array(A_list, dtype=float)
                b_np = np.array(b_list, dtype=float)
                if A_np.ndim != 2 or A_np.shape[1] != num_vars or \
                   b_np.ndim != 1 or A_np.shape[0] != b_np.shape[0]:
                    return "Lỗi: Kích thước A/b không khớp với số biến."
                # Ghép G và A (hàng A là đẳng thức, sẽ mark sign = 0)
                G_np = np.vstack([G_np, A_np])
                h_np = np.concatenate([h_np, b_np])
                num_cons = G_np.shape[0]
                if signs_list is None:
                    # Hàng đẳng thức sẽ để sign = 0
                    signs_list = [-1] * (num_cons - A_np.shape[0]) + [0] * A_np.shape[0]
            else:
                # A_list = None, hoặc A_list = [] hoặc b_list = [] → không có đẳng thức
                if signs_list is None:
                    signs_list = [-1] * num_cons

            # Chuyển sang numpy
            constraint_signs = np.array(signs_list, dtype=int)

            # Nếu API không trả variable_signs thì mặc định x ≥ 0
            if var_signs_list is None:
                var_signs_list = [1] * num_vars
            variable_signs = np.array(var_signs_list, dtype=int)

        except (ValueError, TypeError) as e:
            return f"Lỗi chuẩn bị dữ liệu LP: {e}"

        # --- (5) CHUYỂN QUERY “≥ → ≤” (nếu constraint_signs[i] == +1) ---
        for i, s in enumerate(constraint_signs):
            if s == 1:
                # a_i x ≥ b_i  →  (−a_i) x ≤ −b_i
                G_np[i, :]    *= -1
                h_np[i]       *= -1
                constraint_signs[i] = -1

        # CHÚ Ý: KHÔNG tự đổi biến “x ≤ 0 → x' ≥ 0” tại đây.
        # Solver.py đã có sẵn logic để xử lý variable_signs.

        # --- (6) Khởi tạo solver và (nếu cần) áp override thuật toán ---
        is_min_flag = (objective_type == "minimize")
        try:
            lp_solver = LinearProgrammingProblem(
                num_vars=num_vars,
                num_cons=len(constraint_signs),
                is_min=is_min_flag,
                obj_coeffs=np.array(c_internal, dtype=float),
                constraint_matrix=G_np,
                constraint_rhs=h_np,
                constraint_signs=constraint_signs,
                variable_signs=variable_signs
            )
        except Exception as e:
            return f"Lỗi khi khởi tạo LP solver: {e}"

        # === BẮT ĐỀU override THUẬT TOÁN TỪ 'user_text_prompt' ===
        prompt_lower = user_text_prompt.lower()

        # Nếu có từ khoá "dantzig" hoặc "đơn hình", force Dantzig (return 0)
        if "dantzig" in prompt_lower or "đơn hình" in prompt_lower:
            lp_solver.choose_algorithm = lambda: 0

        # Nếu có từ khoá "bland", force Bland (return 1)
        elif "bland" in prompt_lower:
            lp_solver.choose_algorithm = lambda: 1

        # Nếu có từ khoá "hai pha" / "2 pha" / "two phase" / "two-phase", force Two‐phase (return 2)
        elif ("hai pha" in prompt_lower 
              or "2 pha" in prompt_lower 
              or "two phase" in prompt_lower 
              or "two-phase" in prompt_lower):
            lp_solver.choose_algorithm = lambda: 2

        # Nếu không detect được từ khoá, solver sẽ tự động chọn thuật toán qua choose_algorithm() gốc.

        # --- (7) Gọi solve() và trả kết quả ---
        try:
            output_html = lp_solver.solve()
        except Exception as e:
            return f"Lỗi khi chạy custom LP solver: {e}"

        return output_html
