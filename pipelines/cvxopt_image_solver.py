"""
title: CVXOPT Image Solver Pipeline
author: Your Name
date: 2024-08-16
version: 1.2 # Incremented version
license: MIT
description: A pipeline that uses a Vision API to extract optimization problem parameters from an image and solves it using a custom LP solver.
requirements: requests, pillow, numpy
"""

import os
import json
import base64
import requests
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from solver import LinearProgrammingProblem  # Chỉ import custom LP solver


class Pipeline:
    """
    Pipeline xử lý bài toán tối ưu từ hình ảnh, chỉ hỗ trợ LP thông qua LinearProgrammingProblem.
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
        except Exception:
            return ""

    def _get_image_mime_type(self, image_filename: str) -> str:
        ext = image_filename.split(".")[-1].lower() if "." in image_filename else ""
        if ext == "png": return "image/png"
        if ext in ["jpg", "jpeg"]: return "image/jpeg"
        if ext == "gif": return "image/gif"
        if ext == "webp": return "image/webp"
        return "application/octet-stream"

    def _extract_problem_from_image_api(
        self, image_bytes: bytes, image_filename: str, prompt_instruction: str
    ) -> tuple[bool, dict | str]:
        """
        Gọi Vision API để trích xuất bài toán.
        Trả về (True, problem_data_dict) nếu thành công,
        hoặc (False, error_message) nếu thất bại.
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

        # Prompt cố định để Vision API trả về JSON hợp lệ
        user_prompt = """
        QUAN TRỌNG TUYỆT ĐỐI: Phản hồi của bạn BẮT BUỘC CHỈ ĐƯỢC PHÉP LÀ MỘT ĐỐI TƯỢNG JSON HỢP LỆ. KHÔNG được có bất kỳ ký tự, từ ngữ, câu văn, lời giải thích, lời chào, hay bất kỳ văn bản nào khác nằm ngoài bản thân đối tượng JSON đó. Phản hồi phải bắt đầu bằng '{' và kết thúc bằng '}'. Nếu bạn không thể trích xuất thành JSON, hãy trả về một đối tượng JSON rỗng là {}.
        
        Bạn là một chuyên gia toán học có khả năng đọc hiểu và trích xuất thông tin từ hình ảnh các bài toán tối ưu.
        Từ hình ảnh được cung cấp, hãy trích xuất các thành phần của bài toán Quy hoạch Tuyến tính (LP), Quy hoạch Toàn phương (QP), hoặc Quy hoạch Conic Bậc hai (CONELP).
        1. Xác định loại bài toán: 'LP', 'QP', 'CONELP'. Lưu vào khóa 'problem_type'.
        2. Xác định xem đây là bài toán MINIMIZE hay MAXIMIZE. Lưu vào khóa 'objective_type'.
        3. Vector chi phí 'c' (hoặc 'q' cho QP) của hàm mục tiêu.
        4. Ma trận 'P' cho thành phần toàn phương 0.5*x'*P*x (chỉ cho QP, nếu không có thì null).
        5. Các ma trận ràng buộc bất đẳng thức 'G' và vector vế phải 'h'. Hãy chuẩn hóa TẤT CẢ các bất đẳng thức về dạng Gx <= h.
           - Nếu gặp 'expr >= val', chuyển thành '-expr <= -val'.
           - Ràng buộc biến như 'x >= 0' phải được chuyển thành '-x <= 0'.
        6. Các ma trận ràng buộc đẳng thức 'A' và vector vế phải 'b' (nếu có, nếu không thì null hoặc danh sách rỗng).
        7. (Chỉ cho CONELP) Thông số 'dims' mô tả kích thước của các nón. Ví dụ: {"l": số_ràng_buộc_tuyến_tính, "q": [kích_thước_nón_SOC_1, ...], "s": [kích_thước_ma_trận_SDP_1,...]}. Nếu không phải CONELP hoặc không có ràng buộc conic, để dims là null hoặc {"l": tổng_số_ràng_buộc_trong_G, "q": [], "s": []}.

        Hãy trả về kết quả dưới dạng một đối tượng JSON.
        Ví dụ LP:
        {
          "problem_type": "LP",
          "objective_type": "maximize",
          "c": [5.0, 3.0],
          "P": null,
          "G": [[1.0, 1.0], [2.0, 1.0], [1.0, 4.0], [-1.0, 0.0], [0.0, -1.0]],
          "h": [10.0, 16.0, 32.0, 0.0, 0.0],
          "A": null,
          "b": null,
          "dims": null
        }
        LƯU Ý: Luôn bao gồm tất cả các khóa: 'problem_type', 'objective_type', 'c', 'P', 'G', 'h', 'A', 'b', 'dims'. Nếu một thành phần không có (ví dụ A, b, P, dims), hãy đặt giá trị là null hoặc danh sách rỗng tương ứng. Vector 'c' luôn phải có. 'G' và 'h' cũng phải có, ngay cả khi chỉ chứa ràng buộc x_i >= 0.
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
            response = requests.post(self.valves.VISION_API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            api_resp = response.json()
            content = api_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                return False, "Lỗi: Vision API không trả về nội dung."
            # Nếu có ```json ... ``` thì bóc ra
            if content.strip().startswith("```json"):
                content = content.strip()[7:]
                if content.strip().endswith("```"):
                    content = content.strip()[:-3]
            problem_data = json.loads(content.strip())
            # Kiểm tra các khóa bắt buộc
            required = ["problem_type", "objective_type", "c", "G", "h"]
            missing = [k for k in required if k not in problem_data]
            if missing:
                return False, f"JSON thiếu khóa: {', '.join(missing)}"
            # Đảm bảo tồn tại các khóa P, A, b, dims (nếu không có, gán null)
            for k in ["P", "A", "b", "dims"]:
                problem_data.setdefault(k, None)
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
        Xử lý request, chỉ hỗ trợ LP. 
        Cho phép user nhập thêm 'method=dantzig', 'method=bland' hoặc 'method=two-phase' trong prompt.
        """
        print(f"[{self.name}] PIPE called. Stream? {body.get('stream', False)}")

        # Bước 0: Kiểm tra Vision API Key
        if not self.valves.VISION_API_KEY:
            return "Lỗi cấu hình: VISION_API_KEY chưa được cấp."

        # --- Bước 1: Lấy ảnh từ messages hoặc body['files'] ---
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

        # --- Bước 2: Lấy prompt text bổ sung (user_text_prompt) ---
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

        # --- Bước 3: Gọi Vision API để trích problem_data ---
        ok, problem_or_err = self._extract_problem_from_image_api(
            image_info["bytes"], image_info["filename"], user_text_prompt
        )
        if not ok:
            return problem_or_err  # Trả về lỗi ngay

        problem_data = problem_or_err
        print(f"[{self.name}] Problem data: {json.dumps(problem_data, indent=2)}")

        # --- Bước 4: Đọc và chuẩn hóa thông tin LP ---
        problem_type   = problem_data.get("problem_type", "LP").upper()
        objective_type = problem_data.get("objective_type", "minimize").lower()
        c_list         = problem_data.get("c")
        G_list         = problem_data.get("G")
        h_list         = problem_data.get("h")
        A_list         = problem_data.get("A")
        b_list         = problem_data.get("b")

        # Chỉ hỗ trợ LP
        if problem_type != "LP":
            return f"Lỗi: Chỉ hỗ trợ Linear Programming. Loại bài toán '{problem_type}' chưa hỗ trợ."

        # Kiểm tra c, G, h là list
        if not (isinstance(c_list, list) and isinstance(G_list, list) and isinstance(h_list, list)):
            return "Lỗi: c, G, h phải là danh sách."

        try:
            # Chuyển c sang float
            c_internal = [float(x) for x in c_list]
            num_vars = len(c_internal)

            # G_matrix và h_vector
            G_np = np.array(G_list, dtype=float)
            h_np = np.array(h_list, dtype=float)
            num_cons = len(h_np)

            # Nếu có A, b (constraint đẳng thức), ghép vào
            if A_list and b_list:
                A_np = np.array(A_list, dtype=float)
                b_np = np.array(b_list, dtype=float)
                if A_np.shape[1] != num_vars or b_np.ndim != 1 or A_np.shape[0] != b_np.shape[0]:
                    raise ValueError("Kích thước A/b không khớp với số biến.")
                combined_matrix = np.vstack([G_np, A_np])
                combined_rhs    = np.concatenate([h_np, b_np])
                signs_G = [-1] * num_cons
                signs_A = [0] * A_np.shape[0]
                constraint_signs = np.array(signs_G + signs_A, dtype=int)
                G_np = combined_matrix
                h_np = combined_rhs
                num_cons = G_np.shape[0]
            else:
                # Chỉ Gx ≤ h
                constraint_signs = np.array([-1] * num_cons, dtype=int)

            # Mặc định biến x ≥ 0
            variable_signs = np.array([1] * num_vars, dtype=int)

        except (ValueError, TypeError) as e:
            return f"Lỗi chuẩn bị dữ liệu LP: {e}"

        # --- Bước 5: Tìm từ khóa method trong user_text_prompt ---
        low = user_text_prompt.lower()
        user_method = None
        if "method=đơn hình" in low or "method=dantzig" in low:
            user_method = "dantzig"
        elif "method=bland" in low:
            user_method = "bland"
        elif ("method=two-phase" in low or "method=two phase" in low or "method=two_phase" in low or "method=2 pha" in low or "method=hai pha" in low):
            user_method = "two-phase"
        # Nếu không tìm thấy, user_method vẫn là None → solver tự quyết

        # --- Bước 6: Khởi tạo và giải bằng LinearProgrammingProblem ---
        is_min_flag = (objective_type == "minimize")
        lp_solver = LinearProgrammingProblem(
            num_vars=num_vars,
            num_cons=num_cons,
            is_min=is_min_flag,
            obj_coeffs=np.array(c_internal, dtype=float),
            constraint_matrix=G_np,
            constraint_rhs=h_np,
            constraint_signs=constraint_signs,
            variable_signs=variable_signs
        )

        try:
            # Truyền vào method nếu user đã chỉ định
            result_text = lp_solver.solve(method=user_method)
        except Exception as e:
            return f"Lỗi khi chạy LP solver: {e}"

        # --- Bước 7: Trả kết quả (plain-text) ---
        return result_text
