"""
title: CVXOPT Image Solver Pipeline
author: Your Name
date: 2024-08-16
version: 1.2 # Incremented version
license: MIT
description: A pipeline that uses a Vision API to extract optimization problem parameters from an image and solves it using CVXOPT.
requirements: requests, pillow, cvxopt
"""

import asyncio 
import base64
import io
import json
import os
from typing import (Any, Dict, Generator,  
                    List, Optional, Union)

import requests
from cvxopt import matrix
from cvxopt.solvers import conelp, lp
from cvxopt.solvers import options as cvxopt_solver_options
from cvxopt.solvers import qp
from PIL import Image
from pydantic import BaseModel, Field


class Pipeline:
    """
    Pipeline xử lý bài toán tối ưu từ hình ảnh.
    Sử dụng Vision API để trích xuất tham số và CVXOPT để giải.
    """
    class Valves(BaseModel):
        # ... (Valves giữ nguyên) ...
        VISION_API_URL: str = Field(default="https://openrouter.ai/api/v1/chat/completions", description="URL of the external Vision API endpoint.")
        VISION_API_KEY: str = Field(default="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", description="API Key for the external Vision API.") # Để trống, user phải nhập
        VISION_MODEL_ID: str = Field(default="qwen/qwen2.5-vl-72b-instruct:free", description="Specific model ID for the Vision API.")
        MAX_TOKENS_VISION_API: int = Field(default=3000, description="Maximum tokens for the Vision API response.")
        CVXOPT_SHOW_PROGRESS: bool = Field(default=False, description="Show CVXOPT solver progress.")
        CVXOPT_MAXITERS: int = Field(default=100, description="Maximum iterations for CVXOPT solver.")
        CVXOPT_ABSTOL: float = Field(default=1e-7, description="Absolute tolerance for CVXOPT solver.")
        CVXOPT_RELTOL: float = Field(default=1e-6, description="Relative tolerance for CVXOPT solver.")
        CVXOPT_FEASTOL: float = Field(default=1e-7, description="Feasibility tolerance for CVXOPT solver.")
        CVXOPT_REFINEMENT: int = Field(default=1, description="Refinement iterations for CVXOPT solver.")


    def __init__(self):
        """
        Khởi tạo pipeline, thiết lập tên và cấu hình valves từ biến môi trường hoặc giá trị mặc định.
        Cập nhật các tùy chọn cho CVXOPT solver.
        """
        self.name = "CVXOPT Image Solver Pipeline"
        valves_defaults = self.Valves().model_dump()
        self.valves = self.Valves(
            **{
                "VISION_API_URL": os.getenv("VISION_API_URL", valves_defaults["VISION_API_URL"]),
                "VISION_API_KEY": os.getenv("VISION_API_KEY", valves_defaults["VISION_API_KEY"]), # Lấy từ env hoặc default là ""
                "VISION_MODEL_ID": os.getenv("VISION_MODEL_ID", valves_defaults["VISION_MODEL_ID"]),
                "MAX_TOKENS_VISION_API": int(os.getenv("MAX_TOKENS_VISION_API", valves_defaults["MAX_TOKENS_VISION_API"])),
                "CVXOPT_SHOW_PROGRESS": os.getenv("CVXOPT_SHOW_PROGRESS", str(valves_defaults["CVXOPT_SHOW_PROGRESS"])).lower() == "true",
                "CVXOPT_MAXITERS": int(os.getenv("CVXOPT_MAXITERS", valves_defaults["CVXOPT_MAXITERS"])),
                "CVXOPT_ABSTOL": float(os.getenv("CVXOPT_ABSTOL", valves_defaults["CVXOPT_ABSTOL"])),
                "CVXOPT_RELTOL": float(os.getenv("CVXOPT_RELTOL", valves_defaults["CVXOPT_RELTOL"])),
                "CVXOPT_FEASTOL": float(os.getenv("CVXOPT_FEASTOL", valves_defaults["CVXOPT_FEASTOL"])),
                "CVXOPT_REFINEMENT": int(os.getenv("CVXOPT_REFINEMENT", valves_defaults["CVXOPT_REFINEMENT"])),
            }
        )
        self._update_cvxopt_options()
        print(f"[{self.name}] Initialized. Vision API Key set: {'Yes' if self.valves.VISION_API_KEY else 'No (REQUIRED!)'}")
        if not self.valves.VISION_API_KEY:
            print(f"[{self.name}] WARNING: VISION_API_KEY is not set. The pipeline will not function correctly.")

    async def on_startup(self): # Giữ async nếu OpenWebUI hỗ trợ
        """
        Hàm được gọi khi pipeline khởi động.
        In thông báo pipeline đã khởi động.
        """
        print(f"[{self.name}] Started.")

    async def on_shutdown(self): # Giữ async nếu OpenWebUI hỗ trợ
        """
        Hàm được gọi khi pipeline tắt.
        In thông báo pipeline đang tắt.
        """
        print(f"[{self.name}] Shutting down.")

    def _update_cvxopt_options(self):
        """
        Cập nhật các tùy chọn của CVXOPT solver dựa trên cấu hình valves.
        """
        cvxopt_solver_options["show_progress"] = self.valves.CVXOPT_SHOW_PROGRESS
        cvxopt_solver_options["maxiters"] = self.valves.CVXOPT_MAXITERS
        cvxopt_solver_options["abstol"] = self.valves.CVXOPT_ABSTOL
        cvxopt_solver_options["reltol"] = self.valves.CVXOPT_RELTOL
        cvxopt_solver_options["feastol"] = self.valves.CVXOPT_FEASTOL
        cvxopt_solver_options["refinement"] = self.valves.CVXOPT_REFINEMENT

    def _encode_image_to_base64(self, image_bytes: bytes) -> Optional[str]:
        """
        Mã hóa bytes của hình ảnh sang chuỗi base64.

        Args:
            image_bytes: Bytes của hình ảnh.

        Returns:
            Chuỗi base64 của hình ảnh hoặc None nếu có lỗi.
        """
        try:
            return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            print(f"[{self.name}] Error encoding image bytes: {e}")
            return None

    def _get_image_mime_type(self, image_filename: str) -> str:
        """
        Xác định kiểu MIME của hình ảnh dựa trên phần mở rộng của tên file.

        Args:
            image_filename: Tên file hình ảnh.

        Returns:
            Chuỗi kiểu MIME (ví dụ: 'image/png') hoặc 'application/octet-stream' nếu không xác định được.
        """
        ext = ""
        if "." in image_filename:
            ext = image_filename.split(".")[-1].lower()
        
        if ext == "png": return "image/png"
        if ext in ["jpg", "jpeg"]: return "image/jpeg"
        if ext == "gif": return "image/gif"
        if ext == "webp": return "image/webp"
        
        print(f"[{self.name}] Warning: Unknown image extension '{ext}' for filename '{image_filename}', using application/octet-stream.")
        return "application/octet-stream"

    def _format_solver_output(
        self, result: Optional[Dict[str, Any]], problem_type: str, objective_type: str
    ) -> str:
        """
        Định dạng kết quả từ CVXOPT solver thành chuỗi văn bản dễ đọc.

        Args:
            result: Dictionary chứa kết quả từ solver.
            problem_type: Loại bài toán ('LP', 'QP', 'CONELP').
            objective_type: Loại mục tiêu ('minimize' hoặc 'maximize').

        Returns:
            Chuỗi văn bản mô tả kết quả giải bài toán.
        """
        if result is None:
            return f"Lỗi: Không có kết quả từ solver cho bài toán {problem_type}."

        status = result.get("status", "unknown")
        output_str = (
            f"Kết quả giải bài toán {problem_type} (Mục tiêu: {objective_type.upper()}):\n"
        )
        output_str += f"Trạng thái: {status}\n"

        if status == "optimal":
            primal_obj = result.get("primal objective")
            if objective_type.lower() == "maximize" and primal_obj is not None:
                primal_obj = -primal_obj
            output_str += f"Giá trị mục tiêu tối ưu: {primal_obj:.6f}\n" if primal_obj is not None else "Giá trị mục tiêu tối ưu: Không có\n"

            if "x" in result and result["x"] is not None:
                solution_vars = [f"{val:.4f}" for val in result["x"]]
                output_str += f"Nghiệm tối ưu (x): [{', '.join(solution_vars)}]\n"
            else:
                output_str += "Không tìm thấy vector nghiệm (x).\n"
            if "y" in result and result["y"] is not None:
                dual_vars_y = [f"{val:.4f}" for val in result["y"]]
                output_str += f"Biến đối ngẫu (y) cho Gx<=h: [{', '.join(dual_vars_y)}]\n"
            if "z" in result and result["z"] is not None:
                dual_vars_z = [f"{val:.4f}" for val in result["z"]]
                output_str += f"Biến đối ngẫu (z) cho Ax=b: [{', '.join(dual_vars_z)}]\n"
            if "s" in result and result["s"] is not None:
                slack_vars = [f"{val:.4f}" for val in result["s"]]
                output_str += f"Biến bù (s) cho Gx+s=h: [{', '.join(slack_vars)}]\n"
        elif status == "unbounded":
            output_str += "Bài toán không bị chặn (unbounded).\n"
        elif status == "infeasible":
            output_str += "Bài toán vô nghiệm (infeasible).\n"
        else:
            output_str += f"Solver kết thúc với trạng thái không xác định hoặc lỗi: {status}.\n"
            if result.get("primal objective") is not None:
                 output_str += f"Giá trị mục tiêu (có thể không tối ưu): {result['primal objective']:.6f}\n"
        return output_str.strip()

    def _extract_problem_from_image_api(
        self, image_bytes: bytes, image_filename: str, prompt_instruction: Optional[str]
    ) -> Union[Dict[str, Any], str]:
        """
        Gọi Vision API để trích xuất các tham số của bài toán tối ưu từ hình ảnh.

        Args:
            image_bytes: Bytes của hình ảnh.
            image_filename: Tên file hình ảnh (để xác định MIME type).
            prompt_instruction: Hướng dẫn bổ sung từ người dùng cho Vision API.

        Returns:
            Dictionary chứa các tham số của bài toán nếu thành công,
            hoặc chuỗi thông báo lỗi nếu thất bại.
        """
        if not self.valves.VISION_API_KEY:
            return "Lỗi cấu hình: VISION_API_KEY chưa được cung cấp. Vui lòng cấu hình trong Valves."

        base64_image = self._encode_image_to_base64(image_bytes)
        if not base64_image: return "Lỗi: Không thể mã hóa hình ảnh."

        mime_type = self._get_image_mime_type(image_filename)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.VISION_API_KEY}",
        }
        # USER PROMPT CỦA BẠN (GIỮ NGUYÊN HOẶC RÚT GỌN NẾU CẦN)
        user_prompt = """
        QUAN TRỌNG TUYỆT ĐỐI: Phản hồi của bạn BẮT BUỘC CHỈ ĐƯỢC PHÉP LÀ MỘT ĐỐI TƯỢNG JSON HỢP LỆ. KHÔNG được có bất kỳ ký tự, từ ngữ, câu văn, lời giải thích, lời chào, hay bất kỳ văn bản nào khác nằm ngoài bản thân đối tượng JSON đó. Phản hồi phải bắt đầu bằng '{' và kết thúc bằng '}'. Nếu bạn không thể trích xuất thành JSON, hãy trả về một đối tượng JSON rỗng là {}."
        
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
        7. (Chỉ cho CONELP) Thông số 'dims' mô tả kích thước của các nón. Ví dụ: {"l": số_ràng_buộc_tuyến_tính, "q": [kích_thước_nón_SOC_1, kích_thước_nón_SOC_2,...], "s": [kích_thước_ma_trận_SDP_1,...]}. Nếu không phải CONELP hoặc không có ràng buộc conic, để dims là null hoặc {"l": tổng_số_ràng_buộc_trong_G, "q": [], "s": []}.

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
            user_prompt += f"\n\nLưu ý thêm từ người dùng: {prompt_instruction}"

        payload = {
            "model": self.valves.VISION_MODEL_ID,
            "messages": [{"role": "user","content": [{"type": "text", "text": user_prompt},{"type": "image_url","image_url": {"url": f"data:{mime_type};base64,{base64_image}"},},],}],
            "max_tokens": self.valves.MAX_TOKENS_VISION_API,
        }
        print(f"[{self.name}] Gọi Vision API (đồng bộ): {self.valves.VISION_API_URL} với model {self.valves.VISION_MODEL_ID}")
        api_response_json = None
        content_extracted = ""
        try:
            response = requests.post(self.valves.VISION_API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            api_response_json = response.json()
            content_extracted = api_response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content_extracted: return "Lỗi: Vision API không trả về nội dung."
            if content_extracted.strip().startswith("```json"):
                content_extracted = content_extracted.strip()[7:]
                if content_extracted.strip().endswith("```"):
                    content_extracted = content_extracted.strip()[:-3]
            problem_data = json.loads(content_extracted.strip())
            required_keys = ["problem_type", "objective_type", "c", "G", "h"]
            if not all(key in problem_data for key in required_keys):
                missing_keys = [key for key in required_keys if key not in problem_data]
                raise ValueError(f"JSON từ Vision API thiếu các khóa bắt buộc: {', '.join(missing_keys)}")
            for key in ["P", "A", "b", "dims"]: problem_data.setdefault(key, None)
            return problem_data
        except requests.exceptions.Timeout: return f"[{self.name}] Lỗi: Vision API call timed out."
        except requests.exceptions.RequestException as e: return f"[{self.name}] Lỗi kết nối Vision API: {e}"
        except json.JSONDecodeError as e: return f"[{self.name}] Lỗi parse JSON: {e}. Raw: {content_extracted[:500]}..."
        except ValueError as e: return f"[{self.name}] Lỗi cấu trúc JSON: {e}. Raw: {content_extracted[:500]}..."
        except Exception as e: return f"[{self.name}] Lỗi không xác định Vision API: {e}. Response: {api_response_json}"


    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator[str, None, None]]: 
        """
        Hàm chính xử lý yêu cầu của người dùng.
        Quy trình:
        1. Kiểm tra VISION_API_KEY.
        2. Trích xuất hình ảnh từ `messages` hoặc `body['files']`.
        3. Trích xuất prompt bổ sung từ người dùng.
        4. Gọi `_extract_problem_from_image_api` để lấy dữ liệu bài toán từ Vision API.
        5. Chuẩn bị dữ liệu cho CVXOPT solver.
        6. Gọi CVXOPT solver tương ứng (LP, QP, CONELP).
        7. Định dạng và trả về kết quả.

        Args:
            user_message: Tin nhắn từ người dùng (thường là prompt cuối cùng).
            model_id: ID của model được yêu cầu (không dùng trực tiếp trong logic này).
            messages: Lịch sử tin nhắn, dùng để lấy ảnh và prompt.
            body: Request body, chứa thông tin stream và files (ảnh đính kèm).

        Returns:
            Chuỗi kết quả hoặc một generator nếu streaming được yêu cầu.
        """
        
        print(f"[{self.name}] SYNC Pipe called. User message: '{user_message}', Model: '{model_id}', Stream: {body.get('stream')}")
        
        if not self.valves.VISION_API_KEY:
            error_msg = "Lỗi cấu hình: VISION_API_KEY chưa được cung cấp."
            if body.get("stream", False):
                yield error_msg
                return # Kết thúc generator
            else:
                return error_msg

        image_info = None
        if messages:
            last_message = messages[-1] if messages else {}
            last_message_content = last_message.get("content")
            if isinstance(last_message_content, list):
                for item in last_message_content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        img_data_url = item.get("image_url", {}).get("url", "")
                        if img_data_url.startswith("data:"):
                            try:
                                header, encoded = img_data_url.split(",", 1)
                                image_bytes_content = base64.b64decode(encoded)
                                mime_type_from_url = header.split(":")[1].split(";")[0] if ":" in header and ";" in header else "image/jpeg"
                                ext_from_mime = mime_type_from_url.split("/")[-1] if "/" in mime_type_from_url else "jpg"
                                image_info = {"bytes": image_bytes_content, "filename": f"uploaded_image.{ext_from_mime}"}
                                print(f"[{self.name}] Extracted image from data URL (MIME: {mime_type_from_url}).")
                                break
                            except Exception as e:
                                error_msg = f"Lỗi giải mã data URL: {e}"
                                print(f"[{self.name}] {error_msg}")
                                if body.get("stream", False): yield error_msg; return
                                else: return error_msg
                        elif img_data_url:
                            error_msg = "Lỗi: Chỉ hỗ trợ ảnh base64 (data URL)."
                            if body.get("stream", False): yield error_msg; return
                            else: return error_msg
        
        if not image_info and body.get("files") and isinstance(body["files"], list) and body["files"]:
            file_data = body["files"][0]
            try:
                image_bytes_content = base64.b64decode(file_data["content"])
                image_info = {"bytes": image_bytes_content, "filename": file_data.get("name", "attached_image.jpg")}
                print(f"[{self.name}] Extracted image from body['files']: {image_info['filename']}")
            except Exception as e:
                error_msg = f"Lỗi xử lý file đính kèm: {e}"
                if body.get("stream", False): yield error_msg; return
                else: return error_msg

        if not image_info:
            error_msg = "Lỗi: Không tìm thấy hình ảnh."
            if body.get("stream", False): yield error_msg; return
            else: return error_msg

        user_text_prompt = ""
        if messages:
            last_message_content = messages[-1].get("content") if messages else None
            if isinstance(last_message_content, list):
                for item_prompt in last_message_content: 
                    if isinstance(item_prompt, dict) and item_prompt.get("type") == "text":
                        user_text_prompt = item_prompt.get("text", "")
                        break
            elif isinstance(last_message_content, str): user_text_prompt = last_message_content
        if not user_text_prompt and user_message: user_text_prompt = user_message

        print(f"[{self.name}] Using additional prompt: '{user_text_prompt}'")

        problem_data_or_error = self._extract_problem_from_image_api(
            image_info["bytes"], image_info["filename"], user_text_prompt
        )

        if isinstance(problem_data_or_error, str):
            if body.get("stream", False): yield problem_data_or_error; return
            else: return problem_data_or_error
        
        problem_data = problem_data_or_error
        print(f"[{self.name}] Problem data: {json.dumps(problem_data, indent=2)}")

        problem_type = problem_data.get("problem_type", "LP").upper()
        objective_type = problem_data.get("objective_type", "minimize").lower()
        c_list = problem_data.get("c"); G_list = problem_data.get("G"); h_list = problem_data.get("h")
        A_list = problem_data.get("A"); b_list = problem_data.get("b"); P_list = problem_data.get("P")
        dims = problem_data.get("dims")

        if not (isinstance(c_list, list) and isinstance(G_list, list) and isinstance(h_list, list)):
            error_msg = "Lỗi: c, G, h từ Vision API không phải là danh sách."
            if body.get("stream", False): yield error_msg; return
            else: return error_msg

        try:
            c_internal = [float(x) for x in c_list]
            if objective_type == "maximize": c_internal = [-x for x in c_internal]
            c_matrix = matrix(c_internal)
            num_vars = len(c_list)
            if not G_list and not h_list:
                G_matrix = matrix(0.0, (0, num_vars)); h_matrix = matrix(0.0, (0,1))
            elif G_list and h_list:
                if len(G_list) != len(h_list): raise ValueError(f"G rows ({len(G_list)}) != h len ({len(h_list)})")
                if G_list and any(len(row) != num_vars for row in G_list): raise ValueError("G columns != num_vars")
                G_matrix = matrix([[float(x) for x in row] for row in G_list]).T
                h_matrix = matrix([float(x) for x in h_list])
            else: raise ValueError("G và h phải cùng được cung cấp hoặc cùng rỗng.")
            A_matrix, b_matrix = None, None
            if A_list and b_list:
                if len(A_list) != len(b_list): raise ValueError(f"A rows ({len(A_list)}) != b len ({len(b_list)})")
                if A_list and any(len(row) != num_vars for row in A_list): raise ValueError("A columns != num_vars")
                A_matrix = matrix([[float(x) for x in row] for row in A_list]).T
                b_matrix = matrix([float(x) for x in b_list])
            elif A_list or b_list: raise ValueError("A và b phải cùng được cung cấp hoặc cùng rỗng.")
        except (ValueError, TypeError) as e:
            error_msg = f"Lỗi chuẩn bị dữ liệu CVXOPT: {e}"
            if body.get("stream", False): yield error_msg; return
            else: return error_msg

        solution_dict = None
        self._update_cvxopt_options()
        try:
            print(f"[{self.name}] Solving {problem_type} ({objective_type})...")
            if problem_type == "LP":
                solution_dict = lp(c_matrix, G_matrix, h_matrix, A_matrix, b_matrix)
            elif problem_type == "QP":
                if not P_list or not isinstance(P_list, list):
                    error_msg = "Lỗi: P là bắt buộc cho QP."
                    if body.get("stream", False): yield error_msg; return
                    else: return error_msg
                if len(P_list) != num_vars or any(len(row) != num_vars for row in P_list):
                    error_msg = f"Lỗi: Kích thước ma trận P không đúng ({len(P_list)}x... so với {num_vars} biến)."
                    if body.get("stream", False): yield error_msg; return
                    else: return error_msg
                P_matrix = matrix([[float(x) for x in row] for row in P_list])
                solution_dict = qp(P_matrix, c_matrix, G_matrix, h_matrix, A_matrix, b_matrix)
            elif problem_type == "CONELP":
                if not dims or not isinstance(dims, dict):
                    error_msg = "Lỗi: dims là bắt buộc cho CONELP."
                    if body.get("stream", False): yield error_msg; return
                    else: return error_msg
                dims.setdefault("l", G_matrix.size[0] if G_matrix else 0)
                dims.setdefault("q", []); dims.setdefault("s", [])
                solution_dict = conelp(c_matrix, G_matrix, h_matrix, dims, A_matrix, b_matrix)
            else:
                error_msg = f"Lỗi: Loại bài toán '{problem_type}' không hỗ trợ."
                if body.get("stream", False): yield error_msg; return
                else: return error_msg
        except (ValueError, ArithmeticError, Exception) as e:
            error_msg = f"Lỗi CVXOPT: {type(e).__name__}: {e}"
            if body.get("stream", False): yield error_msg; return
            else: return error_msg

        formatted_output = self._format_solver_output(solution_dict, problem_type, objective_type)

        if body.get("stream", False):
            for line in formatted_output.splitlines(True):
                yield line
        else:
            return formatted_output
