import os
import json
import base64
import requests
import numpy as np
import hashlib
from pydantic import BaseModel, Field
from solver import LinearProgrammingProblem  # solver.py đã có sẵn, không cần sửa
    

# Load API keys from environment variables
def load_api_keys_from_env():
    """Load API keys from environment variables"""
    keys = []
    for i in range(1, 11):  # Check for API_KEY_1 to API_KEY_10
        key = os.getenv(f"API_KEY_{i}")
        if key:
            keys.append(key)
    
    # If no keys found in environment, use default backup
    if not keys:
        backup_key = os.getenv("VISION_API_KEY")
        if backup_key:
            keys.append(backup_key)
    
    print(f"[Pipeline] Loaded {len(keys)} API keys from environment variables")
    return keys

API_KEYS = load_api_keys_from_env()

class Pipeline:
    """
    Pipeline xử lý LP từ hình ảnh. 
    Chỉ đảm nhận việc:
      - Gọi Vision API, parse JSON
      - Chuyển constraint "≥ → ≤"
      - Ghép G và A (nếu có) mà không báo lỗi khi A/b = None hoặc rỗng
      - Ghép mảng variable_signs trực tiếp xuống solver để solver tự xử lý biến.
      - Nếu trong prompt có chứa "dantzig"/"đơn hình" → force dùng Dantzig
        Nếu có chứa "bland" → force dùng Bland
        Nếu có chứa "hai pha"/"2 pha"/"two phase"/"two-phase" → force dùng Two‐phase
      - Xoay vòng API keys khi gặp lỗi kết nối
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
        TEMPERATURE: float = Field(
            default=0.1,
            description="Temperature cho Vision Model (0.0-2.0). Giá trị thấp = ít ngẫu nhiên, giá trị cao = nhiều ngẫu nhiên."
        )
        ENABLE_CACHE: bool = Field(
            default=True,
            description="Bật/tắt cache để tiết kiệm API calls."
        )
        MAX_CACHE_SIZE: int = Field(
            default=100,
            description="Số lượng tối đa entries trong cache."
        )
        MAX_TOKENS_VISION_API: int = Field(
            default=3000,
            description="Số token tối đa cho Vision API."
        )

    def __init__(self):
        self.name = "LP Image Solver Pipeline (Custom)"
        self.current_api_key_index = 0  # Theo dõi API key hiện tại
        self.cache = {}  # Cache để lưu kết quả API calls
        valves_defaults = self.Valves().model_dump()
        self.valves = self.Valves(
            **{
                "VISION_API_URL": os.getenv("VISION_API_URL", valves_defaults["VISION_API_URL"]),
                "VISION_API_KEY": os.getenv("VISION_API_KEY", valves_defaults["VISION_API_KEY"]),
                "VISION_MODEL_ID": os.getenv("VISION_MODEL_ID", valves_defaults["VISION_MODEL_ID"]),
                "MAX_TOKENS_VISION_API": int(os.getenv("MAX_TOKENS_VISION_API", valves_defaults["MAX_TOKENS_VISION_API"])),
                "TEMPERATURE": float(os.getenv("TEMPERATURE", valves_defaults["TEMPERATURE"])),
                "ENABLE_CACHE": bool(os.getenv("ENABLE_CACHE", valves_defaults["ENABLE_CACHE"])),
                "MAX_CACHE_SIZE": int(os.getenv("MAX_CACHE_SIZE", valves_defaults["MAX_CACHE_SIZE"])),
            }
        )
        print(f"[{self.name}] Initialized. Vision API Key set: {'Yes' if self.valves.VISION_API_KEY else 'No (REQUIRED!)'}")
        print(f"[{self.name}] Cache enabled: {self.valves.ENABLE_CACHE}, Max cache size: {self.valves.MAX_CACHE_SIZE}")
        if not self.valves.VISION_API_KEY:
            print(f"[{self.name}] WARNING: VISION_API_KEY is empty! Pipeline sẽ không hoạt động đúng.")

    def _get_current_api_key(self) -> str:
        """Lấy API key hiện tại theo index"""
        if API_KEYS:
            return API_KEYS[self.current_api_key_index]
        return self.valves.VISION_API_KEY

    def _rotate_api_key(self):
        """Chuyển sang API key tiếp theo (xoay vòng)"""
        if API_KEYS:
            self.current_api_key_index = (self.current_api_key_index + 1) % len(API_KEYS)
            print(f"[{self.name}] Chuyển sang API key thứ {self.current_api_key_index + 1}")

    def _generate_cache_key(self, image_bytes: bytes, prompt_instruction: str) -> str:
        """Tạo cache key từ image bytes và prompt"""
        image_hash = hashlib.md5(image_bytes).hexdigest()
        prompt_hash = hashlib.md5(prompt_instruction.encode('utf-8')).hexdigest()
        return f"{image_hash}_{prompt_hash}"

    def _get_from_cache(self, cache_key: str) -> dict | None:
        """Lấy kết quả từ cache"""
        if not self.valves.ENABLE_CACHE:
            return None
        return self.cache.get(cache_key)

    def _save_to_cache(self, cache_key: str, data: dict):
        """Lưu kết quả vào cache với giới hạn kích thước"""
        if not self.valves.ENABLE_CACHE:
            return
        
        # Nếu cache đã đầy, xóa entry cũ nhất (FIFO)
        if len(self.cache) >= self.valves.MAX_CACHE_SIZE:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[cache_key] = data
        print(f"[{self.name}] Saved to cache. Cache size: {len(self.cache)}/{self.valves.MAX_CACHE_SIZE}")

    def clear_cache(self):
        """Xóa toàn bộ cache - có thể gọi từ bên ngoài"""
        self.cache.clear()
        print(f"[{self.name}] Cache cleared.")

    def get_cache_info(self) -> str:
        """Trả về thông tin cache"""
        if not self.valves.ENABLE_CACHE:
            return "Cache bị tắt"
        return f"Cache: {len(self.cache)}/{self.valves.MAX_CACHE_SIZE} entries"

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
        Gọi Vision API, parse JSON LP với khả năng xoay vòng API keys khi gặp lỗi:
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

        # Kiểm tra cache trước
        cache_key = self._generate_cache_key(image_bytes, prompt_instruction or "")
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            print(f"[{self.name}] Cache hit! Trả về kết quả từ cache.")
            return True, cached_result

        current_api_key = self._get_current_api_key()
        if not current_api_key:
            return False, "Lỗi cấu hình: Không có API key khả dụng."

        base64_image = self._encode_image_to_base64(image_bytes)
        if not base64_image:
            return False, "Lỗi: Không thể mã hóa hình ảnh."

        mime_type = self._get_image_mime_type(image_filename)

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
        - constraint_signs[i] = -1 nghĩa là "≤", =0 nghĩa là "=", =+1 nghĩa là "≥".
        - variable_signs[j]   =  1 nghĩa là x_j ≥ 0, =0 nghĩa là x_j tự do, =-1 nghĩa là x_j ≤ 0.
        Nếu không trích được, trả về {}.
        """
        user_prompt = """
        QUAN TRỌNG TUYỆT ĐỐI: Phản hồi của bạn BẮT BUỘC CHỈ ĐƯỢC PHÉP LÀ MỘT ĐỐI TƯỢNG JSON HỢP LỆ. KHÔNG được có bất kỳ ký tự, từ ngữ, câu văn, lời giải thích, lời chào, hay bất kỳ văn bản nào khác nằm ngoài bản thân đối tượng JSON đó. Phản hồi phải bắt đầu bằng '{' và kết thúc bằng '}'. Nếu bạn không thể trích xuất thành JSON, hãy trả về một đối tượng JSON rỗng là {}.
        `
        Bạn là một chuyên gia toán học có nhiệm vụ đọc và trích xuất thông tin từ hình ảnh của một bài toán Quy hoạch Tuyến tính (LP). Hãy tuân thủ nghiêm ngặt các bước sau để tạo đối tượng JSON cuối cùng.

        Cấu trúc JSON mục tiêu:
        {
            "problem_type": "LP",
            "objective_type": "maximize" | "minimize",
            "c": [...],
            "constraint_signs": [...],
            "G": [[...]],
            "h": [...],
            "A": [[...]],
            "b": [...],
            "variable_signs": [...]
        }

        Hướng dẫn chi tiết:

        1.  Loại bài toán (`problem_type`):
            - Giá trị này phải luôn là chuỗi "LP".

        2.  Loại hàm mục tiêu (`objective_type`):
            - Xác định bài toán là tìm "maximize" (tối đa) hay "minimize" (tối thiểu).

        3.  Vector chi phí (`c`):
            - Trích xuất các hệ số của các biến trong hàm mục tiêu và đưa vào một danh sách (list) các số thực. Thứ tự các hệ số phải tương ứng với thứ tự các biến (x_1, x_2, ...).

        4.  Xử lý Ràng buộc Chính (`G`, `h`, `constraint_signs`):
            - Xác định tất cả các ràng buộc bất đẳng thức (`<=`, `>=`) và đẳng thức (`=`).
            - Với mỗi ràng buộc, hãy trích xuất các hệ số của biến ở vế trái vào một hàng của ma trận `G`.
            - Trích xuất hằng số ở vế phải vào vector `h`.
            - Tạo vector `constraint_signs` tương ứng. Với mỗi ràng buộc `i`:
                - Nếu là dạng `... <= ...`, đặt `constraint_signs[i] = -1`.
                - Nếu là dạng `... = ...`, đặt `constraint_signs[i] = 0`.
                - Nếu là dạng `... >= ...`, đặt `constraint_signs[i] = 1`.
            - QUAN TRỌNG: Thứ tự các hàng trong `G`, các phần tử trong `h`, và các giá trị trong `constraint_signs` phải khớp với nhau một cách chính xác.

        5.  Ràng buộc Đẳng thức (`A`, `b`):
            - Trường này dùng để thể hiện riêng các ràng buộc đẳng thức.
            - Nếu có ràng buộc dạng `... = ...`, hãy đưa các hệ số vế trái vào ma trận `A` và hằng số vế phải vào vector `b`.
            - LƯU Ý: Để nhất quán, bạn có thể chọn một trong hai cách biểu diễn đẳng thức:
                - Cách 1 (Ưu tiên): Biểu diễn trong `G`, `h` với `constraint_signs` là `0`, và để `A`, `b` là danh sách rỗng (`[]`).
                - Cách 2: Biểu diễn trong `A` và `b`, và không đưa vào `G`, `h`.
            - Nếu không có ràng buộc đẳng thức, hãy để `A` và `b` là danh sách rỗng (`[]`).

        6.  Dấu của Biến (`variable_signs`):
            - Đây là phần cực kỳ quan trọng để xác định miền giá trị của từng biến. KHÔNG đưa các ràng buộc dấu của biến (ví dụ: `x >= 0`) vào ma trận `G`.
            - Tạo một vector `variable_signs` có độ dài bằng số lượng biến. Với mỗi biến `x_j`:
                - Nếu `x_j >= 0` (hoặc không có ghi chú gì, mặc định là không âm), đặt `variable_signs[j] = 1`.
                - Nếu `x_j <= 0`, đặt `variable_signs[j] = -1`.
                - Nếu `x_j` tự do (unrestricted in sign), đặt `variable_signs[j] = 0`.

        Ví dụ về cách áp dụng (Lưu ý đầu vào của ảnh có thể khác biệt):

        Cho bài toán:
        Maximize Z = 5*x1 + 3*x2
        Subject to:
        1*x1 + 1*x2 <= 10
        2*x1 + 1*x2 >= 16
        x2 = 5
        x1 >= 0, x2 <= 0

        Kết quả JSON (áp dụng cách 1 cho đẳng thức):
        {
            "problem_type": "LP",
            "objective_type": "maximize",`
            "c": [5.0, 3.0],
            "constraint_signs": [-1, 1, 0],
            "G": [[1.0, 1.0], [2.0, 1.0], [0.0, 1.0]],
            "h": [10.0, 16.0, 5.0],
            "A": [],
            "b": [],
            "variable_signs": [1, -1]
        }

        Nếu không trích xuất được thông tin, hãy trả về `{}`.
        
        """
        
        if prompt_instruction:
            user_prompt += f"\n\nLưu ý thêm: {prompt_instruction}"

        # Thử tất cả API keys nếu gặp lỗi
        initial_key_index = self.current_api_key_index
        max_api_retries = len(API_KEYS) if API_KEYS else 1
        max_json_retries = 5  # Số lần thử lại cho JSON parse error
        
        for api_attempt in range(max_api_retries):
            current_api_key = self._get_current_api_key()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {current_api_key}",
            }

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
                "temperature": self.valves.TEMPERATURE,
            }

            # Thử với API key hiện tại, có retry cho JSON parse error
            for json_attempt in range(max_json_retries):
                try:
                    print(f"[{self.name}] Thử API key thứ {self.current_api_key_index + 1} (API attempt {api_attempt + 1}/{max_api_retries}, JSON attempt {json_attempt + 1}/{max_json_retries})")
                    resp = requests.post(self.valves.VISION_API_URL, headers=headers, json=payload, timeout=120)
                    resp.raise_for_status()
                    api_resp = resp.json()
                    content = api_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content:
                        if json_attempt < max_json_retries - 1:
                            print(f"[{self.name}] Vision API không trả về nội dung, thử lại...")
                            continue
                        else:
                            break  # Chuyển sang API key khác

                    # Nếu output có ```json ... ```, bóc ra
                    if content.strip().startswith("```json"):
                        content = content.strip()[7:]
                        if content.strip().endswith("```"):
                            content = content.strip()[:-3]

                    problem_data = json.loads(content.strip())

                    required = ["problem_type", "objective_type", "c", "G", "h"]
                    missing = [k for k in required if k not in problem_data]
                    if missing:
                        if json_attempt < max_json_retries - 1:
                            print(f"[{self.name}] JSON thiếu khóa: {', '.join(missing)}, thử lại...")
                            continue
                        else:
                            break  # Chuyển sang API key khác

                    # Đảm bảo có ít nhất hai mảng dấu; nếu không, sẽ tự gán None
                    problem_data.setdefault("constraint_signs", None)
                    problem_data.setdefault("variable_signs", None)
                    # Đảm bảo A, b tồn tại (có thể None)
                    problem_data.setdefault("A", None)
                    problem_data.setdefault("b", None)

                    print(f"[{self.name}] API key thứ {self.current_api_key_index + 1} thành công!")
                    self._save_to_cache(cache_key, problem_data)
                    return True, problem_data

                except requests.exceptions.Timeout:
                    error_msg = f"[{self.name}] Lỗi: Vision API timeout với key thứ {self.current_api_key_index + 1}."
                    print(error_msg)
                    break  # Chuyển sang API key khác
                    
                except requests.exceptions.RequestException as e:
                    error_msg = f"[{self.name}] Lỗi kết nối Vision API với key thứ {self.current_api_key_index + 1}: {e}"
                    print(error_msg)
                    break  # Chuyển sang API key khác
                    
                except json.JSONDecodeError as e:
                    error_msg = f"[{self.name}] Lỗi parse JSON với key thứ {self.current_api_key_index + 1} (lần {json_attempt + 1}): {e}. Raw: {content[:200] if 'content' in locals() else 'N/A'}"
                    print(error_msg)
                    if json_attempt < max_json_retries - 1:
                        continue  # Thử lại với cùng API key
                    else:
                        # Đã thử hết 5 lần với API key này, kiểm tra xem còn API key khác không
                        if api_attempt < max_api_retries - 1:
                            break  # Chuyển sang API key khác
                        else:
                            # Đã thử hết tất cả API keys và JSON retries
                            return False, "Hệ thống vừa bị lỗi, bạn hãy nhập lại yêu cầu"
                    
                except Exception as e:
                    error_msg = f"[{self.name}] Lỗi không xác định với key thứ {self.current_api_key_index + 1}: {e}"
                    print(error_msg)
                    break  # Chuyển sang API key khác

            # Chuyển sang API key tiếp theo nếu chưa hết
            if api_attempt < max_api_retries - 1:
                self._rotate_api_key()

        return False, f"[{self.name}] Đã thử tất cả {max_api_retries} API keys nhưng đều thất bại."

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

        # --- (5) CHUYỂN QUERY "≥ → ≤" (nếu constraint_signs[i] == +1) ---
        for i, s in enumerate(constraint_signs):
            if s == 1:
                # a_i x ≥ b_i  →  (−a_i) x ≤ −b_i
                G_np[i, :]    *= -1
                h_np[i]       *= -1
                constraint_signs[i] = -1

        # CHÚ Ý: KHÔNG tự đổi biến "x ≤ 0 → x' ≥ 0" tại đây.
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
