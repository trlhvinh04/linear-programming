# Linear Programming Pipeline

Hệ thống pipeline xử lý bài toán Linear Programming với CVXOPT và Open WebUI.

## 📋 Yêu cầu hệ thống

- Python 3.11
- Docker
- UV package manager

## 🚀 Hướng dẫn cài đặt

### Bước 1: Tạo môi trường ảo bằng UV

```bash
# Cài đặt UV nếu chưa có
curl -LsSf https://astral.sh/uv/install.sh | sh

# Tạo môi trường ảo với Python 3.11
uv venv --python 3.11
```

### Bước 2: Kích hoạt môi trường ảo và cài đặt dependencies

```bash
# Kích hoạt môi trường ảo
source .venv/bin/activate

# Cài đặt pip trong môi trường UV
python -m ensurepip --upgrade

# Cài đặt các thư viện từ requirements.txt
pip install -r requirements.txt

# Cài đặt thêm các thư viện cần thiết cho CVXOPT pipeline
pip install requests pillow cvxopt pydantic
```

### Bước 3: Chạy Pipeline Server

```bash
# Cấp quyền thực thi cho start.sh
chmod +x start.sh

# Chạy server
./start.sh
```

Server sẽ chạy trên `http://localhost:9099`

## 🐳 Chạy Open WebUI bằng Docker

### Quick Start with Docker

Thực hiện các bước sau để cài đặt Open WebUI với Docker.

#### Bước 1: Pull Docker Image của Open WebUI

```bash
docker pull ghcr.io/open-webui/open-webui:main
```

#### Bước 2: Chạy Container

Chạy container với cấu hình mặc định:

```bash
docker run -d -p 3000:8080 -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
```

#### Các flags quan trọng:
- **Volume Mapping** (`-v open-webui:/app/backend/data`): Đảm bảo lưu trữ dữ liệu bền vững
- **Port Mapping** (`-p 3000:8080`): Mở WebUI trên port 3000 của máy local

#### Hỗ trợ GPU (Nvidia)

Để sử dụng GPU Nvidia, thêm flag `--gpus all`:

```bash
docker run -d -p 3000:8080 --gpus all -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:cuda
```

#### Single-User Mode (Bỏ qua đăng nhập)

Để bỏ qua trang đăng nhập cho chế độ single-user:

```bash
docker run -d -p 3000:8080 -e WEBUI_AUTH=False -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
```

## 🔗 Kết nối Open WebUI với Pipeline

### Bước 1: Truy cập Open WebUI
Mở trình duyệt và truy cập: `http://localhost:3000`

### Bước 2: Cấu hình kết nối
1. Điều hướng đến **Settings > Connections > OpenAI API**
2. Thiết lập cấu hình:
   - **API URL**: `http://localhost:9099`
   - **API Key**: `0p3n-w3bu!`

### ⚠️ Lưu ý quan trọng
Nếu Open WebUI chạy trong Docker container, thay `localhost` bằng `host.docker.internal` trong API URL:
- **API URL**: `http://host.docker.internal:9099`

## 📁 Cấu trúc dự án

```
linear-programming/
├── pipelines/              # Thư mục chứa các pipeline
│   └── cvxopt_image_solver/ # Pipeline xử lý CVXOPT
├── requirements.txt         # Dependencies Python
├── start.sh                # Script khởi động server
├── main.py                 # File chính của ứng dụng
└── README.md              # File hướng dẫn này
```

## 🛠️ Troubleshooting

### Lỗi "No module named pip"
```bash
# Cài đặt pip trong môi trường UV
python -m ensurepip --upgrade
```

### Pipeline không load được
```bash
# Kiểm tra log khi chạy start.sh
# Đảm bảo đã cài đặt đầy đủ dependencies
pip install requests pillow cvxopt pydantic
```

### Không kết nối được với Open WebUI
- Kiểm tra server pipeline đang chạy trên port 9099
- Nếu dùng Docker, sử dụng `host.docker.internal` thay vì `localhost`

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng kiểm tra:
1. Môi trường ảo đã được kích hoạt
2. Tất cả dependencies đã được cài đặt
3. Server pipeline đang chạy trên port 9099
4. Open WebUI có thể truy cập được pipeline server 