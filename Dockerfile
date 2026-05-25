FROM python:3.10-slim

# 安装系统依赖（Tesseract OCR + OpenCV依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .
COPY watermark_remover.py .
COPY templates/ templates/

EXPOSE 5001

# 使用Gunicorn启动（生产环境）
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5001", "app:app"]
