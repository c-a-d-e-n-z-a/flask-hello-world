# 使用輕量級的 Python 映像檔
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製當前目錄內容到容器中的 /app
COPY . /app

# 安裝 requirements.txt 中的套件
RUN pip install --no-cache-dir -r requirements.txt

# 設定環境變數 (讓 Flask 顯示 log)
ENV PYTHONUNBUFFERED True

# Cloud Run 預設會傳入 PORT 環境變數 (通常是 8080)
# 使用 gunicorn 啟動 Flask，並綁定該 PORT
# 假設你的主程式檔名是 app.py，Flask 實例變數名稱是 app
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
