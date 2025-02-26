# Dockerfile cho ứng dụng Flask
FROM python:3.9-slim
# # Thiết lập thư mục làm việc
# WORKDIR /app

# Sao chép file yêu cầu và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install faiss-cpu bs4 nltk spacy
RUN pip install --upgrade flask werkzeug flask-sqlalchemy flask-migrate

RUN python -m spacy download en_core_web_sm
# Sao chép toàn bộ mã nguồn vào container
COPY ./app .

# Chạy ứng dụng Flask
# CMD ["ls"]
CMD ["python", "app.py"]
