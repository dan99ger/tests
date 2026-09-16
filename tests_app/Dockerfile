FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8509", "--server.address=0.0.0.0"]
