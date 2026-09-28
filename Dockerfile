FROM python:3.12-slim

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY app.py .
CMD ["python3", "app.py", "parameter1"]