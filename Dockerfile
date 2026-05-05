FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY inference.py /app/inference.py

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

CMD ["python", "/app/inference.py"]