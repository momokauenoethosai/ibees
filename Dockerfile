# Python 3.11 slim ベース
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 依存に必要なビルドツール（numpy等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存インストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体（kawakura と webapp と assets を同梱）
COPY kawakura /app/kawakura
COPY webapp /app/webapp

# Cloud Run 用（Gunicorn で起動）
ENV PORT=8080
CMD ["gunicorn", "-b", ":8080", "webapp.app:app"]
