FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CREDIT_RISK_ROOT=/app \
    PYTHONPATH=/app/src

# libgomp is required by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src ./src
COPY configs ./configs
COPY artifacts ./artifacts

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import json, os, urllib.request; r = json.load(urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)); raise SystemExit(0 if r['status'] == 'ok' else 1)"

CMD ["sh", "-c", "exec uvicorn credit_risk.api.main:app --host 0.0.0.0 --port ${PORT}"]
