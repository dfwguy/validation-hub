FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VALIDATION_DATA_ROOT=/var/lib/validation-hub
WORKDIR /app
COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-cache-dir -r requirements.lock \
    && groupadd --gid 10001 validationhub \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /app validationhub \
    && mkdir -p /var/lib/validation-hub \
    && chown 10001:10001 /var/lib/validation-hub
COPY --chown=10001:10001 backend ./backend
COPY --chown=10001:10001 frontend ./frontend
COPY --chown=10001:10001 tools ./tools
USER 10001:10001
EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/api/health',timeout=4).read()"
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1"]
