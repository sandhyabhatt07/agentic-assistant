FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LOG_FILE=""

WORKDIR /app
COPY pyproject.toml README.md ./
COPY agentic_assistant ./agentic_assistant
RUN pip install . && useradd --create-home appuser

USER appuser
WORKDIR /home/appuser
ENTRYPOINT ["agentic-assistant"]
