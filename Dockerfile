# Vaak web app. Works on Render (sets $PORT) and Hugging Face Spaces (port 7860, user 1000).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root user with UID 1000 (required by Hugging Face Spaces, harmless elsewhere).
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=user core core
COPY --chown=user voice_io voice_io
COPY --chown=user tools tools
COPY --chown=user web web
RUN mkdir -p out

# Public-demo defaults. Override any of these in the host's environment settings.
# API keys are NOT set here: add SARVAM_API_KEY (and optionally OPENAI_API_KEY) as secrets.
ENV PORT=7860 \
    VAAK_RATE_LIMIT_PER_MIN=5 \
    VAAK_RATE_LIMIT_PER_DAY=30 \
    VAAK_DAILY_CAP_TOTAL=100 \
    VAAK_TRUST_PROXY=1 \
    VAAK_AUDIO_TTL_MIN=30 \
    VAAK_AUDIO_MAX_FILES=200

EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn web.server:app --host 0.0.0.0 --port ${PORT:-7860} --log-level warning"]
