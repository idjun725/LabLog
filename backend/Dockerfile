FROM python:3.12-slim

# 시스템 의존성: ffmpeg(STT/VAD), libgl(opencv), libglib(easyocr)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces 권장: 비-root 사용자
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /code

COPY --chown=user requirements.txt /code/
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . /code/

# HF Spaces 기본 포트
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
