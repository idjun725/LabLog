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

# EasyOCR 모델(CRAFT 탐지 + 한·영 인식) 사전 다운로드.
# 없으면 컨테이너 첫 실행에서 ~150MB 다운로드 + 로드가 무료 CPU에서 5~15분 걸린다.
# 빌드 layer에 캐시되어 다음 빌드부터 startup이 거의 즉시.
RUN python -c "import easyocr; easyocr.Reader(['ko', 'en'], gpu=False)"

# SBERT 모델(jhgan/ko-sroberta-multitask, ~420MB) 사전 다운로드.
# GRU 단계 분류 입력으로 STT 문장 임베딩을 쓴다. 미리 캐시하지 않으면
# 첫 /api/classify 호출 시 다운로드 지연이 크다.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sroberta-multitask')"

COPY --chown=user . /code/

# HF Spaces 기본 포트
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
