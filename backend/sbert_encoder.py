"""SBERT 문장 임베딩 인코더 — STT 발화 문장을 의미 벡터로 변환.

`jhgan/ko-sroberta-multitask` (768-dim, 한국어 특화)를 싱글턴으로 로드.
빈 문자열은 인코딩을 건너뛰고 zero vector를 반환해 (a) has_speech 신호와의
정보 중복을 피하고 (b) 불필요한 인코딩 비용을 절감한다.

vectorizer.py의 FEATURE_DIM은 SBERT_DIM에 의존하므로 모델 교체 시 GRU 재학습 필수.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "jhgan/ko-sroberta-multitask"
SBERT_DIM = 768

_model = None
_load_failed = False


def _load_model():
    """싱글턴 로더 — 실패해도 한 번만 시도 후 이후 호출은 즉시 반환."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        _load_failed = True
        raise RuntimeError(
            "sentence-transformers 패키지가 설치되지 않았습니다. "
            "backend/requirements.txt 를 pip install 하세요."
        ) from e
    try:
        _model = SentenceTransformer(MODEL_NAME)
        _model.eval()
    except Exception as e:
        _load_failed = True
        raise RuntimeError(
            f"SBERT 모델 로드 실패 ({MODEL_NAME}): {e}"
        ) from e
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """문장 리스트 → (N, SBERT_DIM) numpy 행렬.

    빈 문자열('' 또는 whitespace-only)은 SBERT 호출 없이 zero row로 채운다.
    한 문장이라도 있으면 batch 인코딩으로 처리.
    """
    n = len(texts)
    out = np.zeros((n, SBERT_DIM), dtype=np.float32)
    if n == 0:
        return out

    # 빈 텍스트는 인코딩 스킵 — 인덱스만 기억해두고 non-empty만 batch encode
    non_empty_idx = [i for i, t in enumerate(texts) if t and t.strip()]
    if not non_empty_idx:
        return out

    model = _load_model()
    non_empty_texts = [texts[i] for i in non_empty_idx]
    # convert_to_numpy=True 로 torch tensor 대신 ndarray 반환 (numpy 파이프라인 호환)
    embeddings = model.encode(
        non_empty_texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    for src_i, dst_i in enumerate(non_empty_idx):
        out[dst_i] = embeddings[src_i].astype(np.float32)
    return out


def is_available() -> bool:
    """모델 로드 가능 여부 — 실패 시 False (호출자가 fallback 처리)."""
    if _load_failed:
        return False
    try:
        _load_model()
        return True
    except RuntimeError:
        return False
