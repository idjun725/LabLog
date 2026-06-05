"""학습된 GRU로 records 시퀀스의 단계 분류 (기획서 3단계 추론 단계).

가중치 파일(gru_weights.pt)이 없으면 RuntimeError. 학습을 먼저 진행해야 함:
  python train_gru.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from gru_model import GRUStageClassifier, PHASES
from vectorizer import FEATURE_DIM, vectorize_records

WEIGHTS_PATH = Path(__file__).parent / "gru_weights.pt"

# 첫 호출에서 한 번 로드 후 메모리에 캐시 (모델 자체는 가벼움 — 수십KB)
_model: GRUStageClassifier | None = None


def _load_model() -> GRUStageClassifier:
    global _model
    if _model is not None:
        return _model

    if not WEIGHTS_PATH.exists():
        raise RuntimeError(
            f"GRU 가중치 파일이 없습니다: {WEIGHTS_PATH.name}\n"
            f"먼저 학습을 진행하세요: python train_gru.py"
        )

    model = GRUStageClassifier(input_dim=FEATURE_DIM)
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # 가장 흔한 케이스: PHASES 개수 또는 FEATURE_DIM이 바뀐 뒤 재학습 미실행
        raise RuntimeError(
            f"GRU 가중치 파일이 현재 모델 구조와 호환되지 않습니다 "
            f"(PHASES 또는 FEATURE_DIM이 변경된 뒤 재학습되지 않은 것 같습니다). "
            f"다음을 실행하세요: python train_gru.py\n"
            f"원본 오류: {e}"
        ) from e
    model.eval()
    _model = model
    return model


def classify_records(records: list[dict]) -> list[str]:
    """records 시퀀스를 받아 각 record의 단계 라벨을 예측해 반환.

    반환: ['준비', '측정', ...] 길이 len(records)와 동일.
    """
    if not records:
        return []

    model = _load_model()
    x = vectorize_records(records)  # (T, D)
    tensor = torch.from_numpy(x).unsqueeze(0)  # (1, T, D)
    with torch.no_grad():
        logits = model(tensor)  # (1, T, NUM_PHASES)
        preds = logits.argmax(dim=-1).squeeze(0).tolist()
    return [PHASES[p] for p in preds]
