"""기획서 2단계 — records를 PyTorch 입력용 수치 텐서로 변환.

records의 각 항목(YOLO + OCR + STT)을 고정 차원 특징 벡터로 변환한다.
출력 형태: (T, FEATURE_DIM) 행렬 — T는 records 개수.

**FEATURE_DIM 구성** (SBERT 통합 후):
- YOLO 25차원 one-hot 최고 신뢰도 (`YOLO_VOCAB` 길이만큼)
- OCR 6차원: has_text + confidence + 단위 4종(mass/volume/temp/ph) binary
- STT 2차원 (baseline): has_speech + word_count/20
- STT 768차원 (SBERT): `jhgan/ko-sroberta-multitask` 문장 임베딩
- 합계: 25 + 6 + 2 + 768 = 801차원 (production, use_sbert=True)
- Baseline: 25 + 6 + 2 = 33차원 (use_sbert=False — 평가/비교용)

설계 원칙:
- YOLO: 미리 정의된 라벨 vocab의 한 자리당 최고 신뢰도. vocab에 없는 라벨은 무시.
- OCR: 텍스트 유무, 신뢰도, 단위(질량/부피/온도/pH) 4종 binary.
- STT: 발화 유무 + 길이 정규화 + SBERT 임베딩(의미).

vocab 확장 시 `YOLO_VOCAB` 업데이트 → GRU 재학습. SBERT 모델 교체 시도 재학습 필수.
"""

from __future__ import annotations

import re

import numpy as np

from sbert_encoder import SBERT_DIM, encode as sbert_encode

# YOLO 라벨 vocab — best.pt(2026-05 fine-tuned, 25개 실험기구 클래스)의 model.names와 일치.
# 순서가 곧 특징 벡터의 차원 인덱스이므로 임의 순서 변경 금지(GRU 가중치와 호환성 깨짐).
# 변경 시 train_gru.py 재실행 필수.
YOLO_VOCAB = [
    "Beaker",
    "Buchner_Funnel",
    "Burette_Stands",
    "Calorimeter",
    "Conical_Flask",
    "Funnel",
    "Glass_Rod",
    "Measuring_Cylinder",
    "Mechanical_Balance_Scale",
    "Nessler_Reagent_Bottle",
    "Pipette",
    "Porcelain_Mortar Pestle",
    "Precision_Weight_Scale",
    "Reagent_Bottle",
    "Round_Bottom_Flask_Borosilicate_Glass_1_Neck",
    "Round_Bottom_Flask_Borosilicate_Glass_2_Neck",
    "Round_Bottom_Flask_Borosilicate_Glass_3_Neck",
    "Separating_Funnel",
    "Spirit_Lamp",
    "TestTube_Holder",
    "Test_Tube",
    "Volumetric_Flask",
    "Volumetric_Pipet",
    "Wash_Bottle",
    "Weighing_Bottle",
]
YOLO_DIM = len(YOLO_VOCAB)

# OCR 측정값 단위 인코딩 — 단위 종류가 단계 분류에 강한 신호가 됨.
# "4.0g" → 측정 단계, "pH 7.4" → 관찰 단계, "25°C" → 반응 단계 등.
UNIT_PATTERNS = {
    "mass": re.compile(r"\d+\.?\d*\s*(g|kg|mg)\b", re.IGNORECASE),
    "volume": re.compile(r"\d+\.?\d*\s*(ml|l|cc)\b", re.IGNORECASE),
    "temp": re.compile(r"\d+\.?\d*\s*°?\s*[cf]\b", re.IGNORECASE),
    "ph": re.compile(r"\bph\s*\d", re.IGNORECASE),
}

OCR_DIM = 2 + len(UNIT_PATTERNS)  # has_text + confidence + 단위 4종 = 6
SPEECH_BASELINE_DIM = 2            # has_speech + word_count_normalized

BASELINE_FEATURE_DIM = YOLO_DIM + OCR_DIM + SPEECH_BASELINE_DIM       # 25 + 6 + 2 = 33
SBERT_FEATURE_DIM = BASELINE_FEATURE_DIM + SBERT_DIM                   # 33 + 768 = 801

# production 기본값. gru_classifier / train_gru가 참조하는 상수.
FEATURE_DIM = SBERT_FEATURE_DIM


def feature_dim(use_sbert: bool = True) -> int:
    """모드별 특징 차원 반환 — 평가 스크립트가 baseline/sbert를 오갈 때 사용."""
    return SBERT_FEATURE_DIM if use_sbert else BASELINE_FEATURE_DIM


def _vectorize_non_text(record: dict) -> np.ndarray:
    """YOLO + OCR + STT baseline(2-dim) 특징만 뽑는다 (SBERT 미포함)."""
    yolo = record.get("yolo") or {}
    yolo_vec = np.zeros(YOLO_DIM, dtype=np.float32)
    for label, conf in yolo.items():
        if label in YOLO_VOCAB:
            yolo_vec[YOLO_VOCAB.index(label)] = float(conf)

    ocr_text = record.get("ocr", "") or ""
    ocr_conf = float(record.get("ocr_confidence", 0.0) or 0.0)
    ocr_vec = np.zeros(OCR_DIM, dtype=np.float32)
    ocr_vec[0] = 1.0 if ocr_text else 0.0
    ocr_vec[1] = ocr_conf
    for i, pat in enumerate(UNIT_PATTERNS.values()):
        ocr_vec[2 + i] = 1.0 if pat.search(ocr_text) else 0.0

    speech = record.get("speech", "") or ""
    word_count = len(speech.split())
    speech_vec = np.array(
        [
            1.0 if speech else 0.0,
            min(word_count / 20.0, 1.0),  # 20 단어 기준 정규화 → [0, 1]
        ],
        dtype=np.float32,
    )

    return np.concatenate([yolo_vec, ocr_vec, speech_vec])  # (BASELINE_FEATURE_DIM,)


def vectorize_record(record: dict) -> np.ndarray:
    """단일 record → BASELINE_FEATURE_DIM(33) 1D 벡터. SBERT 미포함.

    남아있는 이유: 하위 호환. 실제 학습/추론 파이프라인은 vectorize_records를 쓴다.
    """
    return _vectorize_non_text(record)


def vectorize_records(records: list[dict], use_sbert: bool = True) -> np.ndarray:
    """records 리스트 → 2D (T, D) 행렬.

    use_sbert=True(기본): 발화 텍스트를 배치로 SBERT 인코딩해 concat. D=SBERT_FEATURE_DIM.
    use_sbert=False: 기존 33-dim baseline만 사용. D=BASELINE_FEATURE_DIM.

    빈 발화는 sbert_encoder가 zero row로 채우므로 별도 처리 불필요.
    """
    dim = feature_dim(use_sbert)
    if not records:
        return np.zeros((0, dim), dtype=np.float32)

    base = np.stack([_vectorize_non_text(r) for r in records])  # (T, 33)
    if not use_sbert:
        return base

    speech_texts = [(r.get("speech", "") or "") for r in records]
    sbert_vecs = sbert_encode(speech_texts)  # (T, 768)
    return np.concatenate([base, sbert_vecs], axis=1)  # (T, 801)
