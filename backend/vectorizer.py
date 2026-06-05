"""기획서 2단계 — records를 PyTorch 입력용 수치 텐서로 변환.

records의 각 항목(YOLO + OCR + STT)을 고정 차원 특징 벡터로 변환한다.
출력 형태: (T, FEATURE_DIM) 행렬 — T는 records 개수, FEATURE_DIM은 23.

설계 원칙:
- YOLO: 미리 정의된 라벨 vocab의 한 자리당 최고 신뢰도. vocab에 없는 라벨은 무시.
- OCR: 텍스트 유무, 신뢰도, 단위(질량/부피/온도/pH) 4종 binary.
- STT: 발화 유무 + 길이 정규화.

vocab 확장은 YOLO 파인튜닝 후 자연스럽게 필요해진다. 그땐 이 파일의 YOLO_VOCAB을
업데이트하고 GRU를 재학습하면 된다.
"""

from __future__ import annotations

import re

import numpy as np

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

OCR_DIM = 2 + len(UNIT_PATTERNS)  # has_text + confidence + 단위 4종
SPEECH_DIM = 2                    # has_speech + word_count_normalized
FEATURE_DIM = YOLO_DIM + OCR_DIM + SPEECH_DIM  # 15 + 6 + 2 = 23


def vectorize_record(record: dict) -> np.ndarray:
    """단일 record → 길이 FEATURE_DIM 의 1D 특징 벡터."""
    # YOLO part
    yolo = record.get("yolo") or {}
    yolo_vec = np.zeros(YOLO_DIM, dtype=np.float32)
    for label, conf in yolo.items():
        if label in YOLO_VOCAB:
            yolo_vec[YOLO_VOCAB.index(label)] = float(conf)

    # OCR part
    ocr_text = record.get("ocr", "") or ""
    ocr_conf = float(record.get("ocr_confidence", 0.0) or 0.0)
    ocr_vec = np.zeros(OCR_DIM, dtype=np.float32)
    ocr_vec[0] = 1.0 if ocr_text else 0.0
    ocr_vec[1] = ocr_conf
    for i, pat in enumerate(UNIT_PATTERNS.values()):
        ocr_vec[2 + i] = 1.0 if pat.search(ocr_text) else 0.0

    # STT part
    speech = record.get("speech", "") or ""
    word_count = len(speech.split())
    speech_vec = np.array(
        [
            1.0 if speech else 0.0,
            min(word_count / 20.0, 1.0),  # 20 단어 기준 정규화 → [0, 1]
        ],
        dtype=np.float32,
    )

    return np.concatenate([yolo_vec, ocr_vec, speech_vec])


def vectorize_records(records: list[dict]) -> np.ndarray:
    """records 리스트 → 2D (T, FEATURE_DIM) 행렬."""
    if not records:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack([vectorize_record(r) for r in records])
