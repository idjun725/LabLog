"""학습된 YOLOv8 모델로 7세그먼트 숫자 탐지 (다중 자릿수 지원).

각 crop에서 여러 자릿수를 동시에 탐지하고, x 좌표(좌→우)로 정렬해 문자열로 합친다.
예: "1", "0", "0"이 각각 탐지되면 → "100" 반환.

가중치 파일(`seven_segment_weights.pt`)이 없을 때는 `is_available()`이 False를 반환하고
`detect_digits()`가 None을 반환한다 — graceful degradation. analyzer.py는 EasyOCR로 fallback.

학습은 train_seven_segment.py로 진행.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS_PATH = Path(__file__).parent / "seven_segment_weights.pt"

# YOLO 기본보다 다소 높게 — false positive(비-숫자 영역) 방지.
DEFAULT_CONF_THRESHOLD = 0.5

_model: YOLO | None = None
_load_failed: bool = False


def is_available() -> bool:
    """가중치 파일이 존재하는지 (경량 체크 — 매 프레임 호출 가능)."""
    return WEIGHTS_PATH.exists()


def _load_model() -> YOLO | None:
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None
    if not WEIGHTS_PATH.exists():
        _load_failed = True
        return None
    try:
        _model = YOLO(str(WEIGHTS_PATH))
        return _model
    except Exception as e:
        print(f"[7seg] 가중치 로드 실패: {e}", flush=True)
        _load_failed = True
        return None


def detect_digits(
    crop: np.ndarray, conf_threshold: float = DEFAULT_CONF_THRESHOLD
) -> tuple[str, float] | None:
    """crop에서 7세그먼트 숫자들을 탐지하고 좌→우 순으로 합친 문자열 반환.

    crop: BGR 이미지 권장 (학습 데이터가 컬러). grayscale 입력은 자동으로 3채널 변환.
    반환: (조합된_숫자_문자열, 평균_신뢰도) 또는 None.
    """
    model = _load_model()
    if model is None or crop.size == 0:
        return None

    # YOLO는 3채널 입력 필요 — grayscale이 들어오면 BGR로 복제
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    result = model(crop, verbose=False, conf=conf_threshold)[0]
    if len(result.boxes) == 0:
        return None

    # bbox + 클래스 라벨 + 신뢰도 수집
    digits: list[tuple[float, str, float]] = []
    for box in result.boxes:
        x1 = float(box.xyxy[0][0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]  # "0" ~ "9"
        digits.append((x1, label, conf))

    digits.sort(key=lambda d: d[0])  # 좌→우 정렬
    text = "".join(d[1] for d in digits)
    avg_conf = sum(d[2] for d in digits) / len(digits)
    return text, avg_conf
