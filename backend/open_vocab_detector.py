"""YOLO-World 기반 개방 어휘(open-vocabulary) 객체 탐지 — best.pt의 고정 25개 클래스
바깥에 있는 사물을 사용자가 입력한 텍스트 프롬프트로 잡아내는 보조 탐지기.

best.pt는 파인튜닝된 고정 클래스라 빠르고 정확하지만 학습에 없던 사물은 절대 못 잡는다.
YOLO-World는 반대로 어떤 텍스트든 프롬프트로 줄 수 있지만 파인튜닝 모델보다 정밀도가
낮고 느리다 — 그래서 이 모듈은 항상 best.pt와 **병행**해서만 쓰고, 대체하지 않는다.

프론트엔드에서 사용자가 실험 전 화면에서 선택/추가한 클래스 목록이 있을 때만 호출된다
(custom_classes가 비어 있으면 analyzer.py가 아예 이 모듈을 부르지 않음 — 불필요한
추론 비용을 피한다).

가중치(`yolov8s-worldv2.pt`)는 최초 호출 시 ultralytics가 자동 다운로드한다 (인터넷 필요).
다운로드/로드 실패 시 `is_available()`이 False가 되고 analyzer.py는 best.pt 결과만 사용
— graceful degradation (seven_segment_classifier.py와 동일한 패턴).
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

# v2 계열이 prompt-then-detect 정확도가 더 높음 (ultralytics 권장). "s" 크기는
# CPU 추론(로컬 개발, HF Spaces 무료 티어)과 속도-정확도 균형을 맞춘 선택.
MODEL_NAME = "yolov8s-worldv2.pt"

# 개방 어휘 탐지는 파인튜닝된 best.pt보다 신뢰도 점수가 전반적으로 낮게 나오는 경향이
# 있어 DEFAULT_CONF_THRESHOLD(0.35)보다 낮게 설정 — false negative(놓침)가 더 치명적이라
# 판단. 노이즈가 많으면 0.2~0.25로 올릴 것.
DEFAULT_CONF_THRESHOLD = 0.15

_model: YOLO | None = None
_load_failed: bool = False
_current_classes: list[str] | None = None  # set_classes 중복 호출 방지용 캐시


def is_available() -> bool:
    """모델 로드가 (아직) 실패하지 않았는지 — 경량 체크, 실제 로드는 지연 수행."""
    return not _load_failed


def _load_model() -> YOLO | None:
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None
    try:
        _model = YOLO(MODEL_NAME)
        return _model
    except Exception as e:
        print(f"[open_vocab] 모델 로드 실패: {e}", flush=True)
        _load_failed = True
        return None


def detect_objects(
    frame: np.ndarray,
    classes: list[str],
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """frame에서 classes 프롬프트로 개방 어휘 탐지. 실패/미가용 시 빈 리스트.

    반환: [(라벨, 신뢰도, (x1, y1, x2, y2)), ...] — analyzer.py가 Detection으로 변환.
    """
    global _current_classes
    if not classes:
        return []

    model = _load_model()
    if model is None:
        return []

    # set_classes는 텍스트 프롬프트를 다시 인코딩하므로 목록이 바뀌었을 때만 호출.
    if classes != _current_classes:
        model.set_classes(classes)
        _current_classes = classes

    result = model(frame, verbose=False, conf=conf_threshold)[0]
    detections = []
    for box in result.boxes:
        detections.append(
            (
                model.names[int(box.cls[0])],
                float(box.conf[0]),
                tuple(map(int, box.xyxy[0])),
            )
        )
    return detections
