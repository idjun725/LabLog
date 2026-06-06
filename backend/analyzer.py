"""LabLog 멀티모달 분석 모듈.

YOLO(객체) + EasyOCR(텍스트) + Groq Whisper(음성)를 한 결과 객체에 통합한다.
server.py(FastAPI)가 호출한다.
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import timedelta

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

from stt import extract_and_transcribe, find_speech_at
import seven_segment_classifier

DEFAULT_YOLO_WEIGHTS = "best.pt"           # 실험기구 25개 클래스 fine-tuned (2026-05)
DEFAULT_OCR_LANGS = ["ko", "en"]
DEFAULT_CONF_THRESHOLD = 0.35

# ── OCR 후처리 필터 — 잡음·오탐(false positive) 감소용 ─────────────────
# 실험 측정값은 보통 "숫자 + 단위" 형태 ("4.0g", "100ml", "pH 7.4")라는 점을 활용한다.
# 너무 짧거나, 신뢰도가 낮거나, 글자 대비 숫자 비율이 너무 낮은 인식은 노이즈로 보고 버린다.
NUMERIC_PATTERN = re.compile(r"\d")
OCR_MIN_CONFIDENCE = 0.5   # EasyOCR 단일 박스 신뢰도 임계값 (0~1, 기본 통과 0.0)
OCR_MIN_LENGTH = 2         # 1글자 인식은 거의 노이즈로 가정
OCR_MIN_DIGIT_RATIO = 0.25 # 공백 제외 글자 중 숫자 비율의 최소값
OCR_TEXT_SEPARATOR = " | " # 여러 crop의 OCR 결과를 하나의 문자열로 합칠 때 구분자

# ── 7세그먼트 detector 라우팅 ───────────────────────────────────────────
# 7세그먼트 YOLOv8 detector는 EasyOCR보다 훨씬 빨라서 전체 프레임에 적용 (다중 자릿수 지원).
# 디지털 디스플레이는 7-seg가 잡고, 일반 라벨/측정값은 EasyOCR이 crop에서 잡는 역할 분담.
# conf 임계값은 seven_segment_classifier.DEFAULT_CONF_THRESHOLD가 소유.


def _filter_ocr_raw(raw) -> list[tuple[str, float]]:
    """EasyOCR readtext 결과에 노이즈 필터 적용."""
    filtered: list[tuple[str, float]] = []
    for _, text, conf in raw:
        text = text.strip()
        if len(text) < OCR_MIN_LENGTH:
            continue
        if float(conf) < OCR_MIN_CONFIDENCE:
            continue
        if not NUMERIC_PATTERN.search(text):
            continue
        non_space = text.replace(" ", "")
        if not non_space:
            continue
        digit_ratio = sum(c.isdigit() for c in non_space) / len(non_space)
        if digit_ratio < OCR_MIN_DIGIT_RATIO:
            continue
        filtered.append((text, float(conf)))
    return filtered

# ── YOLO 시간적 일관성 필터 — 깜빡이는 오탐 제거용 ─────────────────────
# 최근 PERSISTENCE_WINDOW 샘플 중 PERSISTENCE_MIN_VOTES번 이상 등장한 라벨만 인정.
# 단, 현재 프레임에 그 라벨이 실제로 있어야 함 (사라진 객체를 계속 표시하지 않도록).
# 신뢰도가 아무리 높아도 한 번만 반짝하고 사라지는 라벨은 잡음으로 간주해 제거된다.
PERSISTENCE_WINDOW = 4
PERSISTENCE_MIN_VOTES = 2


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class AnalysisResult:
    """프론트엔드 ↔ 백엔드 사이의 JSON 계약. 모델이 교체돼도 이 형식은 유지된다."""
    timestamp: str
    yolo: dict[str, float]          # {라벨: 최고 신뢰도} — 요약용
    detections: list[Detection]     # 박스 좌표 포함 — 실시간 오버레이용
    ocr: str
    ocr_confidence: float
    speech: str                     # Groq Whisper 전사 (해당 시각 발화)
    avg_yolo_confidence: float
    brightness: float
    frame_width: int                # 박스 좌표를 매핑할 원본 프레임 크기
    frame_height: int


def format_timestamp(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds))).zfill(8)


def parse_timestamp(ts: str) -> float:
    """'HH:MM:SS' → 초. format_timestamp의 역함수. 잘못된 형식이면 0.0."""
    try:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return 0.0


def get_assist_messages(
    result: AnalysisResult,
    yolo_conf_threshold: float = 0.4,
    ocr_conf_threshold: float = 0.65,
    brightness_threshold: float = 40.0,
) -> list[str]:
    """기획서의 '실시간 영상 보조 기능' — 화면에 띄울 안내 메시지 리스트.

    문제 별로 메시지가 다르므로 list로 반환 (여러 조건이 동시에 충족 가능).
    빈 리스트면 안내 불필요.

    트리거 조건:
    - YOLO 평균 신뢰도가 낮음 → 시야 확보 안내
    - OCR 신뢰도가 임계값 미만 (단, 인식된 텍스트가 있을 때만) → 텍스트 가까이 비춤 안내
    - 평균 밝기가 낮음 → 조명 안내
    """
    msgs: list[str] = []
    if result.brightness < brightness_threshold:
        msgs.append("화면이 어둡습니다. 조명을 확보해주십시오.")
    if bool(result.yolo) and result.avg_yolo_confidence < yolo_conf_threshold:
        msgs.append("객체 인식이 어렵습니다. 카메라 시야를 확보해주십시오.")
    if bool(result.ocr) and 0 < result.ocr_confidence < ocr_conf_threshold:
        msgs.append("화면 텍스트 인식이 어렵습니다. 가까이 또는 또렷하게 비춰주십시오.")
    return msgs


class LabLogAnalyzer:
    def __init__(
        self,
        yolo_weights: str = DEFAULT_YOLO_WEIGHTS,
        ocr_langs: list[str] | None = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        gpu: bool = False,
    ):
        self.yolo = YOLO(yolo_weights)
        self.ocr = easyocr.Reader(ocr_langs or DEFAULT_OCR_LANGS, gpu=gpu)
        self.conf_threshold = conf_threshold

    def detect_objects(self, frame: np.ndarray) -> list[Detection]:
        result = self.yolo(frame, verbose=False, conf=self.conf_threshold)[0]
        detections = []
        for box in result.boxes:
            detections.append(
                Detection(
                    label=self.yolo.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    box=tuple(map(int, box.xyxy[0])),
                )
            )
        return detections

    def detect_text_in_regions(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> tuple[str, float, str]:
        """7세그먼트 YOLO (전체 프레임) + EasyOCR (YOLO bbox crop만) 병렬 적용.

        - 7세그먼트 YOLO는 빠르므로 **전체 프레임**에 1회 적용 — 디지털 디스플레이를
          best.pt가 탐지 못 한 위치(예: 스톱워치)에서도 캡처.
        - EasyOCR은 무거우므로 **YOLO 탐지 박스**에만 적용 — 일반 라벨/측정값 인석 담당.
        - 텍스트가 중복되면 (예: 두 모델이 같은 디스플레이를 모두 읽음) dedup.

        반환: (결합_텍스트, 평균_신뢰도, seg7_텍스트).
        seg7_텍스트는 호출 측이 persistence filter를 우회하는 판단에 쓰도록 따로 노출.
        7-seg는 자체 conf threshold로 이미 검증되므로 매 프레임 반복 검증할 필요가 없다.
        """
        all_texts: list[tuple[str, float]] = []
        seg7_text = ""

        if seven_segment_classifier.is_available():
            seg7_full = seven_segment_classifier.detect_digits(frame)
            if seg7_full is not None:
                seg7_text = seg7_full[0]
                all_texts.append(seg7_full)

        h, w = frame.shape[:2]
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            all_texts.extend(_filter_ocr_raw(self.ocr.readtext(binary)))

        if not all_texts:
            return "", 0.0, seg7_text

        # Dedup: 두 모델이 같은 텍스트를 잡았을 때 첫 confidence만 (dict 키 순서로 순서 보존)
        unique: dict[str, float] = {}
        for text, conf in all_texts:
            unique.setdefault(text, conf)
        combined = OCR_TEXT_SEPARATOR.join(unique)
        avg_conf = sum(unique.values()) / len(unique)
        return combined, avg_conf, seg7_text

    def _build_result(
        self,
        frame: np.ndarray,
        timestamp: str,
        detections: list[Detection],
        ocr_text: str,
        ocr_conf: float,
        speech: str = "",
    ) -> AnalysisResult:
        yolo_map: dict[str, float] = {}
        for d in detections:
            if d.label not in yolo_map or d.confidence > yolo_map[d.label]:
                yolo_map[d.label] = d.confidence
        avg_conf = (
            sum(d.confidence for d in detections) / len(detections)
            if detections
            else 0.0
        )
        h, w = frame.shape[:2]
        return AnalysisResult(
            timestamp=timestamp,
            yolo={k: round(v, 3) for k, v in yolo_map.items()},
            detections=detections,
            ocr=ocr_text,
            ocr_confidence=round(ocr_conf, 3),
            speech=speech,
            avg_yolo_confidence=round(avg_conf, 3),
            brightness=round(float(frame.mean()), 2),
            frame_width=int(w),
            frame_height=int(h),
        )

    def analyze_frame(self, frame: np.ndarray, timestamp: str) -> AnalysisResult:
        detections = self.detect_objects(frame)
        ocr_text, ocr_conf, _seg7 = self.detect_text_in_regions(frame, detections)
        # 실시간 단일 프레임에서는 STT를 돌리지 않는다 (스트리밍 STT는 /ws/transcribe가 담당)
        return self._build_result(frame, timestamp, detections, ocr_text, ocr_conf, speech="")

    def analyze_video(
        self,
        path: str,
        sample_fps: float = 1.0,
        ocr_period_seconds: float = 3.0,
    ) -> list[AnalysisResult]:
        """변화 감지 기반 멀티모달 영상 분석 (STT/비디오 병렬 처리).

        STT(I/O 바운드 — Google API)와 비디오 분석(CPU 바운드 — YOLO/OCR)을 동시에
        실행해 총 처리 시간을 max(STT, 비디오)로 줄인다. 자원이 겹치지 않아 병렬 안전.

        - sample_fps Hz로 프레임을 추출.
        - 각 샘플에서 YOLO는 항상 실행 (빠름).
        - OCR(느림)은 YOLO 객체 집합이 바뀌거나 ocr_period_seconds 이상 지난 경우에만 실행.
        - record 저장은 (객체 집합 변화) ∨ (새/변경된 OCR 텍스트) 발생 시 — 첫 샘플은 무조건 저장.
        - speech는 비디오 루프 종료 후 후처리로 각 record에 첨부.
          (병렬 처리 영향으로 speech-only 변화는 dedup 트리거에서 빠짐 — 발화 텍스트는
          가장 가까운 record의 speech 필드에 들어가므로 보고서 단계에서 정보는 보존됨.)
        """
        t_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # STT를 백그라운드 스레드에서 시작 — 영상 분석과 자원이 겹치지 않으므로 안전
            stt_future = executor.submit(extract_and_transcribe, path)

            # ── 비디오 분석 (메인 스레드) ───────────────────────────────
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError(f"비디오를 열 수 없습니다: {path}")

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            total_seconds = total_frames / src_fps if src_fps > 0 else 0.0
            step = max(int(src_fps / sample_fps), 1)
            est_samples = max(1, total_frames // step) if total_frames else 0
            ocr_period_samples = max(int(ocr_period_seconds * sample_fps), 1)
            print(
                f"[analyze] 비디오 분석 시작 — 길이 {total_seconds:.1f}초, "
                f"{src_fps:.1f}fps, 예상 샘플 {est_samples}개 (sample_fps={sample_fps})",
                flush=True,
            )

            results: list[AnalysisResult] = []
            t_secs: list[float] = []  # records와 평행하게 시각을 보관 (후처리에서 speech 매칭용)
            prev_objects: frozenset[str] | None = None
            prev_ocr: str = ""
            prev_seg7: str = ""  # 7-seg 텍스트는 persistence를 우회하므로 별도 추적
            last_ocr_sample = -ocr_period_samples  # 첫 프레임에서 무조건 OCR 실행되도록
            # 최근 PERSISTENCE_WINDOW 샘플의 YOLO 라벨 집합 — 깜빡이는 라벨 제거용
            recent_label_sets: deque[set[str]] = deque(maxlen=PERSISTENCE_WINDOW)
            # OCR 시간적 일관성 필터 — YOLO와 같은 WINDOW/MIN_VOTES 상수 사용 (EasyOCR 부분만)
            recent_ocr_texts: deque[str] = deque(maxlen=PERSISTENCE_WINDOW)

            idx = 0
            sample_idx = 0
            t_loop = time.monotonic()
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % step != 0:
                        idx += 1
                        continue

                    raw_detections = self.detect_objects(frame)
                    # 시간적 일관성 필터 — 최근 윈도우에서 PERSISTENCE_MIN_VOTES번 이상 보이고
                    # 동시에 현재 프레임에 존재하는 라벨만 인정 (반짝 객체는 잡음으로 제거).
                    raw_labels = {d.label for d in raw_detections}
                    recent_label_sets.append(raw_labels)
                    label_counts: Counter[str] = Counter()
                    for prev_set in recent_label_sets:
                        label_counts.update(prev_set)
                    confirmed_labels = {
                        lbl
                        for lbl, cnt in label_counts.items()
                        if cnt >= PERSISTENCE_MIN_VOTES and lbl in raw_labels
                    }
                    detections = [d for d in raw_detections if d.label in confirmed_labels]

                    current_objects = frozenset(d.label for d in detections)
                    yolo_changed = current_objects != prev_objects
                    ocr_due = (sample_idx - last_ocr_sample) >= ocr_period_samples

                    if yolo_changed or ocr_due:
                        ocr_text, ocr_conf, seg7_text = self.detect_text_in_regions(
                            frame, detections
                        )
                        last_ocr_sample = sample_idx
                    else:
                        ocr_text, ocr_conf, seg7_text = "", 0.0, ""

                    # 7-seg는 자체 conf threshold로 이미 검증되므로 persistence 우회 — 변화가
                    # 보이는 즉시 record에 반영해야 디지털 측정값(스톱워치·온도) 추적이 가능하다.
                    seg7_changed = bool(seg7_text) and seg7_text != prev_seg7
                    if seg7_changed:
                        print(
                            f"[analyze] seg7 변화: '{prev_seg7}' → '{seg7_text}' (샘플 {sample_idx})",
                            flush=True,
                        )

                    # EasyOCR 부분은 시간적 일관성 필터 — 최근 window에서 min_votes 이상
                    # 반복된 결합 텍스트만 유효 (YOLO의 PERSISTENCE_WINDOW/MIN_VOTES 공유).
                    recent_ocr_texts.append(ocr_text)
                    text_vote_count = recent_ocr_texts.count(ocr_text)
                    is_persistent_ocr = (
                        text_vote_count >= PERSISTENCE_MIN_VOTES and bool(ocr_text)
                    )
                    ocr_changed = (
                        seg7_changed
                        or (is_persistent_ocr and ocr_text != prev_ocr)
                    )
                    is_first = sample_idx == 0

                    if is_first or yolo_changed or ocr_changed:
                        t_sec = idx / src_fps
                        ts = format_timestamp(t_sec)
                        # speech는 STT 결과를 기다린 뒤 후처리에서 채움
                        results.append(
                            self._build_result(
                                frame, ts, detections, ocr_text, ocr_conf, speech=""
                            )
                        )
                        t_secs.append(t_sec)
                        prev_objects = current_objects
                        if ocr_text:
                            prev_ocr = ocr_text
                        if seg7_text:
                            prev_seg7 = seg7_text

                    sample_idx += 1
                    idx += 1

                    # 매 10 샘플마다 진척 로그 (예상 샘플 대비 %)
                    if sample_idx % 10 == 0:
                        pct = (sample_idx / est_samples * 100) if est_samples else 0.0
                        print(
                            f"[analyze] 샘플 {sample_idx}/{est_samples} "
                            f"({pct:.0f}%, records {len(results)}, "
                            f"경과 {time.monotonic() - t_loop:.0f}초)",
                            flush=True,
                        )
            finally:
                cap.release()

            print(
                f"[analyze] 비디오 루프 종료 — 샘플 {sample_idx}, "
                f"records {len(results)} (소요 {time.monotonic() - t_loop:.1f}초)",
                flush=True,
            )
            print("[analyze] STT 결과 대기 중...", flush=True)
            t_wait = time.monotonic()
            stt_segments = stt_future.result()
            print(
                f"[analyze] STT 대기 종료 ({time.monotonic() - t_wait:.1f}초, "
                f"세그먼트 {len(stt_segments)}개)",
                flush=True,
            )

        # ── 후처리: 각 record에 해당 시각의 speech 첨부 ───────────────
        for record, t_sec in zip(results, t_secs):
            record.speech = find_speech_at(t_sec, stt_segments)

        print(
            f"[analyze] 전체 분석 완료 — 총 {time.monotonic() - t_start:.1f}초, "
            f"records {len(results)}개",
            flush=True,
        )
        return results
