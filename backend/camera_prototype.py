"""LabLog 로컬 웹캠 데모.

analyzer.LabLogAnalyzer를 호출해 웹캠 프레임을 분석하고 결과를 JSONL로 기록한다.
서버 모드는 server.py를 참조.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from analyzer import (
    AnalysisResult,
    LabLogAnalyzer,
    format_timestamp,
    needs_assist,
)

# 윈도우 콘솔에서 한글 출력이 깨지지 않도록
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OCR_INTERVAL_FRAMES = 30   # 약 1초마다 OCR 실행 (30fps 기준)
LOG_DIR = Path(__file__).parent / "logs"
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def open_camera(max_index: int = 3):
    """백엔드(MSMF → DSHOW → ANY)와 인덱스(0..max_index)를 차례로 시도."""
    backends = [
        ("MSMF", cv2.CAP_MSMF),
        ("DSHOW", cv2.CAP_DSHOW),
        ("ANY", cv2.CAP_ANY),
    ]
    for backend_name, backend_id in backends:
        for idx in range(max_index + 1):
            cap = cv2.VideoCapture(idx, backend_id)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f"[init] 카메라 연결: backend={backend_name} index={idx}")
                    return cap
                cap.release()
    return None


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    SNAPSHOT_DIR.mkdir(exist_ok=True)

    print("[init] YOLO + EasyOCR 모델 로드 중 (최초 실행 시 가중치 다운로드)...")
    analyzer = LabLogAnalyzer()

    cap = open_camera()
    if cap is None:
        raise RuntimeError(
            "카메라를 열 수 없습니다.\n"
            "다음을 확인해주십시오:\n"
            "  1) 다른 앱(Zoom, Teams, 브라우저 등)이 카메라를 사용 중인지\n"
            "  2) 설정 → 개인정보 및 보안 → 카메라\n"
            "     - '카메라 액세스' 켜짐\n"
            "     - '앱이 카메라에 액세스하도록 허용' 켜짐\n"
            "     - '데스크톱 앱이 카메라에 액세스하도록 허용' 켜짐\n"
            "  3) Windows Store 파이썬 대신 python.org 정식 배포판 설치 시도"
        )

    log_path = LOG_DIR / f"session_{int(time.time())}.jsonl"
    log_file = log_path.open("w", encoding="utf-8")

    print(f"[run] 시작. q=종료 / s=스냅샷. 로그: {log_path}")

    start = time.time()
    frame_idx = 0
    last_ocr_text = ""
    last_ocr_conf = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] 프레임을 읽지 못했습니다.")
                break

            ts = format_timestamp(time.time() - start)

            # YOLO는 매 프레임, OCR는 비용이 커서 주기적으로만 실행
            detections = analyzer.detect_objects(frame)
            for d in detections:
                x1, y1, x2, y2 = d.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    frame,
                    f"{d.label} {d.confidence:.2f}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 0),
                    2,
                )

            ran_ocr = frame_idx % OCR_INTERVAL_FRAMES == 0
            if ran_ocr:
                last_ocr_text, last_ocr_conf = analyzer.detect_text(frame)

                # 서버 응답과 동일한 스키마로 기록
                yolo_map: dict[str, float] = {}
                for d in detections:
                    if d.label not in yolo_map or d.confidence > yolo_map[d.label]:
                        yolo_map[d.label] = d.confidence
                avg_conf = (
                    sum(d.confidence for d in detections) / len(detections)
                    if detections
                    else 0.0
                )
                result = AnalysisResult(
                    timestamp=ts,
                    yolo={k: round(v, 3) for k, v in yolo_map.items()},
                    ocr=last_ocr_text,
                    ocr_confidence=round(last_ocr_conf, 3),
                    avg_yolo_confidence=round(avg_conf, 3),
                    brightness=round(float(frame.mean()), 2),
                )
                line = json.dumps(asdict(result), ensure_ascii=False)
                print(line)
                log_file.write(line + "\n")
                log_file.flush()

                if needs_assist(result):
                    print("[assist] 카메라가 가려져 있습니다. 시야를 확보해주십시오.")

            # ── 오버레이 (한글은 cv2.putText가 제대로 렌더링하지 못하므로 영문 사용) ──
            cv2.putText(
                frame,
                f"t={ts}  objs={len(detections)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if last_ocr_text:
                preview = last_ocr_text if last_ocr_text.isascii() else "[KOR text]"
                cv2.putText(
                    frame,
                    f"OCR({last_ocr_conf:.2f}): {preview[:40]}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("LabLog Prototype", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                snap = SNAPSHOT_DIR / f"snap_{int(time.time())}.jpg"
                cv2.imwrite(str(snap), frame)
                print(f"[snap] {snap}")

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print(f"[done] 로그 저장 완료: {log_path}")


if __name__ == "__main__":
    main()
