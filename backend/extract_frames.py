"""실험 영상 → 학습용 프레임 추출 (YOLO fine-tuning 라벨링 후보 생성).

본인이 촬영한 실험 영상에서 N초 간격으로 JPG를 뽑아 Roboflow 웹 UI에 업로드 →
bounding box 라벨링 → YOLO 학습 데이터셋으로 사용한다.

사용 예:
  python extract_frames.py video.mp4 --out frames/ --interval 2.0
  python extract_frames.py video.mp4 --interval 1.5 --max 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: str,
    out_dir: str = "frames",
    interval_sec: float = 2.0,
    max_frames: int | None = None,
) -> int:
    """video_path에서 interval_sec 간격으로 JPG 프레임 추출. 저장된 개수 반환.

    max_frames가 지정되면 그 수만큼만 저장 (긴 영상에서 균일 샘플링용은 아님 — 앞에서부터).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(int(fps * interval_sec), 1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem
    saved = 0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                ts_sec = idx / fps
                fname = f"{stem}_{int(ts_sec):05d}s.jpg"
                cv2.imwrite(str(out_path / fname), frame)
                saved += 1
                if max_frames is not None and saved >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    duration = total_frames / fps if total_frames else idx / fps
    print(f"저장 {saved}개 (영상 {duration:.1f}초, {fps:.1f}fps, {interval_sec}초 간격)")
    print(f"출력 경로: {out_path.resolve()}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="영상에서 라벨링용 프레임 추출")
    parser.add_argument("video", help="입력 영상 파일")
    parser.add_argument("--out", default="frames", help="출력 디렉토리 (기본: ./frames)")
    parser.add_argument("--interval", type=float, default=2.0, help="추출 간격(초) (기본: 2.0)")
    parser.add_argument("--max", type=int, default=None, help="최대 저장 프레임 수 (기본: 제한 없음)")
    args = parser.parse_args()
    extract_frames(args.video, args.out, args.interval, args.max)
