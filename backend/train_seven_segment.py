r"""7세그먼트 YOLOv8 detector 학습 CLI.

사용:
  $env:ROBOFLOW_API_KEY = "your_key"
  .\.venv\Scripts\python.exe train_seven_segment.py

Roboflow → ultralytics → seven_segment_weights.pt. 캐시·백업 로직은 train_common.
"""

from __future__ import annotations

from pathlib import Path

from train_common import download_roboflow_dataset, train_and_export

ROBOFLOW_WORKSPACE = "s-workspace-qeozq"
ROBOFLOW_PROJECT = "seven-segment-display-2-n9a02-drell"
ROBOFLOW_VERSION = 1

DATA_DIR = Path(__file__).parent / "seven_segment_data"
WEIGHTS_PATH = Path(__file__).parent / "seven_segment_weights.pt"
RUNS_DIR = Path(__file__).parent / "runs"
RUN_NAME = "seven_segment"
BASE_MODEL = "yolov8n.pt"
EPOCHS = 30
IMGSZ = 320


def main() -> None:
    data_path = download_roboflow_dataset(
        ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION, DATA_DIR
    )
    train_and_export(
        base_model=BASE_MODEL,
        data_yaml=data_path / "data.yaml",
        runs_dir=RUNS_DIR,
        run_name=RUN_NAME,
        weights_out=WEIGHTS_PATH,
        epochs=EPOCHS,
        imgsz=IMGSZ,
    )
    print("[train] 추론은 analyzer.detect_text_in_regions가 자동으로 라우팅합니다.")


if __name__ == "__main__":
    main()
