r"""메인 YOLO 객체 탐지 모델 학습 CLI.

Roboflow의 chemistry-lab-object-detection 데이터셋을 yolov8n.pt 위에 fine-tune해
실험기구 탐지 정확도를 높인다. 학습 완료 시 결과를 best.pt로 복사 (기존 백업).

사용:
  $env:ROBOFLOW_API_KEY = "your_key"
  .\.venv\Scripts\python.exe train_main_yolo.py

CPU 학습은 데이터셋 크기에 따라 수 시간 걸린다 — Kaggle GPU 오프로딩 권장
(KAGGLE_TRAINING.md 참고).
"""

from __future__ import annotations

from pathlib import Path

from train_common import download_roboflow_dataset, train_and_export

ROBOFLOW_WORKSPACE = "s-workspace-qeozq"
ROBOFLOW_PROJECT = "chemistry-lab-object-detection-topas"
ROBOFLOW_VERSION = 1

DATA_DIR = Path(__file__).parent / "chemistry_lab_data"
WEIGHTS_PATH = Path(__file__).parent / "best.pt"  # 분석에 사용되는 최종 가중치
RUNS_DIR = Path(__file__).parent / "runs"
RUN_NAME = "main_yolo"
BASE_MODEL = "yolov8n.pt"
EPOCHS = 50
IMGSZ = 640
PATIENCE = 15  # val mAP 정체 시 조기 종료


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
        patience=PATIENCE,
    )
    print(
        "[train] 서버 재시작이 필요합니다 — "
        "LabLogAnalyzer가 시작 시 1회만 모델을 로드하므로."
    )


if __name__ == "__main__":
    main()
