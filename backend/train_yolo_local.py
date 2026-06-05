"""LabLog YOLOv8 로컬 학습 스크립트 (Colab 대안).

GPU 자동 감지 후 학습. Roboflow에서 데이터셋을 받아 yolov8n.pt를 fine-tuning한다.

사전 준비:
  1. venv 활성화 후 roboflow 설치:
       .\.venv\Scripts\python.exe -m pip install roboflow
  2. 아래 ROBOFLOW_* 상수를 본인 데이터셋 snippet으로 교체
     (Roboflow → Dataset → Versions → Download Dataset → YOLOv8 → Show download code)
  3. GPU 종류에 맞춰 가속 라이브러리 설치:
     - NVIDIA: CUDA용 torch (https://pytorch.org/get-started/locally/)
     - AMD (Windows): DirectML
         .\.venv\Scripts\python.exe -m pip install torch-directml
     - 둘 다 없으면 CPU 모드 (매우 느림 — Colab 권장)

실행:
  .\.venv\Scripts\python.exe train_yolo_local.py

출력:
  runs/lablog_yolo/weights/best.pt  ← 이 파일을 backend/로 복사 후 통합
"""

from __future__ import annotations

import os
import sys
import time

import torch

# ── Roboflow snippet — 본인 데이터셋 정보로 교체 ─────────────────────────
# API key는 환경변수 ROBOFLOW_API_KEY로 설정 (저장소에 평문 commit 금지).
#   PowerShell: $env:ROBOFLOW_API_KEY = 'your-key-here'
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY")
ROBOFLOW_WORKSPACE = "YOUR_WORKSPACE"
ROBOFLOW_PROJECT = "YOUR_PROJECT"
ROBOFLOW_VERSION = 1

# ── 학습 하이퍼파라미터 ──────────────────────────────────────────────────
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16        # OOM 시 8 또는 4로 낮춤. DirectML은 자동으로 8로 시작.
PATIENCE = 10          # N epoch 동안 mAP 개선 없으면 조기 종료
RUN_NAME = "lablog_yolo"


def detect_device() -> tuple[object, int, bool]:
    """가속기 자동 감지 — (device, 권장_batch_size, AMP 사용 여부) 반환.

    우선순위: CUDA (NVIDIA) → DirectML (AMD/Intel on Windows) → CPU.
    AMP(자동 혼합 정밀도)는 CUDA만 안정적. DirectML/CPU는 비활성화해야 한다 —
    ultralytics의 check_amp가 torch.cuda.get_device_name을 무조건 호출해서
    AMD GPU + DirectML 조합에서 'Torch not compiled with CUDA enabled' 오류 발생.
    """
    if torch.cuda.is_available():
        print(f"✓ CUDA GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        return 0, BATCH_SIZE, True

    try:
        import torch_directml  # type: ignore
        dml = torch_directml.device()
        name = torch_directml.device_name(0)
        print(f"✓ DirectML GPU: {name}")
        print("  AMP 비활성화 (DirectML 호환성). 일부 연산이 fallback될 수 있습니다.")
        return dml, min(BATCH_SIZE, 8), False
    except ImportError:
        pass

    print("⚠️  GPU 감지 안 됨 — CPU로 학습 (매우 느림, 권장 X)")
    print("   계속하려면 5초 안에 Ctrl+C로 취소하지 않으면 진행")
    time.sleep(5)
    return "cpu", min(BATCH_SIZE, 4), False


def main():
    print("=" * 60)
    print("LabLog YOLOv8 로컬 학습")
    print("=" * 60)
    device, batch, use_amp = detect_device()
    print()

    # 1. Roboflow 데이터셋 다운로드
    print("[1/3] Roboflow 데이터셋 다운로드 중...")
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow 미설치 — 다음 명령 먼저 실행:")
        print("  .\\.venv\\Scripts\\python.exe -m pip install roboflow")
        sys.exit(1)

    if ROBOFLOW_API_KEY == "YOUR_API_KEY":
        print("ERROR: 이 스크립트 상단의 ROBOFLOW_* 상수를 본인 값으로 교체하세요.")
        sys.exit(1)

    t0 = time.monotonic()
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov8")
    print(f"   완료 ({time.monotonic() - t0:.0f}초) — {dataset.location}")
    print()

    # 2. 학습
    print(f"[2/3] 학습 시작 — epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={batch}")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    t0 = time.monotonic()
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=batch,
        patience=PATIENCE,
        device=device,
        amp=use_amp,
        project="runs",
        name=RUN_NAME,
        exist_ok=True,
    )
    print(f"   학습 완료 ({(time.monotonic() - t0) / 60:.1f}분)")
    print()

    # 3. 검증
    print("[3/3] 검증 (val set)...")
    metrics = model.val()
    print(f"   mAP50:    {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")
    print()
    print("클래스별 mAP50:")
    for i, name in enumerate(model.names.values()):
        print(f"  {name:50s} {metrics.box.maps[i]:.3f}")
    print()

    best_pt = f"runs/{RUN_NAME}/weights/best.pt"
    print("=" * 60)
    print(f"완료. 가중치 경로: {best_pt}")
    print("다음: 이 파일을 backend/best.pt 로 복사 후")
    print("      analyzer.py / vectorizer.py / GRU 재학습으로 통합")
    print("=" * 60)


if __name__ == "__main__":
    main()
