"""ultralytics + Roboflow 학습 스크립트 공통 helper.

train_seven_segment.py / train_main_yolo.py가 공유. 새 학습 스크립트도 같은 패턴.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    print(
        "[train] roboflow 패키지가 필요합니다: pip install roboflow",
        file=sys.stderr,
    )
    sys.exit(1)

from ultralytics import YOLO


def backup_with_timestamp(path: Path) -> None:
    """파일이 있으면 같은 디렉터리에 'name.backup_YYYYMMDD_HHMMSS.ext'로 복사."""
    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}.backup_{timestamp}{path.suffix}")
    shutil.copy(path, backup_path)
    print(f"[train] 기존 가중치 백업: {backup_path.name}")


def download_roboflow_dataset(
    workspace: str, project: str, version: int, location: Path
) -> Path:
    """yolov8 포맷으로 다운로드. 같은 버전 재다운로드 방지용 캐시."""
    if location.exists() and (location / "data.yaml").exists():
        print(f"[train] 데이터셋 이미 존재 — 재사용: {location}")
        return location

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print(
            "[train] ROBOFLOW_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "  PowerShell: $env:ROBOFLOW_API_KEY = '<your_key>'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[train] Roboflow 다운로드: {workspace}/{project} v{version}")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)
    dataset = ver.download("yolov8", location=str(location))
    result = Path(dataset.location)
    print(f"[train] 다운로드 완료: {result}")
    return result


def train_and_export(
    *,
    base_model: str,
    data_yaml: Path,
    runs_dir: Path,
    run_name: str,
    weights_out: Path,
    epochs: int,
    imgsz: int,
    device: str = "cpu",
    patience: int | None = None,
) -> None:
    """ultralytics 학습 → runs/<run_name>/weights/best.pt → weights_out으로 복사.

    weights_out에 기존 가중치가 있으면 타임스탬프 백업 후 교체.
    """
    if not data_yaml.exists():
        print(f"[train] data.yaml이 없습니다: {data_yaml}", file=sys.stderr)
        sys.exit(1)

    backup_with_timestamp(weights_out)

    patience_str = f", patience={patience}" if patience else ""
    print(
        f"[train] 학습 시작 — base={base_model}, epochs={epochs}, "
        f"imgsz={imgsz}{patience_str} (device={device})"
    )
    t0 = time.monotonic()

    train_kwargs: dict = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        verbose=True,
        project=str(runs_dir),
        name=run_name,
        exist_ok=True,
    )
    if patience is not None:
        train_kwargs["patience"] = patience

    model = YOLO(base_model)
    model.train(**train_kwargs)

    best_pt = runs_dir / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[train] best.pt를 찾을 수 없습니다: {best_pt}", file=sys.stderr)
        sys.exit(1)

    shutil.copy(best_pt, weights_out)
    elapsed = time.monotonic() - t0
    size_kb = weights_out.stat().st_size // 1024
    print(
        f"[train] 가중치 교체 완료: {weights_out.name} "
        f"({size_kb}KB, 총 {elapsed/60:.1f}분)"
    )
