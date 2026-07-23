"""best.pt 학습에 쓰인 것과 같은 Roboflow 데이터셋(ground-truth 라벨 포함)을 다운로드.

이 데이터셋으로 순수 YOLO-World(파인튜닝 없음)를 평가해 정확도를 측정한다.
원본 프로젝트는 backend/train_main_yolo.py 참고 (같은 workspace/project/version).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

try:
    from roboflow import Roboflow
except ImportError:
    print(
        "roboflow 패키지가 필요합니다. backend/.venv를 재사용한다면 이미 설치돼 있습니다:\n"
        "  ..\\backend\\.venv\\Scripts\\python.exe download_data.py",
        file=sys.stderr,
    )
    sys.exit(1)

WORKSPACE = "s-workspace-qeozq"
PROJECT = "chemistry-lab-object-detection-topas"
VERSION = 1
DATA_DIR = Path(__file__).parent / "data"


def _fix_data_yaml_paths(data_yaml_path: Path) -> None:
    """Roboflow yolov8 내보내기의 고질적인 버그 수정.

    Roboflow가 쓰는 train/val/test 경로(예: '../valid/images')는 원래 dataset 루트보다
    한 단계 더 깊은 위치에서 실행되는 걸 가정한 상대경로라, ultralytics가 data.yaml
    바로 옆(같은 dataset 루트)에서 resolve하면 한 단계 위로 잘못 올라가 실제 폴더를
    못 찾는다 (FileNotFoundError). explicit 'path'를 dataset 루트 절대경로로 박고
    train/val/test를 '../' 없이 다시 쓰면 어디서 실행하든 안전하게 해결된다.
    """
    with open(data_yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = data_yaml_path.parent.resolve()
    cfg["path"] = str(root)
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        rel = f"{folder}/images"
        if (root / rel).exists():
            cfg[split] = rel

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"[download] data.yaml 경로 보정 완료 (path={root})")


def main() -> None:
    data_yaml_path = DATA_DIR / "data.yaml"
    if data_yaml_path.exists():
        print(f"[download] 이미 존재 — 재사용: {DATA_DIR}")
        _fix_data_yaml_paths(data_yaml_path)
        return

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print(
            "ROBOFLOW_API_KEY 환경변수가 필요합니다.\n"
            "  PowerShell: $env:ROBOFLOW_API_KEY = '<your_key>'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[download] {WORKSPACE}/{PROJECT} v{VERSION} 다운로드 중...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(WORKSPACE).project(PROJECT)
    ver = proj.version(VERSION)
    dataset = ver.download("yolov8", location=str(DATA_DIR))
    print(f"[download] 완료: {dataset.location}")
    _fix_data_yaml_paths(data_yaml_path)


if __name__ == "__main__":
    main()
