"""best.pt(파인튜닝) vs 순수 YOLO-World 예측을 같은 이미지에 나란히 그려서 눈으로 비교.

evaluate.py가 전체 통계(mAP 등)를 내는 스크립트라면, 이건 실제로 어떤 사진에서
YOLO-World가 놓치거나 헷갈리는지 직접 눈으로 확인하기 위한 보조 도구.

사전 준비: download_data.py를 먼저 실행해 ./data/data.yaml이 있어야 한다.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

DATA_DIR = Path(__file__).parent / "data"
DATA_YAML = DATA_DIR / "data.yaml"
BEST_PT = Path(__file__).parent.parent / "backend" / "best.pt"
YOLO_WORLD_MODEL = "yolov8s-worldv2.pt"  # backend/open_vocab_detector.py와 동일
RESULTS_DIR = Path(__file__).parent / "results" / "qualitative"
N_SAMPLES = 12
# analyzer.py DEFAULT_CONF_THRESHOLD / open_vocab_detector.DEFAULT_CONF_THRESHOLD와 동일 값
BEST_PT_CONF = 0.35
YOLO_WORLD_CONF = 0.15


def find_val_images() -> list[Path]:
    with open(DATA_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    val_rel = cfg.get("val") or cfg.get("valid") or "valid/images"
    val_dir = (DATA_DIR / val_rel).resolve()
    if not val_dir.exists():
        val_dir = DATA_DIR / "valid" / "images"  # Roboflow yolov8 내보내기 기본 경로
    return sorted(val_dir.glob("*.jpg")) + sorted(val_dir.glob("*.png"))


def main() -> None:
    if not DATA_YAML.exists():
        raise SystemExit(f"data.yaml이 없습니다: {DATA_YAML}\n먼저 download_data.py 실행")
    if not BEST_PT.exists():
        raise SystemExit(f"best.pt를 찾을 수 없습니다: {BEST_PT}")

    with open(DATA_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names_field = cfg["names"]
    names = names_field if isinstance(names_field, list) else list(names_field.values())

    images = find_val_images()
    if not images:
        raise SystemExit("검증 이미지를 찾을 수 없습니다.")
    sample = random.sample(images, min(N_SAMPLES, len(images)))

    print(f"[compare] best.pt 로드: {BEST_PT}")
    best_model = YOLO(str(BEST_PT))
    print(f"[compare] {YOLO_WORLD_MODEL} 로드 (순수, 파인튜닝 없음)")
    world_model = YOLO(YOLO_WORLD_MODEL)
    world_model.set_classes(names)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for img_path in sample:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        r_best = best_model(frame, verbose=False, conf=BEST_PT_CONF)[0]
        r_world = world_model(frame, verbose=False, conf=YOLO_WORLD_CONF)[0]

        vis_best = r_best.plot()
        vis_world = r_world.plot()
        h = min(vis_best.shape[0], vis_world.shape[0])
        vis_best = cv2.resize(vis_best, (int(vis_best.shape[1] * h / vis_best.shape[0]), h))
        vis_world = cv2.resize(vis_world, (int(vis_world.shape[1] * h / vis_world.shape[0]), h))
        divider = cv2.copyMakeBorder(
            vis_best, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(0, 0, 255)
        )
        combined = cv2.hconcat([divider, vis_world])

        out_path = RESULTS_DIR / f"{img_path.stem}_compare.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"[compare] {out_path.name} 저장")

    print(
        f"\n[compare] 완료 — {RESULTS_DIR}에서 확인 "
        f"(왼쪽=best.pt conf>={BEST_PT_CONF}, 빨간 선 오른쪽=YOLO-World conf>={YOLO_WORLD_CONF})"
    )


if __name__ == "__main__":
    main()
