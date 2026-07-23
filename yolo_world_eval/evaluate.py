"""순수 YOLO-World(파인튜닝 없음)의 실험기구 탐지 정확도 측정.

best.pt가 학습한 것과 같은 Roboflow 데이터셋(ground-truth 라벨 포함)에 대해,
YOLO-World에 25개 클래스명을 텍스트 프롬프트로 주고 zero-shot으로 평가한다.
ultralytics의 표준 .val() 파이프라인을 그대로 써서 mAP50/mAP50-95/precision/recall을
공식 방식으로 계산 — 직접 IOU 매칭 코드를 짜는 것보다 신뢰도 높음.

옵션:
  --natural       translate_labels.py가 생성한 자연어 프롬프트 사용 (기본: 원본 Roboflow 라벨)
  --model NAME    YOLO-World 모델 크기 (기본: yolov8s-worldv2.pt. m/l/x-worldv2.pt도 가능)
  --imgsz N       검증 입력 해상도 (기본: 640)

주의: ultralytics의 WorldValidator(models/yolo/world/val.py)는 model.set_classes()로
미리 설정한 프롬프트를 무시하고 .val() 호출 시 data.yaml의 names를 다시 읽어 자체
재설정한다. 그래서 --natural 모드에서는 원본 data.yaml을 그대로 쓰지 않고, names만
자연어로 치환한 임시 yaml(data/data_natural.yaml)을 만들어 그걸 .val(data=...)에
넘긴다 — ground truth 라벨 파일은 클래스를 인덱스로 저장하므로 names 텍스트만
바꾸는 건 안전하다 (순서를 바꾸면 안 됨).

사전 준비: download_data.py를 먼저 실행해 ./data/data.yaml이 있어야 한다.
자연어 비교를 하려면 translate_labels.py도 먼저 실행해야 한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

DATA_DIR = Path(__file__).parent / "data"
DATA_YAML = DATA_DIR / "data.yaml"
NATURAL_NAMES_PATH = Path(__file__).parent / "natural_names.json"
RESULTS_DIR = Path(__file__).parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="순수 YOLO-World 정확도 측정")
    parser.add_argument(
        "--natural", action="store_true",
        help="자연어 프롬프트 사용 (기본은 Roboflow 원본 라벨)",
    )
    parser.add_argument(
        "--model", default="yolov8s-worldv2.pt",
        help="YOLO-World 가중치 — yolov8{s,m,l,x}-worldv2.pt 중 하나. "
        "backend/open_vocab_detector.py 기본값은 's'.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="검증 입력 해상도 (기본 640). 높일수록 작은 물체 탐지에 유리하나 느려짐.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(
            f"data.yaml이 없습니다: {DATA_YAML}\n먼저 download_data.py를 실행하세요."
        )

    with open(DATA_YAML, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    names_field = data_cfg["names"]
    class_names = names_field if isinstance(names_field, list) else list(names_field.values())

    # 결과 폴더명에 모델 크기·해상도·프롬프트 방식을 모두 담아 실행마다 겹치지 않게.
    model_tag = Path(args.model).stem.replace("-worldv2", "")  # "yolov8s-worldv2.pt" -> "yolov8s"
    prompt_tag = "natural" if args.natural else "raw"
    run_name = f"{prompt_tag}_{model_tag}_imgsz{args.imgsz}"

    prompt_names = class_names
    val_data_yaml = DATA_YAML
    if args.natural:
        if not NATURAL_NAMES_PATH.exists():
            raise SystemExit(
                f"{NATURAL_NAMES_PATH}가 없습니다.\n먼저 translate_labels.py를 실행하세요."
            )
        mapping = json.loads(NATURAL_NAMES_PATH.read_text(encoding="utf-8"))
        # data.yaml의 클래스 순서(=ground truth 인덱스)를 그대로 유지한 채 텍스트만 치환
        prompt_names = [mapping.get(n, n) for n in class_names]

        # WorldValidator가 .val() 호출 시 data.yaml의 names를 다시 읽어 자체
        # set_classes()를 호출하므로(위 모듈 docstring 참고), 미리 해둔
        # model.set_classes(prompt_names)가 무시된다. names 자체를 자연어로
        # 바꾼 임시 yaml을 만들어 넘겨야 실제로 적용된다.
        natural_cfg = dict(data_cfg)
        natural_cfg["names"] = prompt_names
        val_data_yaml = DATA_DIR / "data_natural.yaml"
        with open(val_data_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(natural_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[eval] 평가 대상 클래스 {len(class_names)}개")
    print(f"[eval] 프롬프트: {prompt_names}")
    print(f"[eval] 모델: {args.model} (파인튜닝 없음, 순수 zero-shot), imgsz={args.imgsz}")

    model = YOLO(args.model)
    model.set_classes(prompt_names)

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = model.val(
        data=str(val_data_yaml),
        split="val",  # data.yaml의 키는 항상 'val' — 실제 폴더명이 'valid/'여도 무관
        imgsz=args.imgsz,
        project=str(RESULTS_DIR),
        name=run_name,
        exist_ok=True,
    )

    summary = {
        "model": args.model,
        "finetuned": False,
        "prompt_style": "natural" if args.natural else "raw_roboflow_label",
        "imgsz": args.imgsz,
        "num_classes": len(class_names),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
    }
    out_path = RESULTS_DIR / run_name / "summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"model      : {args.model}  (imgsz={args.imgsz}, prompt={summary['prompt_style']})")
    print(f"mAP50      : {summary['mAP50']:.3f}")
    print(f"mAP50-95   : {summary['mAP50-95']:.3f}")
    print(f"precision  : {summary['precision_mean']:.3f}")
    print(f"recall     : {summary['recall_mean']:.3f}")
    print("=" * 60)
    print(f"[eval] 요약 저장: {out_path}")
    print(
        f"[eval] 클래스별 AP·confusion matrix·PR curve는 "
        f"{RESULTS_DIR / run_name}에 자동 저장됨"
    )


if __name__ == "__main__":
    main()
