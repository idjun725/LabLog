"""새 실험 영상을 분석하고 구간 단위로 phase 라벨을 붙여 training_data.json에 추가.

사용:
  .\\.venv\\Scripts\\python.exe label_experiment.py <video_path> [--title "제목"]
                                                                [--sample-fps 1.0]
                                                                [--dry-run]

Flow:
  1) analyze_video()로 영상에서 records 추출 (배치 STT 포함, 몇 분 소요)
  2) records를 표로 출력
  3) "구간: phase" 형식으로 대화식 입력 (예: "1-5: 1")
  4) 모든 record가 라벨링됐는지 검증
  5) training_data.json 백업 후 새 실험 항목을 append

phase 코드:
  1 = 준비
  2 = 반응
  3 = 측정 및 관찰
  4 = 정리

주의:
  - 재분석이라 batch STT(Google), YOLO, OCR을 다시 돌림 → 영상 길이에 비례한 시간.
  - GOOGLE_APPLICATION_CREDENTIALS 없으면 speech 필드는 빈 문자열.
  - training_data.json은 저장 직전 자동 백업(training_data.backup_YYYYMMDD_HHMMSS.json).
  - Ctrl+C로 언제든 중단 가능 — 백업 이후 실패 시 백업본으로 복구.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from analyzer import LabLogAnalyzer
from gru_model import PHASES

TRAINING_DATA_PATH = Path(__file__).parent / "training_data.json"

# CLI 프롬프트에서 쓰는 코드 → PHASES 인덱스 매핑
PHASE_MENU = {i + 1: p for i, p in enumerate(PHASES)}
# {1: '준비', 2: '반응', 3: '측정 및 관찰', 4: '정리'}


def _to_training_record(result_dict: dict) -> dict:
    """AnalysisResult(dict) → training_data.json 스키마에 맞춘 최소 필드.

    학습에 실제로 쓰이는 필드만 저장해 파일 크기를 낮춘다. vectorizer.py는
    yolo/ocr/ocr_confidence/speech만 소비한다. timestamp는 사람이 검토할 때만 참고.
    """
    return {
        "timestamp": result_dict.get("timestamp", ""),
        "yolo": dict(result_dict.get("yolo", {})),
        "ocr": result_dict.get("ocr", "") or "",
        "ocr_confidence": float(result_dict.get("ocr_confidence", 0.0) or 0.0),
        "speech": result_dict.get("speech", "") or "",
    }


def _print_records_table(records: list[dict]) -> None:
    print()
    print("=" * 100)
    print(f"{'#':>3}  {'시간':<10}  {'YOLO (라벨)':<40}  {'OCR':<20}  발화")
    print("-" * 100)
    for i, r in enumerate(records, 1):
        yolo_str = ", ".join(list(r.get("yolo", {}).keys())[:4]) or "-"
        yolo_str = yolo_str[:38]
        ocr_str = (r.get("ocr", "") or "-")[:18]
        speech_str = (r.get("speech", "") or "-")[:40]
        print(f"{i:>3}  {r.get('timestamp', ''):<10}  {yolo_str:<40}  {ocr_str:<20}  {speech_str}")
    print("=" * 100)
    print(f"총 {len(records)}개 record\n")


def _print_phase_menu() -> None:
    print("Phase 코드:")
    for code, name in PHASE_MENU.items():
        print(f"  {code} = {name}")
    print()
    print("입력 형식: \"시작-끝: 코드\" 또는 \"단일번호: 코드\"")
    print("예: \"1-5: 1\"  →  1~5번 record 를 '준비'로")
    print("빈 줄 입력 = 완료.\n")


def _parse_label_line(line: str, n_records: int) -> tuple[int, int, int]:
    """'1-5: 2' 또는 '3: 4' 형식 파싱 → (start, end, phase_code). 1-indexed inclusive."""
    if ":" not in line:
        raise ValueError("':' 구분자가 필요합니다. 예: '1-5: 2'")
    left, right = line.split(":", 1)
    left = left.strip()
    right = right.strip()

    if "-" in left:
        s_str, e_str = left.split("-", 1)
        start = int(s_str.strip())
        end = int(e_str.strip())
    else:
        start = end = int(left)

    phase_code = int(right)

    if phase_code not in PHASE_MENU:
        raise ValueError(f"phase 코드는 {list(PHASE_MENU.keys())} 중 하나여야 합니다.")
    if start < 1 or end > n_records or start > end:
        raise ValueError(f"범위가 1~{n_records} 안에 있어야 합니다.")
    return start, end, phase_code


def _collect_labels_interactively(n_records: int) -> list[str]:
    """대화식으로 구간별 phase 입력받아 각 record의 라벨 리스트 구성.

    반환: 길이 n_records 의 phase 이름 리스트.
    라벨 미지정 record가 있으면 재입력을 요구.
    """
    labels: list[str | None] = [None] * n_records

    while True:
        for line in iter(lambda: input("> ").strip(), ""):
            try:
                start, end, code = _parse_label_line(line, n_records)
            except ValueError as e:
                print(f"  [입력 오류] {e}")
                continue
            phase_name = PHASE_MENU[code]
            overwritten = [i for i in range(start - 1, end) if labels[i] is not None and labels[i] != phase_name]
            for i in range(start - 1, end):
                labels[i] = phase_name
            print(f"  → {start}~{end}번 record = '{phase_name}' 배정"
                  + (f" ({len(overwritten)}개 덮어씀)" if overwritten else ""))

        missing = [i + 1 for i, l in enumerate(labels) if l is None]
        if not missing:
            return [l for l in labels if l is not None]  # type: ignore[misc]
        print(f"\n[미라벨링] record 번호: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        print("이 record들에도 phase를 배정해 주세요.\n")


def _print_summary(records: list[dict], labels: list[str], title: str) -> None:
    from collections import Counter
    counts = Counter(labels)
    print("\n" + "=" * 60)
    print("저장 요약")
    print("=" * 60)
    print(f"실험 제목: {title}")
    print(f"총 record: {len(records)}")
    print("phase 분포:")
    for phase in PHASES:
        c = counts.get(phase, 0)
        pct = c / len(labels) * 100 if labels else 0
        print(f"  {phase:<15} {c:>3}개 ({pct:>5.1f}%)")
    print()


def _backup_and_append(new_item: dict) -> Path:
    """기존 training_data.json 백업 후 새 항목 append. 백업 경로 반환."""
    if TRAINING_DATA_PATH.exists():
        with TRAINING_DATA_PATH.open("r", encoding="utf-8") as f:
            items = json.load(f)
    else:
        items = []

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = TRAINING_DATA_PATH.with_name(f"training_data.backup_{stamp}.json")
    if TRAINING_DATA_PATH.exists():
        backup_path.write_text(TRAINING_DATA_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    items.append(new_item)
    with TRAINING_DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    return backup_path


def main() -> int:
    parser = argparse.ArgumentParser(description="영상 → records → 구간 라벨링 → training_data.json append.")
    parser.add_argument("video_path", type=Path, help="실험 영상 파일 경로 (mp4, mov, webm 등).")
    parser.add_argument("--title", type=str, default=None,
                        help="실험 제목 (미지정 시 파일명에서 자동 추출).")
    parser.add_argument("--sample-fps", type=float, default=1.0,
                        help="초당 프레임 샘플링 (기본 1.0 = production과 동일).")
    parser.add_argument("--dry-run", action="store_true",
                        help="분석·라벨링만 수행하고 파일에 저장하지 않음.")
    args = parser.parse_args()

    if not args.video_path.exists():
        print(f"[label] 영상 파일 없음: {args.video_path}", file=sys.stderr)
        return 1

    title = args.title or args.video_path.stem
    print(f"[label] 영상: {args.video_path}")
    print(f"[label] 실험 제목: {title}")
    print(f"[label] sample_fps: {args.sample_fps}")

    print("[label] 모델 로드 및 분석 시작 (첫 실행은 SBERT/EasyOCR 다운로드로 오래 걸릴 수 있음)…")
    t0 = time.monotonic()
    analyzer = LabLogAnalyzer()
    results = analyzer.analyze_video(str(args.video_path), sample_fps=args.sample_fps)
    print(f"[label] 분석 완료 — {len(results)}개 record, {time.monotonic() - t0:.1f}초")

    if not results:
        print("[label] record가 0개입니다. 영상이 너무 짧거나 프레임 추출에 실패했을 수 있습니다.", file=sys.stderr)
        return 2

    records = [_to_training_record(asdict(r)) for r in results]
    _print_records_table(records)
    _print_phase_menu()

    print("각 record 번호를 훑어보고 구간별로 phase를 배정하세요.\n")
    labels = _collect_labels_interactively(len(records))

    _print_summary(records, labels, title)

    if args.dry_run:
        print("[label] --dry-run 지정 — training_data.json에 저장하지 않고 종료.")
        return 0

    answer = input("이대로 training_data.json에 append 하시겠습니까? (y/N): ").strip().lower()
    if answer != "y":
        print("[label] 저장 취소.")
        return 0

    new_item = {"experiment": title, "records": records, "labels": labels}
    backup = _backup_and_append(new_item)
    print(f"[label] 저장 완료 → {TRAINING_DATA_PATH.name}")
    print(f"[label] 백업본 → {backup.name}")
    print("[label] 재학습:  .\\.venv\\Scripts\\python.exe train_gru.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[label] 사용자 중단.", file=sys.stderr)
        sys.exit(130)
