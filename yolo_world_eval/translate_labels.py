"""25개 원본 Roboflow 클래스명을 YOLO-World 프롬프트에 적합한 자연어 구문으로 변환.

backend/class_translator.py(한글→영어 번역)와는 목적이 다르다 — 여기 클래스명은 이미
영어지만 "Test_Tube", "Round_Bottom_Flask_Borosilicate_Glass_1_Neck" 같은 ML 데이터셋
라벨링 관례(언더스코어, 복합어)로 되어 있다. CLIP 계열 텍스트 인코더는 이런 라벨
스타일보다 "test tube" 같은 자연스러운 명사구를 훨씬 잘 이해할 가능성이 높다는 가설을
검증하기 위해, 실제 프로덕션(class_translator.py)과 같은 방식 — Groq LLM 일괄 호출 —
으로 변환한다.

class_translator.py에서 llama-3.1-8b-instant가 실제 오역/오타(핀셋→pinset,
리트머스→littmus)를 낸 전례가 있어 같은 이유로 gpt-oss-20b를 사용한다.

사전 준비: download_data.py를 먼저 실행해 ./data/data.yaml이 있어야 한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from openai import OpenAI

DATA_YAML = Path(__file__).parent / "data" / "data.yaml"
OUT_PATH = Path(__file__).parent / "natural_names.json"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "openai/gpt-oss-20b"  # backend/class_translator.py와 동일 모델 선택 이유 참고

SYSTEM_PROMPT = (
    "You rewrite awkward machine-learning dataset labels for chemistry lab equipment "
    "into short, natural English noun phrases suitable as text prompts for an "
    "open-vocabulary object detector (YOLO-World / CLIP). Remove underscores and "
    "dataset-specific jargon, keep it concise (2-6 words), lowercase, no articles. "
    'Given a JSON array of raw labels, reply with a JSON object {"natural": [...]} '
    "containing one rewritten phrase per input, in the exact same order and count."
)


def main() -> None:
    if not DATA_YAML.exists():
        raise SystemExit(f"data.yaml이 없습니다: {DATA_YAML}\n먼저 download_data.py 실행")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY 환경변수가 필요합니다.\n  PowerShell: $env:GROQ_API_KEY = '<key>'"
        )

    with open(DATA_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names_field = cfg["names"]
    raw_names = names_field if isinstance(names_field, list) else list(names_field.values())

    print(f"[translate] 원본 클래스 {len(raw_names)}개를 자연어로 변환 중 (모델: {MODEL_NAME})...")

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(raw_names, ensure_ascii=False)},
        ],
    )
    data = json.loads(response.choices[0].message.content or "{}")
    natural = data.get("natural")
    if not isinstance(natural, list) or len(natural) != len(raw_names):
        raise SystemExit(
            f"번역 결과가 올바르지 않습니다 (입력 {len(raw_names)}개, "
            f"출력 {len(natural) if isinstance(natural, list) else 'N/A'}개). 다시 시도해주세요."
        )

    mapping = dict(zip(raw_names, natural))
    OUT_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n원본 → 자연어 변환 결과:")
    for k, v in mapping.items():
        print(f"  {k:55s} -> {v}")
    print(f"\n[translate] 저장 완료: {OUT_PATH}")
    print("[translate] 이제 evaluate.py --natural 로 이 프롬프트로 평가할 수 있습니다.")


if __name__ == "__main__":
    main()
