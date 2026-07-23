"""한국어 커스텀 클래스명 → 영어 번역 (YOLO-World 프롬프트 입력 전처리).

YOLO-World의 텍스트 인코더(OpenAI CLIP)는 영어 코퍼스로만 학습돼 한국어 프롬프트를
사실상 이해하지 못한다 (실측: conf_threshold=0.05까지 낮춰도 탐지 0개, 동일 이미지에서
영어 프롬프트는 정상 탐지). 프론트엔드에서 사용자가 직접 추가하는 커스텀 클래스명(한글일
수 있음)은 이 모듈을 거쳐 영어로 바꾼 뒤에만 open_vocab_detector에 넘겨야 한다.

best.pt의 기존 25개 클래스가 전부 영어 이름이므로(vectorizer.YOLO_VOCAB 참고), 번역
결과도 영어로 맞추는 게 records의 `yolo` 필드 표기 관례와 일관적이다.

GROQ_API_KEY 미설정/SDK 미설치/API 실패 시 원문(한글) 그대로 반환한다 — report_generator.py와
달리 이 기능은 선택적 보조 탐지라서 하드 실패시키지 않는다. 다만 원문이 그대로 YOLO-World에
들어가면 위 이유로 탐지가 비어있을 수 있다는 걸 로그로 남긴다.
"""

from __future__ import annotations

import json
import os
import re

try:
    from openai import OpenAI
    _LIB_OK = True
except ImportError:
    OpenAI = None  # type: ignore
    _LIB_OK = False

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# 실측: llama-3.1-8b-instant는 "핀셋"→"pinset"(오역), "리트머스 종이"→"littmus_paper"(오타)
# 같은 실질적 오류를 냄 — YOLO-World 인식률에 직접 영향을 주므로 report_generator.py와
# 같은 gpt-oss-20b로 교체 (같은 모델 재사용이지만 번역 요청은 프롬프트가 짧아 TPM 부담 적음).
MODEL_NAME = "openai/gpt-oss-20b"

_HANGUL_PATTERN = re.compile(r"[가-힣]")
_translation_cache: dict[str, str] = {}  # 프로세스 생애주기 동안 동일 클래스명 재번역 방지


def _needs_translation(text: str) -> bool:
    return bool(_HANGUL_PATTERN.search(text))


def translate_classes(classes: list[str]) -> list[str]:
    """한글이 섞인 클래스명만 골라 Groq로 한 번에 번역, 영어는 그대로 통과.

    입력과 같은 순서·길이를 보장 — 번역 실패 시 해당 항목은 원문(한글) 유지.
    """
    to_translate = [
        c for c in classes if _needs_translation(c) and c not in _translation_cache
    ]
    if to_translate:
        translated = _call_groq_translate(to_translate)
        if translated is not None and len(translated) == len(to_translate):
            for original, english in zip(to_translate, translated):
                _translation_cache[original] = english
        else:
            print(
                f"[class_translator] 번역 실패/개수 불일치 — 원문 유지: {to_translate}",
                flush=True,
            )

    return [
        _translation_cache.get(c, c) if _needs_translation(c) else c for c in classes
    ]


def _call_groq_translate(korean_terms: list[str]) -> list[str] | None:
    if not _LIB_OK:
        print("[class_translator] openai SDK 미설치 — 번역 건너뜀", flush=True)
        return None
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(
            "[class_translator] GROQ_API_KEY 미설정 — 번역 건너뜀 (원문 그대로 사용, "
            "YOLO-World가 한글 프롬프트를 인식하지 못해 탐지가 비어있을 수 있음)",
            flush=True,
        )
        return None

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You translate Korean science-lab object/equipment names into "
                        "short English YOLO-style class names (1-3 words, lowercase, no "
                        "articles, no explanations). Given a JSON array of Korean terms, "
                        'reply with a JSON object {"translations": [...]} containing the '
                        "English translations, in the exact same order and count as the input."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(korean_terms, ensure_ascii=False),
                },
            ],
        )
    except Exception as e:
        print(f"[class_translator] Groq 호출 실패: {e}", flush=True)
        return None

    try:
        data = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError as e:
        print(f"[class_translator] 응답 JSON 파싱 실패: {e}", flush=True)
        return None

    translations = data.get("translations")
    if isinstance(translations, list) and all(isinstance(t, str) for t in translations):
        return translations
    return None
