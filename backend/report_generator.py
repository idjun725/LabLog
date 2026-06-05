"""Groq API로 실험 records → 구조화된 보고서 생성.

Groq은 OpenAI 호환 API를 제공한다 (xAI Grok과 다른 회사 — 빠른 추론 인프라).
`openai` SDK를 사용하되 `base_url`만 Groq으로 지정. 무료 등급 사용 가능.
환경변수 `GROQ_API_KEY`가 필요하다. 미설정 / SDK 미설치 시 RuntimeError 발생
(자동 폴백 없음 — 보고서는 진짜 생성이거나 명시적 실패).

기획서 4단계의 LLM 보고서 생성 단계를 담당. 입력은 1~3단계의 결과인 records JSON.
"""

from __future__ import annotations

import os
import time

try:
    from openai import OpenAI
    from pydantic import BaseModel, Field
    _LIB_OK = True
except ImportError:
    OpenAI = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = lambda **_kwargs: None  # type: ignore
    _LIB_OK = False


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Groq에서 strict json_schema(structured outputs)를 지원하는 모델만 사용 가능.
# 2026-05 기준 지원: "openai/gpt-oss-20b", "openai/gpt-oss-120b".
# 20b는 TPM 한도가 120b보다 큼 (무료 등급) — 긴 영상의 많은 records를 보낼 때 유리.
# 품질 우선이고 records가 짧으면 "openai/gpt-oss-120b"로 변경.
# llama-3.3-70b 등 다른 모델은 BadRequest 400 — parse() 호출 시.
# (https://console.groq.com/docs/structured-outputs#supported-models)
MODEL_NAME = "openai/gpt-oss-20b"


class GeneratedReport(BaseModel):
    """기획서 4단계 보고서 출력 형식 — 한국 고등학교 실험 보고서 구조."""
    title: str = Field(description="실험 제목")
    date: str = Field(description="실험 날짜 (YYYY-MM-DD 형식, 영상에서 추론 불가능하면 빈 문자열)")
    preliminary_research: str = Field(description="선행 연구 검토 (2~3문장)")
    objective: str = Field(description="실험 목적 (1~2문장)")
    hypothesis: str = Field(description="가설 (1~2문장)")
    materials: list[str] = Field(description="준비물 목록 (분석에서 등장한 기자재/시약 위주)")
    method: str = Field(description="실험 방법 요약 (2~3문장)")
    procedure: list[str] = Field(description="시간순 실험 과정 단계 (각 단계 1~2문장)")
    results: str = Field(description="관찰된 실험 결과 (OCR 측정값과 음성 내용 기반)")
    conclusion: str = Field(description="결론 (실험 결과에서 도출 가능한 범위)")


SYSTEM_PROMPT = """당신은 한국 고등학교 과학 실험 보고서 작성을 돕는 AI입니다.
학생이 자신의 실험 영상을 AI로 분석한 결과(객체 인식 YOLO + 화면 텍스트 OCR + 음성 STT)를
시간순으로 받습니다. 각 시점이 어느 실험 단계(준비/반응/측정 및 관찰/정리)에 속하는지
정보가 함께 제공될 수 있습니다. 이를 바탕으로 한국 고등학교 형식의 실험 보고서를 작성하세요.

원칙:
- 분석 결과에 보이는 정보만 사용해 작성합니다.
- 가설·결론처럼 추론이 필요한 부분은 데이터에서 합리적으로 도출 가능한 범위에서만 작성하세요.
- 데이터에 근거 없는 정보(존재하지 않는 시약, 측정 안 한 값 등)는 만들어내지 마세요.
- 단계 정보가 주어지면 procedure 항목을 단계별로 묶어 정리하세요.
- 정보가 부족한 항목은 "데이터 부족"이라고 명시하세요.
- 보고서는 한국어로 작성합니다."""


def _format_records(records: list[dict], phases: list[dict] | None = None) -> str:
    """records를 LLM이 읽기 쉬운 텍스트로 변환.

    phases가 있으면 같은 단계가 연속되는 구간을 하나의 블록으로 묶고
    객체·OCR·음성의 중복을 제거해 토큰 수를 줄인다 (단계 구조와 고유 정보는 보존).
    phases가 없으면 record당 한 줄로 그대로 나열한다.
    """
    if not phases:
        lines: list[str] = []
        for r in records:
            ts = r.get("timestamp", "")
            yolo = r.get("yolo") or {}
            ocr = r.get("ocr", "")
            speech = r.get("speech", "")
            parts = [f"[{ts}]"]
            if yolo:
                parts.append(f"객체: {', '.join(yolo.keys())}")
            if ocr:
                parts.append(f"OCR: {ocr}")
            if speech:
                parts.append(f"음성: {speech}")
            lines.append("  ".join(parts))
        return "\n".join(lines)

    phase_labels = [(p or {}).get("phase", "") for p in phases]
    blocks: list[str] = []
    i = 0
    while i < len(records):
        cur_phase = phase_labels[i] if i < len(phase_labels) else ""
        j = i + 1
        while j < len(records) and (
            phase_labels[j] if j < len(phase_labels) else ""
        ) == cur_phase:
            j += 1

        chunk = records[i:j]
        start_ts = chunk[0].get("timestamp", "")
        end_ts = chunk[-1].get("timestamp", "")
        time_range = start_ts if start_ts == end_ts else f"{start_ts}~{end_ts}"

        # 순서를 유지하면서 중복 제거 (dict의 키 순서 보존 특성 활용)
        objects: dict[str, None] = {}
        ocr_texts: dict[str, None] = {}
        speech_texts: dict[str, None] = {}
        for r in chunk:
            for k in (r.get("yolo") or {}).keys():
                objects[k] = None
            if r.get("ocr"):
                ocr_texts[r["ocr"]] = None
            if r.get("speech"):
                speech_texts[r["speech"]] = None

        header = f"[{cur_phase or '미분류'} {time_range}]"
        parts = [header]
        if objects:
            parts.append(f"  객체: {', '.join(objects)}")
        if ocr_texts:
            parts.append(f"  OCR: {' / '.join(ocr_texts)}")
        if speech_texts:
            parts.append(f"  음성: {' / '.join(speech_texts)}")
        blocks.append("\n".join(parts))
        i = j

    return "\n\n".join(blocks)


_INFO_LABEL_BY_KEY = {
    "title": "실험 제목",
    "subject": "실험 주제",
    "date": "실험 날짜",
    "hypothesis": "가설",
    "other": "기타 메모",
}


def _format_info(info: dict | None) -> str:
    """사용자가 미리 입력한 기본 정보를 LLM 입력용 텍스트로 변환.

    빈 값/None은 건너뛴다. 한 항목도 없으면 빈 문자열 반환.
    """
    if not info:
        return ""
    lines: list[str] = []
    for key, label in _INFO_LABEL_BY_KEY.items():
        value = (info.get(key) or "").strip()
        if value:
            lines.append(f"- {label}: {value}")
    if not lines:
        return ""
    return "사용자가 직접 입력한 실험 기본 정보 (분석 영상 외 추가 컨텍스트):\n" + "\n".join(lines)


def generate_report(
    records: list[dict],
    filename: str | None = None,
    phases: list[dict] | None = None,
    info: dict | None = None,
) -> GeneratedReport:
    """records를 받아 Grok으로 실험 보고서 생성.

    실패 시 RuntimeError 발생 (자동 폴백 없음 — 호출 측이 사용자에게 명시적으로 알린다).
    phases가 주어지면 각 record의 단계 정보를 LLM 입력에 포함해 보고서 정확도를 높인다.
    info(사용자가 업로드/촬영 전 입력한 제목·주제·날짜·가설·기타)가 있으면 prompt 상단에 주입해
    LLM이 실험의 의도/맥락을 더 정확히 반영하도록 한다.
    """
    if not _LIB_OK:
        raise RuntimeError(
            "openai 또는 pydantic 패키지가 설치되지 않았습니다. "
            "pip install -r requirements.txt"
        )
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY 환경변수가 설정되지 않았습니다.")

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_BASE_URL,
    )

    info_text = _format_info(info)
    records_text = _format_records(records, phases)
    user_message_parts = [
        f"파일명: {filename or '(unknown)'}",
        f"분석 프레임 수: {len(records)}" + (", 단계 분류 포함" if phases else ""),
    ]
    if info_text:
        user_message_parts.append("\n" + info_text)
    user_message_parts.append("\n시간순 분석 결과:\n" + records_text)
    user_message_parts.append(
        "\n위 정보를 바탕으로 한국 고등학교 실험 보고서를 작성해주세요. "
        "사용자가 입력한 기본 정보(제목·주제·가설 등)가 있으면 그 의도를 존중하되, "
        "분석 결과와 모순되면 분석 결과를 우선합니다."
    )
    user_message = "\n".join(user_message_parts)

    print(
        f"[report] Groq 호출 시작 (records {len(records)}개, "
        f"phases={'있음' if phases else '없음'}, "
        f"info={'있음' if info_text else '없음'}, model={MODEL_NAME})",
        flush=True,
    )
    t0 = time.monotonic()
    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=GeneratedReport,
    )
    elapsed = time.monotonic() - t0
    usage = response.usage
    print(
        f"[report] Groq 응답 ({elapsed:.1f}초, "
        f"prompt {usage.prompt_tokens}, completion {usage.completion_tokens})",
        flush=True,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(
            f"Groq이 유효한 구조화 응답을 반환하지 않았습니다. "
            f"refusal={response.choices[0].message.refusal}"
        )
    return parsed
