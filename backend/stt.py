"""음성 → 텍스트 (Groq Whisper) 모듈.

영상에서 오디오를 추출(ffmpeg)하고 Groq Whisper로 전사한다.
ffmpeg 또는 GROQ_API_KEY가 없으면 자동 비활성화되며, 호출 측은 빈 결과만 받아
시스템 전체는 계속 동작한다 (silent fallback).

- 배치 path: extract_and_transcribe(video_path) → SpeechSegment[]
- 실시간 path: transcribe_bytes(opus_bytes) → str (WebSocket 핸들러용)

배치는 verbose_json으로 시점별 segment를 받고, 실시간은 plain text로 빠르게.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass

try:
    from groq import Groq
    _STT_LIB_OK = True
except ImportError:
    Groq = None  # type: ignore
    _STT_LIB_OK = False


WHISPER_MODEL = "whisper-large-v3-turbo"

# 실험 용어 bias — Whisper가 도메인 단어를 정확히 인식하도록 prompt에 주입.
# (jarvis-assistant의 WHISPER_PROMPT 패턴을 lab 어휘로 교체.)
WHISPER_PROMPT = (
    "비커, 시험관, 전자저울, 메스실린더, 피펫, 뷰렛, 알코올램프, 스포이트, "
    "온도계, 자석교반기, 스톱워치, 시약병, 적정, 중화, 산화, 환원, 전기분해, "
    "ph, 농도, 몰농도, 그램, 밀리리터, 도씨, 측정, 관찰, 반응, 결과"
)

WHISPER_MAX_FILE_BYTES = 25 * 1024 * 1024   # Groq 제한
WHISPER_CHUNK_SECONDS = 600                  # 10분 — 16kHz mono WAV 기준 ~19MB

# 실시간 path의 silent chunk 스킵 임계값. 너무 낮으면 침묵 청크가 Whisper에 들어가
# hallucination("난난난난", "감사합니다 감사합니다") 유발. 잡음 많은 환경이면 더 올릴 것.
REALTIME_RMS_THRESHOLD = 0.015

# Whisper hallucination 후처리 필터.
# 1-3자 토큰이 4회 이상 연속 반복되는 패턴 (예: "난난난난난", "안녕안녕안녕안녕").
_HALLUCINATION_PATTERN = re.compile(r"(.{1,3})\1{3,}")

# 학습 데이터(자막/엔딩) 잔재로 자주 등장하는 문구.
_HALLUCINATION_PHRASES = (
    "끝까지 시청해주셔서",
    "구독과 좋아요",
    "다음 회 예고",
    "감사합니다 감사합니다",
    "subtitled by",
    "ⓒ mbc", "ⓒ kbs", "ⓒ sbs",
)


def _is_hallucination(text: str) -> bool:
    """Whisper의 침묵-구간 hallucination 감지 — True면 호출자가 빈 문자열로 대체."""
    if not text:
        return False
    cleaned = re.sub(r"[\s.,!?…·\-　]+", "", text)
    if len(cleaned) < 4:
        # 너무 짧은 결과는 보수적으로 통과 ("네", "응" 등 유효 응답 보존)
        return False
    if _HALLUCINATION_PATTERN.search(cleaned):
        return True
    if _has_sentence_repetition(text):
        return True
    lower = text.lower()
    return any(phrase in lower for phrase in _HALLUCINATION_PHRASES)


def _has_sentence_repetition(text: str) -> bool:
    """같은 문장이 2회 이상 반복되면 hallucination으로 간주.

    예: '자 이제 10초에 도착합니다. 자 이제 10초에 도착합니다.' → True
    """
    from collections import Counter
    sentences = [s.strip() for s in re.split(r"[.!?。]+", text) if s.strip()]
    if len(sentences) < 2:
        return False
    most_common, count = Counter(sentences).most_common(1)[0]
    return len(most_common) >= 5 and count >= 2


def _is_prompt_echo(text: str, prior_text: str) -> bool:
    """현재 출력이 직전 컨텍스트와 너무 겹치면 prompt echo로 판단 — 빈 문자열로 대체.

    Whisper에 prior_text를 prompt로 넘기면 그 끝부분을 그대로 다시 출력하는 경향이 있음.
    이 함수가 그 패턴을 탐지.
    """
    if not text or not prior_text or len(text) < 10:
        return False
    def _norm(s: str) -> str:
        return re.sub(r"[\s.,!?…·\-]+", " ", s).strip().lower()
    text_n = _norm(text)
    prior_n = _norm(prior_text)
    if not text_n or not prior_n:
        return False
    # 케이스 1: 출력이 직전 컨텍스트의 부분 문자열 (그대로 echo)
    if text_n in prior_n:
        return True
    # 케이스 2: 직전 컨텍스트의 끝 20자 이상이 현재 출력에 들어있음
    prior_tail = prior_n[-30:].strip()
    if len(prior_tail) >= 20 and prior_tail in text_n:
        return True
    return False


@dataclass
class SpeechSegment:
    start: float    # seconds from start of audio
    end: float
    text: str


# ── 가용성/클라이언트 ────────────────────────────────────────────────────

def stt_available() -> bool:
    """STT 실제 사용 가능 여부 (라이브러리 + API key)."""
    if not _STT_LIB_OK:
        return False
    return bool(os.getenv("GROQ_API_KEY"))


def _get_client():
    if not stt_available():
        return None
    try:
        return Groq(api_key=os.environ["GROQ_API_KEY"])
    except Exception as e:
        print(f"[stt] Groq 클라이언트 초기화 실패: {e}", flush=True)
        return None


# ── 오디오 추출 (ffmpeg) ─────────────────────────────────────────────────

def extract_audio(video_path: str) -> str | None:
    """영상에서 16kHz mono PCM WAV를 추출하고 경로 반환. 실패 시 None."""
    out_path = f"{video_path}.wav"
    print("[stt] ffmpeg 오디오 추출 시작", flush=True)
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-ac", "1", "-ar", "16000",
                "-acodec", "pcm_s16le",
                out_path,
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            # 실제 에러는 stderr 마지막 줄들에 있음 (앞 1KB는 ffmpeg 배너).
            lines = result.stderr.decode("utf-8", errors="ignore").strip().splitlines()
            tail = "\n".join(lines[-5:]) if lines else "(stderr 비어있음)"
            print(
                f"[stt] ffmpeg 추출 실패 (returncode={result.returncode}):\n{tail}",
                flush=True,
            )
            return None
        elapsed = time.monotonic() - t0
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"[stt] ffmpeg 추출 완료 ({elapsed:.1f}초, {size_mb:.1f}MB)", flush=True)
        return out_path
    except FileNotFoundError:
        print("[stt] ffmpeg가 설치되지 않음 — 음성 분석을 건너뜁니다.", flush=True)
        return None


# ── 배치 전사 ────────────────────────────────────────────────────────────

def transcribe(wav_path: str, language: str = "ko") -> list[SpeechSegment]:
    """WAV → Groq Whisper → SpeechSegment 리스트. 길면 청크 분할."""
    client = _get_client()
    if client is None:
        if not _STT_LIB_OK:
            print("[stt] groq 미설치 — 건너뜀", flush=True)
        else:
            print("[stt] GROQ_API_KEY 미설정 — 건너뜀", flush=True)
        return []

    duration = _wav_duration(wav_path)
    size = os.path.getsize(wav_path)

    if size <= WHISPER_MAX_FILE_BYTES:
        print(
            f"[stt] 전사 시작 (음성 {duration:.1f}초, "
            f"{size/1024/1024:.1f}MB, 단일 호출)",
            flush=True,
        )
        return _whisper_file(client, wav_path, language, offset=0.0)

    chunk_count = int((duration + WHISPER_CHUNK_SECONDS - 1) // WHISPER_CHUNK_SECONDS)
    print(f"[stt] 전사 시작 (음성 {duration:.1f}초, 청크 {chunk_count}개)", flush=True)

    segments: list[SpeechSegment] = []
    for i in range(chunk_count):
        start = i * WHISPER_CHUNK_SECONDS
        end = min(start + WHISPER_CHUNK_SECONDS, duration)
        chunk_path = _extract_wav_chunk(wav_path, start, end)
        if not chunk_path:
            continue
        print(f"[stt] 청크 {i+1}/{chunk_count} 전사 중 ({start:.0f}s-{end:.0f}s)", flush=True)
        try:
            segments.extend(_whisper_file(client, chunk_path, language, offset=start))
        finally:
            try:
                os.unlink(chunk_path)
            except OSError:
                pass
    return segments


def _whisper_file(client, path: str, language: str, offset: float) -> list[SpeechSegment]:
    """단일 파일을 Whisper에 보내고 SpeechSegment[] 반환. offset은 글로벌 timeline 보정용."""
    t0 = time.monotonic()
    try:
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(path), f.read()),
                model=WHISPER_MODEL,
                language=language,
                prompt=WHISPER_PROMPT,
                response_format="verbose_json",
            )
    except Exception as e:
        print(f"[stt] Whisper 호출 실패: {e}", flush=True)
        return []

    segments: list[SpeechSegment] = []
    raw_segs = getattr(result, "segments", None) or []
    for s in raw_segs:
        if isinstance(s, dict):
            text = (s.get("text") or "").strip()
            start = float(s.get("start") or 0.0)
            end = float(s.get("end") or 0.0)
        else:
            text = (getattr(s, "text", "") or "").strip()
            start = float(getattr(s, "start", 0.0))
            end = float(getattr(s, "end", 0.0))
        if text:
            segments.append(SpeechSegment(start=start + offset, end=end + offset, text=text))

    elapsed = time.monotonic() - t0
    print(f"[stt] Whisper 응답 ({elapsed:.1f}초, 세그먼트 {len(segments)}개)", flush=True)
    return segments


# ── 실시간 전사 (WebSocket용) ────────────────────────────────────────────

# 직전 컨텍스트를 잘라낼 때 쓰는 길이. 너무 길면 prompt feedback loop이 생겨
# Whisper가 컨텍스트의 끝 문장을 반복 출력하는 echo 현상이 일어난다 (관찰 결과).
# 80자면 vocabulary/스타일 힌트는 주되 문장 반복 유도는 약함.
REALTIME_CONTEXT_MAX_CHARS = 80


def transcribe_bytes(
    audio_bytes: bytes,
    suffix: str = ".webm",
    language: str = "ko",
    prior_text: str = "",
) -> str:
    """원본 청크 바이트(MediaRecorder WebM/Opus 등) → 전사 텍스트.

    prior_text: 직전 청크의 인식 결과. Whisper prompt에 함께 주입해 문맥/스타일 연속성 ↑.
    가벼운 호출: verbose_json 대신 json. 빈손이거나 실패 시 빈 문자열.
    """
    client = _get_client()
    if client is None:
        return ""

    # prompt 조합: 정적 어휘 bias + 직전 컨텍스트 (있으면)
    if prior_text:
        trimmed = prior_text[-REALTIME_CONTEXT_MAX_CHARS:].strip()
        prompt = f"{WHISPER_PROMPT} {trimmed}" if trimmed else WHISPER_PROMPT
    else:
        prompt = WHISPER_PROMPT

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    try:
        with open(tmp.name, "rb") as f:
            # temperature=0 — 결정적 디코딩, 무작위 hallucination 감소.
            result = client.audio.transcriptions.create(
                file=(f"chunk{suffix}", f.read()),
                model=WHISPER_MODEL,
                language=language,
                prompt=prompt,
                response_format="json",
                temperature=0.0,
            )
        text = (getattr(result, "text", "") or "").strip()
        if _is_hallucination(text):
            print(f"[stt-rt] hallucination 필터링: '{text[:60]}'", flush=True)
            return ""
        if _is_prompt_echo(text, prior_text):
            print(f"[stt-rt] prompt echo 필터링: '{text[:60]}'", flush=True)
            return ""
        return text
    except Exception as e:
        print(f"[stt-rt] Whisper 호출 실패: {e}", flush=True)
        return ""
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def is_speech_in_chunk(
    audio_bytes: bytes, threshold: float = REALTIME_RMS_THRESHOLD
) -> bool:
    """청크에 발화가 있는지 RMS 기반으로 사전 판단 (jarvis 패턴 차용).

    Opus/WebM → ffmpeg pipe로 PCM 디코드 → 전체 RMS 계산.
    임계값 미만이면 침묵 → Whisper 호출 스킵 → API 절약.
    디코드 실패 시 보수적으로 True (Whisper에 맡김).
    """
    try:
        import numpy as np
    except ImportError:
        return True
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
            input=audio_bytes,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return True  # ffmpeg 없으면 디코드 불가 — 보수적 True
    if result.returncode != 0 or not result.stdout:
        return True
    pcm = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if pcm.size == 0:
        return False
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    return rms > threshold


# ── 호환성 유틸 ──────────────────────────────────────────────────────────

def find_speech_at(timestamp_sec: float, segments: list[SpeechSegment]) -> str:
    """주어진 시각이 포함된 STT 세그먼트의 텍스트를 반환. 없으면 빈 문자열."""
    for seg in segments:
        if seg.start <= timestamp_sec <= seg.end:
            return seg.text
    return ""


def extract_and_transcribe(media_path: str) -> list[SpeechSegment]:
    """미디어 → ffmpeg wav → Groq Whisper → 임시 정리. 호출자 공용 helper."""
    wav_path = extract_audio(media_path)
    if not wav_path:
        return []
    try:
        return transcribe(wav_path)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


# ── WAV 유틸 ─────────────────────────────────────────────────────────────

def _wav_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _extract_wav_chunk(wav_path: str, start_sec: float, end_sec: float) -> str | None:
    """WAV의 [start, end] 구간을 새 임시 WAV로 저장 후 경로 반환."""
    try:
        with wave.open(wav_path, "rb") as src:
            params = src.getparams()
            rate = src.getframerate()
            start_frame = int(start_sec * rate)
            n_frames = max(0, int((end_sec - start_sec) * rate))
            src.setpos(start_frame)
            frames = src.readframes(n_frames)
    except (wave.Error, OSError) as e:
        print(f"[stt] WAV 청크 추출 실패: {e}", flush=True)
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)
    return tmp.name
