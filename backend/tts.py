"""텍스트 → 음성 (Edge TTS) 모듈.

Microsoft Edge TTS는 무료 클라우드 TTS — 자연스러운 한국어(`ko-KR-InJoonNeural` 등) 제공.
jarvis-assistant의 _speak_edge 패턴 차용 (assistant.py:942-957).

비동기 API라 FastAPI 핸들러에서 직접 await 가능. edge-tts 미설치 시 graceful degradation —
synthesize는 None 반환, 호출자(서버)는 503으로 알린다.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress

try:
    import edge_tts
    _TTS_LIB_OK = True
except ImportError:
    edge_tts = None  # type: ignore
    _TTS_LIB_OK = False


# 자연스러운 한국어 남성 보이스 (jarvis와 동일).
# 다른 옵션: ko-KR-SunHiNeural (여성), ko-KR-BongJinNeural, ko-KR-HyunsuNeural 등.
DEFAULT_VOICE = "ko-KR-InJoonNeural"


def tts_available() -> bool:
    """TTS 사용 가능 여부 (edge-tts 패키지 설치 + 네트워크 가용)."""
    return _TTS_LIB_OK


async def synthesize(text: str, voice: str = DEFAULT_VOICE) -> bytes | None:
    """text → mp3 bytes (Edge TTS).

    실패/미설치/빈 텍스트 시 None. edge_tts는 자체적으로 임시 파일에 save한 뒤 읽어 반환.
    """
    if not _TTS_LIB_OK or not text or not text.strip():
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp.name)
        with open(tmp.name, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"[tts] Edge TTS 실패: {e}", flush=True)
        return None
    finally:
        with suppress(OSError):
            os.unlink(tmp.name)
