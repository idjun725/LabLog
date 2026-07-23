"""LabLog FastAPI 백엔드.

엔드포인트:
  GET  /api/health           — 헬스체크
  POST /api/analyze/video    — 업로드 영상 일괄 분석
  POST /api/tts              — 텍스트 → mp3 음성 (Edge TTS)
  WS   /ws/analyze           — 실시간 프레임 스트림 분석
  WS   /ws/transcribe        — 실시간 STT 스트림 (Groq Whisper)

실행:
  uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from analyzer import (
    LabLogAnalyzer,
    format_timestamp,
    get_assist_messages,
    parse_timestamp,
)
from class_translator import translate_classes
from gru_classifier import classify_records
from report_generator import generate_report
from stt import (
    extract_and_transcribe,
    find_speech_at,
    is_speech_in_chunk,
    transcribe_bytes,
)
from tts import synthesize as tts_synthesize, tts_available

ANALYZER: LabLogAnalyzer | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global ANALYZER
    print("[startup] YOLO + EasyOCR 로드 중 (수 분 소요될 수 있음)...")
    ANALYZER = LabLogAnalyzer()
    print("[startup] 모델 로드 완료")
    yield
    print("[shutdown] 종료")


app = FastAPI(title="LabLog Backend", version="0.1.0", lifespan=lifespan)

# 개발: localhost/127.0.0.1 모든 포트 허용. 배포: 환경변수 PROD_FRONTEND_ORIGIN으로
# 정확한 Vercel/배포 origin 한 개 추가 (e.g. "https://lablog.vercel.app").
# 콤마로 구분해 여러 도메인 지정 가능 ("https://a.vercel.app,https://lablog.com").
_prod_origins = [
    o.strip() for o in os.getenv("PROD_FRONTEND_ORIGIN", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_prod_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_analyzer() -> LabLogAnalyzer:
    if ANALYZER is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")
    return ANALYZER


@app.get("/")
def root():
    """HF Spaces 인프라가 GET /로 헬스 프로브를 보냄. 404 누적 시 컨테이너 죽임."""
    return {"service": "lablog-backend", "health": "/api/health"}


@app.get("/robots.txt")
def robots():
    return Response("User-agent: *\nDisallow: /\n", media_type="text/plain")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": ANALYZER is not None}


@app.post("/api/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    sample_fps: float = Query(1.0, gt=0, le=10, description="초당 분석 프레임 수"),
    custom_classes: str = Form(
        "",
        description="쉼표로 구분한 개방 어휘(YOLO-World) 탐지 대상 클래스 — best.pt 25개 "
        "클래스 밖 사물을 프롬프트로 추가 탐지. 비어 있으면 best.pt만 사용.",
    ),
    allowed_classes: str = Form(
        "",
        description="쉼표로 구분한 best.pt 25개 클래스 중 허용 목록(영문 클래스명, "
        "vectorizer.YOLO_VOCAB과 동일). 비어 있으면 25개 전부 탐지 — 지정하면 "
        "그 클래스들로만 탐지를 제한(다른 24개와의 오분류 방지).",
    ),
):
    analyzer = _require_analyzer()

    classes = [c.strip() for c in custom_classes.split(",") if c.strip()]
    if classes:
        # YOLO-World 텍스트 인코더는 영어 전용 — 한글 클래스명은 여기서 미리 번역
        # (프레임 루프 안에서 매번 부르면 안 되므로 요청당 1회, 루프 진입 전에 처리)
        classes = await asyncio.to_thread(translate_classes, classes)

    restrict_to = [c.strip() for c in allowed_classes.split(",") if c.strip()]

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        body = await file.read()
        tmp.write(body)
        tmp_path = tmp.name
    print(
        f"[server] 업로드 수신: {file.filename} ({len(body)/1024/1024:.1f}MB) → {tmp_path}"
        + (f", custom_classes={classes}" if classes else "")
        + (f", allowed_classes={restrict_to}" if restrict_to else ""),
        flush=True,
    )

    t_req = time.monotonic()
    try:
        # 영상 분석은 CPU 바운드라 이벤트 루프를 막지 않도록 스레드로 분리
        results = await asyncio.to_thread(
            analyzer.analyze_video,
            tmp_path,
            sample_fps,
            custom_classes=classes,
            allowed_classes=restrict_to,
        )
        print(
            f"[server] 분석 응답 전송 — records {len(results)}개, "
            f"요청 처리 {time.monotonic() - t_req:.1f}초",
            flush=True,
        )
        return {
            "filename": file.filename,
            "frame_count": len(results),
            "records": [asdict(r) for r in results],
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class ClassifyRequest(BaseModel):
    records: list[dict]


@app.post("/api/classify")
async def classify_endpoint(req: ClassifyRequest):
    """records 시퀀스를 받아 학습된 GRU로 각 시점의 실험 단계를 예측한다."""
    if not req.records:
        raise HTTPException(status_code=400, detail="records가 비어있습니다.")

    print(f"[server] 단계 분류 요청: records {len(req.records)}개", flush=True)
    t_req = time.monotonic()
    try:
        labels = await asyncio.to_thread(classify_records, req.records)
        print(
            f"[server] 분류 응답 — 처리 {time.monotonic() - t_req:.2f}초",
            flush=True,
        )
        return {
            "phases": [
                {"timestamp": r.get("timestamp", ""), "phase": label}
                for r, label in zip(req.records, labels)
            ]
        }
    except RuntimeError as e:
        # 가중치 파일 없음 등 설정 문제
        print(f"[server] 분류 실패 (설정): {e}", flush=True)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"[server] 분류 실패 (예외): {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"분류 실패: {e}")


class ReportGenerateRequest(BaseModel):
    filename: str | None = None
    records: list[dict]
    phases: list[dict] | None = None  # GRU 단계 분류 결과 (있으면 LLM 컨텍스트로 사용)
    # 사용자가 업로드/촬영 전 입력한 실험 기본 정보 (제목·주제·날짜·가설·기타).
    # 모두 선택사항이라 빈 dict이거나 None일 수 있다.
    info: dict | None = None


@app.post("/api/report/generate")
async def generate_report_endpoint(req: ReportGenerateRequest):
    """records(+ 선택적으로 phases)를 받아 Groq LLM으로 구조화된 실험 보고서 생성."""
    if not req.records:
        raise HTTPException(status_code=400, detail="records가 비어있습니다.")

    print(
        f"[server] 보고서 생성 요청: records {len(req.records)}개, "
        f"phases={'있음' if req.phases else '없음'}, file={req.filename}",
        flush=True,
    )
    t_req = time.monotonic()
    try:
        report = await asyncio.to_thread(
            generate_report, req.records, req.filename, req.phases, req.info
        )
        print(
            f"[server] 보고서 응답 전송 — 요청 처리 {time.monotonic() - t_req:.1f}초",
            flush=True,
        )
        return report.model_dump()
    except RuntimeError as e:
        # GROQ_API_KEY 미설정 / SDK 미설치 — 사용자에게 명시적으로 알린다
        print(f"[server] 보고서 생성 실패 (설정 문제): {e}", flush=True)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"[server] 보고서 생성 실패 (예외): {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"보고서 생성 실패: {e}")


@app.post("/api/record/transcribe")
async def transcribe_recording(
    audio: UploadFile = File(...),
    records: str = Form(...),
):
    """녹화된 오디오를 STT로 전사하고, 기존 records의 speech 필드에 시각 매칭으로 채운다.

    Body (multipart):
      - audio: webm/opus 등 ffmpeg가 디코드 가능한 오디오
      - records: JSON 문자열 (AnalysisRecord 배열)
    Returns:
      - records: 업데이트된 배열 (speech 필드만 채워짐)
      - segments_count: 매칭에 사용된 STT 세그먼트 수
    """
    try:
        records_list = json.loads(records)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"records JSON 파싱 실패: {e}")
    if not isinstance(records_list, list):
        raise HTTPException(status_code=400, detail="records는 배열이어야 합니다.")

    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        body = await audio.read()
        tmp.write(body)
        tmp_path = tmp.name
    print(
        f"[server] 오디오 전사 요청: {audio.filename} ({len(body) / 1024:.1f}KB), "
        f"records {len(records_list)}개",
        flush=True,
    )

    t_req = time.monotonic()
    try:
        segments = await asyncio.to_thread(extract_and_transcribe, tmp_path)
        for r in records_list:
            ts = r.get("timestamp", "")
            t_sec = parse_timestamp(ts)
            r["speech"] = find_speech_at(t_sec, segments) or r.get("speech", "")
        print(
            f"[server] 전사 응답 — segments {len(segments)}개, "
            f"처리 {time.monotonic() - t_req:.1f}초",
            flush=True,
        )
        return {"records": records_list, "segments_count": len(segments)}
    except Exception as e:
        print(f"[server] 전사 실패: {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"전사 실패: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None  # 미지정 시 tts.DEFAULT_VOICE


@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    """텍스트 → mp3 음성 (Edge TTS). 실시간 보조 안내용."""
    if not tts_available():
        raise HTTPException(
            status_code=503,
            detail="edge-tts 패키지가 설치되지 않았습니다. pip install -r requirements.txt",
        )
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text가 비어있습니다.")

    audio = await tts_synthesize(text, req.voice) if req.voice else await tts_synthesize(text)
    if audio is None:
        raise HTTPException(status_code=500, detail="TTS 생성 실패 — 콘솔 로그 확인")
    return Response(content=audio, media_type="audio/mpeg")


@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    """실시간 프레임 스트림.

    프로토콜:
      클라이언트 → 서버: bytes (JPEG 인코딩된 프레임)
      서버 → 클라이언트: JSON
        {
          "timestamp": "00:00:05",
          "yolo": {"bottle": 0.91},
          "ocr": "4.0g",
          "ocr_confidence": 0.87,
          "avg_yolo_confidence": 0.78,
          "brightness": 132.4,
          "assist": "카메라가 가려져 있습니다. 시야를 확보해주십시오."   // 조건 충족 시에만
        }
    """
    await websocket.accept()
    if ANALYZER is None:
        await websocket.close(code=1011, reason="모델이 아직 로드되지 않았습니다.")
        return

    start = time.time()
    try:
        while True:
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "프레임 디코드 실패"})
                continue

            ts = format_timestamp(time.time() - start)
            # 추론은 동기 코드이므로 스레드로 옮겨 이벤트 루프 블로킹 방지
            result = await asyncio.to_thread(ANALYZER.analyze_frame, frame, ts)
            payload = asdict(result)
            messages = get_assist_messages(result)
            if messages:
                payload["assist"] = messages  # list[str] — 여러 조건이 동시 충족 가능
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """실시간 STT 스트림.

    클라이언트가 MediaRecorder timeslice로 보낸 WebM/Opus 청크를 받아 Groq Whisper로
    전사하고 결과를 즉시 전송한다. 침묵 청크는 RMS 사전 검사로 스킵해 API 절약.

    프로토콜:
      클라이언트 → 서버: bytes (3-5초 WebM/Opus 청크)
      서버 → 클라이언트: JSON
        {
          "chunk_index": 3,
          "elapsed_sec": 9.7,        // WS 연결 후 경과 (timestamp 매칭용)
          "text": "비커에 50ml 부었습니다",
          "skipped": false            // RMS 임계값 미만 시 true (Whisper 미호출)
        }
    """
    await websocket.accept()
    start = time.time()
    chunk_index = 0
    # 세션별 누적 컨텍스트 — Whisper prompt로 매 청크마다 함께 보내 문맥 연속성 ↑.
    # 길이는 transcribe_bytes 내부에서 토큰 한도에 맞게 잘림.
    running_context = ""
    # 연속 빈손 카운터 — N회 연속 무수확이면 컨텍스트 리셋 (stale echo 방지).
    empty_streak = 0
    CONTEXT_RESET_AFTER = 3
    print("[ws-transcribe] 클라이언트 연결됨", flush=True)

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk_index += 1
            elapsed = time.time() - start

            if not await asyncio.to_thread(is_speech_in_chunk, data):
                empty_streak += 1
                if empty_streak >= CONTEXT_RESET_AFTER and running_context:
                    running_context = ""
                    print(f"[ws-transcribe] 연속 침묵 {empty_streak}회 — 컨텍스트 리셋", flush=True)
                await websocket.send_json({
                    "chunk_index": chunk_index,
                    "elapsed_sec": round(elapsed, 2),
                    "text": "",
                    "skipped": True,
                })
                continue

            text = await asyncio.to_thread(
                transcribe_bytes, data, ".webm", "ko", running_context
            )
            if text:
                empty_streak = 0
                running_context = f"{running_context} {text}".strip()
            else:
                # Whisper가 텍스트 반환 안 함 또는 hallucination/echo 필터링됨
                empty_streak += 1
                if empty_streak >= CONTEXT_RESET_AFTER and running_context:
                    running_context = ""
                    print(f"[ws-transcribe] 연속 빈손 {empty_streak}회 — 컨텍스트 리셋", flush=True)
            await websocket.send_json({
                "chunk_index": chunk_index,
                "elapsed_sec": round(elapsed, 2),
                "text": text,
                "skipped": False,
            })
            if text:
                print(
                    f"[ws-transcribe] #{chunk_index} (t={elapsed:.1f}s): {text[:60]}",
                    flush=True,
                )
    except WebSocketDisconnect:
        print(
            f"[ws-transcribe] 연결 해제 (총 청크 {chunk_index}개)",
            flush=True,
        )
