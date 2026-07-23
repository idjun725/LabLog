# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LabLog is a multimodal AI system that records / uploads a science experiment video and produces a structured Korean high-school lab report. The pipeline:

1. **Video frame analysis** — Three text/object models with explicit role separation:
   - **YOLO object detection** (`best.pt`, fine-tuned 25 lab equipment classes, 2026-05; retrain in progress on `chemistry-lab-object-detection-topas` to broaden coverage incl. stopwatches etc.).
   - **EasyOCR** for general labels/measurements — runs **only on YOLO bbox crops** (each OTSU-binarized for accuracy).
   - **7-segment YOLOv8 detector** (`seven_segment_weights.pt`, lazy-loaded if present) for digital displays — runs **on the full frame** (fast, catches digits regardless of whether the device is in `best.pt` vocab).
   - Two temporal-consistency filters reject "flashing" detections: one for YOLO labels, one for OCR text — both `PERSISTENCE_WINDOW=4` / `PERSISTENCE_MIN_VOTES=2`. **Trade-off**: text on un-detected non-digit objects (e.g., a chemical bottle label YOLO misses) is lost — the user accepted this in exchange for noise reduction, with the plan to broaden `best.pt` vocab.
2. **Audio analysis** — **하이브리드 STT** (배치는 Google STT, 실시간은 Groq Whisper):
   - **Batch (Google Cloud Speech-to-Text)**: post-hoc transcription via ffmpeg → wav → 55-second chunks → `recognize()` per chunk with word_time_offsets. Used by `/api/analyze/video` and `/api/record/transcribe`. `SpeechContext.phrases`로 실험 도메인 어휘(비커·시험관·전자저울·pH 등) boost. Whisper의 TPM 한도·환각을 피하기 위해 배치만 Google로 이동.
   - **Realtime (Groq Whisper `whisper-large-v3-turbo`)**: `/ws/transcribe` WebSocket — accepts 6s WebM/Opus chunks from MediaRecorder, gated by `is_speech_in_chunk()` RMS VAD (skips silent chunks to save API calls), sends back `TranscribeChunkMessage` per chunk. Hallucination defense: `temperature=0`, regex repetition filter, sentence-repetition filter, prompt-echo filter, plus 3-strike `empty_streak` context reset on the server side. RecordPage streams audio chunks during recording and matches transcripts to records by `elapsed_sec`. 짧은 청크 단위라 Groq 한도 영향 적고 streaming 지연이 낮아 Whisper 유지.
   - **TTS**: Edge TTS (`ko-KR-InJoonNeural`) via `/api/tts` — RecordPage speaks the latest assist message every ≥6s during recording.
3. **Vector conversion** — multimodal `records` → 801-dim feature vectors (25 YOLO vocab one-hot + 6 OCR + 2 STT baseline + 768 SBERT sentence embedding). SBERT via `jhgan/ko-sroberta-multitask` on the `speech` field — batch-encoded per `vectorize_records()` call; empty strings return zero vectors (no encode cost, and preserves the `has_speech` binary as an explicit signal). A `use_sbert=False` mode returns the 33-dim baseline — kept for the LOEO evaluator.
4. **Stage classification** — bidirectional GRU (hidden=32, 1 layer) classifies each timestamp into one of 4 phases: `준비 / 반응 / 측정 및 관찰 / 정리`.
5. **Report generation** — Groq LLM via OpenAI-compatible API. Currently `openai/gpt-oss-20b` (only the `openai/gpt-oss-*` family supports strict `json_schema` on Groq; 20b chosen over 120b for larger TPM headroom on the free tier). When GRU phases are available, `_format_records()` chunks consecutive same-phase records into a single block and dedupes objects/OCR/speech within each block — significantly shrinks the prompt for long videos.

There is **no backend database**. Drafts live entirely in browser `localStorage` under `lablog:drafts`. The realtime WebSocket flow stores records on the frontend; the upload flow returns them in the HTTP response.

## Commands

**Frontend** (run from `frontend/`):
```powershell
npm run dev          # Vite dev server on :5173
npm run build        # tsc -b + vite build → dist/
npm run lint         # ESLint
npm run preview      # serve dist/ on :4173
```

**Backend** (run from `backend/`):
```powershell
# Always invoke via venv's python explicitly. On Windows Store Python the
# `uvicorn` PATH entry may resolve to global Store-Python (missing deps).
.\.venv\Scripts\python.exe -m uvicorn server:app --reload --port 8000

# Train the GRU stage classifier (~30s on CPU, writes gru_weights.pt).
# First run downloads SBERT model ~420MB into HF cache; subsequent runs reuse it.
.\.venv\Scripts\python.exe train_gru.py

# Evaluate baseline vs SBERT via 5-fold Leave-One-Experiment-Out CV + Wilcoxon.
# Reports fold-wise accuracy/macroF1 (mean±std), signed-rank p-value, confusion matrices.
.\.venv\Scripts\python.exe evaluate_gru.py

# Add a new hand-labeled experiment to training_data.json.
# Re-runs analyze_video() on <video_path>, prints records as a table,
# then prompts for range-based phase labels ("1-5: 1", "6-12: 2", …).
# Auto-backs up training_data.json to training_data.backup_YYYYMMDD_HHMMSS.json.
.\.venv\Scripts\python.exe label_experiment.py <video_path> [--title "제목"] [--dry-run]

# Train the 7-segment YOLOv8 detector (CPU OK for small dataset, writes seven_segment_weights.pt)
.\.venv\Scripts\python.exe train_seven_segment.py

# Train/refresh the MAIN YOLO from Roboflow (CPU = hours; offload to Kaggle GPU recommended).
# Backs up existing best.pt to best.backup_YYYYMMDD_HHMMSS.pt before overwriting.
.\.venv\Scripts\python.exe train_main_yolo.py
```

Training the **main YOLO on Kaggle GPU** (recommended for full-size datasets): see [backend/KAGGLE_TRAINING.md](backend/KAGGLE_TRAINING.md) — 6 paste-ready notebook cells. Set Kaggle Secret `ROBOFLOW_API_KEY`. **Use T4, not P100** — recent PyTorch wheels drop compute 6.0 support and P100 crashes with `cudaErrorNoKernelImageForDevice`.

**Environment variables** (PowerShell):
```powershell
# Persistent (User scope) — requires NEW PowerShell window to take effect
[Environment]::SetEnvironmentVariable('GROQ_API_KEY', 'gsk-...', 'User')
[Environment]::SetEnvironmentVariable('GOOGLE_APPLICATION_CREDENTIALS', 'c:\Coding\lablog\backend\gcp-credentials.json', 'User')

# Session-only (current window) — for quick testing
$env:GROQ_API_KEY = 'gsk-...'

# Pull User-scope value into current session without opening new window
$env:GROQ_API_KEY = [Environment]::GetEnvironmentVariable('GROQ_API_KEY', 'User')
```

**STT 자격증명은 path별로 분리**:
- **배치 STT (Google)**: `GCP_CREDENTIALS_JSON` (HF Spaces secret 권장 — Service Account JSON 내용 통째로) **또는** `GOOGLE_APPLICATION_CREDENTIALS` (로컬 — 파일 경로). 둘 중 하나 있으면 동작. 없으면 silent fallback으로 빈 SpeechSegment[] 반환.
- **실시간 STT (Whisper)** + **보고서 생성** + **단계 분류 없이** **TTS도 별개**: `GROQ_API_KEY`. 없으면 `/api/report/generate`는 503, 실시간 STT는 silent skip.
- ffmpeg는 두 path 모두에 필요한 사전 처리 (배치는 WAV 추출, 실시간은 VAD용 PCM 디코드).
- `ROBOFLOW_API_KEY`는 학습 스크립트만 (`train_seven_segment.py`, `train_main_yolo.py`).
- `edge-tts`는 `/api/tts`에 필수 (없으면 503).

## Deployment

- **Backend → Hugging Face Spaces** ([backend/Dockerfile](backend/Dockerfile)): `python:3.12-slim` base + apt `ffmpeg / libgl1 / libglib2.0-0`, non-root `user` (uid 1000), listens on port `7860`. Build layer pre-downloads (a) EasyOCR CRAFT + ko/en recognition models (~150MB) and (b) SBERT `jhgan/ko-sroberta-multitask` (~420MB) — otherwise first request on free-tier CPU stalls 5–15 min on EasyOCR and adds another ~1 min on SBERT while downloading. Add `GCP_CREDENTIALS_JSON` (JSON contents inline) and `GROQ_API_KEY` as Space secrets; don't ship `gcp-credentials.json` on disk.
- **Frontend → Vercel** ([frontend/vercel.json](frontend/vercel.json)): single SPA rewrite `/(.*) → /index.html` so React Router deep links (`/analysis/:id`, `/report/:id`, etc.) don't 404 on hard reload. Set the backend URL via Vite env var at build time.

## Architecture

### Surface — endpoints and routes

**Backend FastAPI** (`backend/server.py`):

| Verb | Path | Purpose |
|---|---|---|
| GET  | `/api/health` | model_loaded check |
| POST | `/api/analyze/video` | upload video → records (HTTP batch; includes batch STT) |
| POST | `/api/classify` | records → GRU phase labels |
| POST | `/api/record/transcribe` | webm audio + records → records w/ speech filled (batch STT) |
| POST | `/api/report/generate` | records (+ optional phases, optional `info`) → GeneratedReport via Groq |
| POST | `/api/tts` | text → mp3 (Edge TTS); 503 if `edge-tts` not installed |
| WS   | `/ws/analyze` | binary JPEG frames → JSON AnalysisRecord per frame (+ `assist?: string[]`) |
| WS   | `/ws/transcribe` | binary WebM/Opus chunks → JSON `TranscribeChunkMessage` per chunk (realtime STT) |
| GET  | `/` | HF Spaces root health probe |
| GET  | `/robots.txt` | static robots response |

CORS allows `localhost`/`127.0.0.1` on any port via regex (dev convenience).

**Frontend routes** (`src/App.tsx`):
`/`, `/upload`, `/record`, `/archive`, `/drafts`, `/loading`, `/analysis/:id`, `/report/:id`, `*` → redirect to `/`.

### Backend module boundaries (intentionally flat — no package)

- **`analyzer.py`** — `AnalysisResult` dataclass is the canonical record schema (its fields appear unchanged in WS/HTTP JSON). `analyze_video()` runs STT in a `ThreadPoolExecutor` parallel to the frame loop and merges segments by timestamp afterwards. Implements:
  - **YOLO temporal-consistency filter**: `PERSISTENCE_WINDOW=4` / `PERSISTENCE_MIN_VOTES=2`. A label must appear in ≥2 of last 4 samples **and** be present in the current frame to count — except a label with confidence ≥ `PERSISTENCE_HIGH_CONF_BYPASS=0.5` is confirmed immediately regardless of vote count.
  - **OCR temporal-consistency filter**: same `PERSISTENCE_WINDOW`/`MIN_VOTES` as YOLO, over a `recent_ocr_texts` deque of joined OCR strings; text must repeat ≥2 times before a record is saved on `ocr_changed`.
  - **`detect_text_in_regions(frame, detections)`** — the production OCR pipeline. Runs **two models with explicit roles**:
    1. **7-segment YOLO on the FULL frame** (single call, fast) via `seven_segment_classifier.detect_digits()`. Catches digital displays regardless of whether the device is in `best.pt` vocab.
    2. **EasyOCR on each YOLO bbox crop** (OTSU-binarized first). Catches printed labels/measurements on detected equipment.
    - Results from both sources are deduped by exact text match, then joined with `OCR_TEXT_SEPARATOR` (`" | "`) into the single `ocr` field.
    - **No full-frame EasyOCR fallback** — the user removed it after switching to full-frame seg7, accepting the trade-off that text on undetected non-digit objects is missed.
  - **OCR noise filter (`_filter_ocr_raw`)**, used by `detect_text_in_regions` and the 7-seg path: `OCR_MIN_CONFIDENCE=0.5`, `OCR_MIN_LENGTH=2`, `OCR_MIN_DIGIT_RATIO=0.25`, plus `NUMERIC_PATTERN` (must contain a digit).
- **`seven_segment_classifier.py`** — Lazy-loading wrapper around an ultralytics `YOLO(seven_segment_weights.pt)`. `is_available()` is a cheap path-existence check safe to call per-frame; `detect_digits(crop, conf_threshold)` returns `(joined_digits_string, avg_confidence) | None` with digits sorted left-to-right by bbox x-coord. Grayscale input is auto-converted to BGR (YOLO requires 3 channels). `_load_failed` flag prevents repeated load-failure attempts. If the weights file is absent, `is_available()` returns False and `detect_text_in_regions` skips the seg7 call entirely — graceful degradation.
- **`train_seven_segment.py`** / **`train_main_yolo.py`** — CLI training scripts following the same pattern: read `ROBOFLOW_API_KEY`, download the named Roboflow project/version as `yolov8` format (reused if `data.yaml` already exists), train via `ultralytics.YOLO.train()`, copy the resulting `runs/<name>/weights/best.pt` to the canonical location (`seven_segment_weights.pt` / `best.pt`). `train_main_yolo.py` additionally **backs up the existing `best.pt`** to `best.backup_YYYYMMDD_HHMMSS.pt` before overwriting — do not skip this when modifying the script.
- **`stt.py`** — **하이브리드 STT** (배치 path와 실시간 path가 다른 엔진 사용). 두 path는 서로 독립적으로 비활성화 가능 — 한 쪽 자격증명만 있어도 다른 쪽은 silent skip.
  - **Batch (Google STT)**: `extract_and_transcribe(media_path)` → `SpeechSegment[]` with timestamps. ffmpeg → 16kHz mono PCM WAV → `GOOGLE_STT_CHUNK_SECONDS=55` 단위 청크 분할 → `client.recognize(config, audio)` per chunk with `enable_word_time_offsets=True` and `SpeechContext(phrases=GOOGLE_STT_PHRASES, boost=15.0)`. `find_speech_at(t_sec, segments)` does the record-to-speech matching.
    - **자격증명 두 가지 경로** (`_get_google_client`): `GCP_CREDENTIALS_JSON` env(JSON 내용 전체, HF Spaces secret 권장) 우선 → fallback으로 `GOOGLE_APPLICATION_CREDENTIALS`(파일 경로, 로컬 개발).
  - **Realtime (Groq Whisper)**: `transcribe_bytes(audio_bytes, suffix, language, prior_text)` → `str`. `whisper-large-v3-turbo`, `response_format="json"`, `temperature=0.0`. `prior_text` is trimmed and prepended to `WHISPER_PROMPT` for vocab/style continuity.
  - **VAD**: `is_speech_in_chunk(audio_bytes)` decodes via ffmpeg pipe → s16le PCM → numpy RMS, compared against `REALTIME_RMS_THRESHOLD`. Below → skip Whisper. ffmpeg/numpy missing → returns True (conservative).
  - **Hallucination defense (3-layer)** in `transcribe_bytes` (Whisper 한정 — Google STT는 환각이 적어 불필요): (1) `_HALLUCINATION_PATTERN` regex `(.{1,3})\1{3,}` ("난난난난"), (2) `_has_sentence_repetition` ("X. X. X." snowballs), (3) `_is_prompt_echo` (Whisper regurgitating its own `prior_text`). All three return `""` so the caller treats it as a silent chunk.
  - On ffmpeg failure the **last 5 lines** of stderr are shown (not the head, since ffmpeg's banner fills the first 1KB).
- **`tts.py`** — Edge TTS (Microsoft, free). `synthesize(text, voice=DEFAULT_VOICE)` is `async` (FastAPI handler awaits directly) and returns `bytes | None`. `DEFAULT_VOICE="ko-KR-InJoonNeural"`. Missing package → `tts_available()` returns False → `/api/tts` returns 503.
- **`train_common.py`** — Shared helpers used by both `train_main_yolo.py` and `train_seven_segment.py`: `backup_with_timestamp(path)`, `download_roboflow_dataset(workspace, project, version, location)`, `train_and_export(base_model, data_yaml, runs_dir, run_name, weights_out, epochs, imgsz, patience, device)`.
- **`vectorizer.py`** — `YOLO_VOCAB` ordering is the feature-index order. **Changing it requires retraining the GRU** (state_dict shape changes). `vectorize_records(records, use_sbert=True)` is the production path (801-dim); passes speech texts to `sbert_encoder.encode()` in a single batch. Legacy `vectorize_record()` returns baseline 33-dim without SBERT — kept only for the LOEO evaluator's baseline arm.
- **`sbert_encoder.py`** — Singleton lazy-loader for `jhgan/ko-sroberta-multitask` (768-dim). `encode(texts)` skips SBERT for empty/whitespace texts and returns zero rows for them (cost + explicit signal). `is_available()` catches missing package / model download failure so callers can fall back. Model cache lives in HF's default location (`~/.cache/huggingface`); pre-downloaded in the Dockerfile build layer for HF Spaces cold-start.
- **`evaluate_gru.py`** — 5-fold Leave-One-Experiment-Out CV comparing baseline (33-dim) vs SBERT (801-dim). Same hyperparameters (Adam lr=1e-3, 200 epochs, seed=42) so accuracy delta is attributable to the feature change alone. Reports fold-wise accuracy/macroF1 (mean±std), Wilcoxon signed-rank p-value (both one-sided `SBERT>Baseline` and two-sided), and confusion matrices for each mode. **n_folds=5 → Wilcoxon is severely underpowered; use p-values as directional hints only.**
- **`gru_model.py`** — `PHASES` list order = output class index. **Changing it requires retraining** (`gru_weights.pt` becomes incompatible; `gru_classifier.py` catches state_dict mismatch and surfaces a retrain prompt).
- **`gru_classifier.py`** — lazy-loads weights into a module-level `_model` singleton.
- **`report_generator.py`** — Groq via `openai>=1.50` SDK with `base_url="https://api.groq.com/openai/v1"`. **Only `openai/gpt-oss-20b` and `openai/gpt-oss-120b` support strict `json_schema`** (required by `client.beta.chat.completions.parse(response_format=PydanticModel)`). Other Groq models (llama, qwen) return 400. `generate_report(records, filename, phases=None, info=None)` — when `phases` provided, `_format_records` groups consecutive same-phase records into one block per phase and dedupes objects/OCR/speech texts within the block (token compression for long videos). When `phases=None`, it falls back to one line per record. When `info` (dict) provided, `_format_info` injects user-supplied experiment context (`실험 제목`, `실험 주제`, `실험 날짜`, `가설`, `기타 메모`) into the user message — these are user-stated facts the LLM should reflect, not infer. The Groq `parse()` returns the parsed Pydantic instance on `response.choices[0].message.parsed`; `None` triggers a `RuntimeError` (no auto-fallback).
- **`server.py`** — FastAPI endpoints. Two realtime WebSockets:
  - `/ws/analyze` calls `analyze_frame()` per-frame and does **not** apply the persistence filter or STT (those need batch context), but **does** use cropping-based OCR (same `detect_text_in_regions` as batch). `get_assist_messages()` is called per WS frame and the resulting `string[]` is attached to the payload as `assist?` (for RecordPage's overlay banner).
  - `/ws/transcribe` maintains a per-session `running_context` string passed to `transcribe_bytes(..., prior_text=running_context)`. After 3 consecutive empty/silent chunks (`empty_streak >= CONTEXT_RESET_AFTER=3`), it resets `running_context = ""` to prevent stale context from triggering Whisper to echo old prompt fragments forever.

### Tunable constants (where each lives)

| File | Constant | Effect |
|---|---|---|
| `analyzer.py` | `OCR_MIN_CONFIDENCE` (0.5), `OCR_MIN_LENGTH` (2), `OCR_MIN_DIGIT_RATIO` (0.25), `NUMERIC_PATTERN` | OCR noise filter — raise to be more aggressive |
| `analyzer.py` | `PERSISTENCE_WINDOW` (4), `PERSISTENCE_MIN_VOTES` (2) | YOLO **and** OCR temporal-consistency (same constants) — raise to filter more flicker |
| `analyzer.py` | `PERSISTENCE_HIGH_CONF_BYPASS` (0.5) | YOLO-only — confidence at/above this skips the vote-count requirement entirely (single-frame confirm) |
| `analyzer.py` | `get_assist_messages()` thresholds | YOLO 0.4, OCR 0.65, brightness 40 |
| `analyzer.py` | `DEFAULT_YOLO_WEIGHTS` (`"best.pt"`) | Main YOLO weights file. Drop in `yolov8n.pt` for COCO classes, or a new `best.pt` after re-fine-tuning. API contract unchanged. |
| `seven_segment_classifier.py` | `DEFAULT_CONF_THRESHOLD` (0.25) | **Sole gate for 7-seg** — `_filter_ocr_raw` and the OCR temporal filter both skip the 7-seg path, so this single threshold decides what's accepted. Deliberately low (aggressive recall) so single-digit / fast-changing displays survive; raise to 0.35–0.4 if false positives surface. |
| `analyzer.py` | `OCR_TEXT_SEPARATOR` (`" \| "`) | Join string for multi-source OCR results. Downstream consumers must split on this exact value. |
| `stt.py` | `WHISPER_MODEL` (`"whisper-large-v3-turbo"`) | Groq Whisper model. `large-v3-turbo` chosen for speed/quality balance; `large-v3` higher quality but ~3× slower. |
| `stt.py` | `WHISPER_PROMPT` | Static vocab bias (실험 도구 + 측정 단위). Always prepended to realtime/batch prompts to nudge Whisper toward domain terms. |
| `stt.py` | `REALTIME_RMS_THRESHOLD` (0.015) | VAD floor. Below → skip Whisper call. Raise in noisy env (silence still hallucinates "난난난난"); lower if quiet speech is being dropped. |
| `stt.py` | `REALTIME_CONTEXT_MAX_CHARS` (80) | Trim length for `prior_text` prompt. Shrunk from 200 after prompt-echo feedback loop — too long → Whisper repeats the prompt tail; too short → loses style continuity. |
| `server.py` | `CONTEXT_RESET_AFTER` (3) in `ws_transcribe` | Consecutive silent/empty chunks before `running_context` resets — stops stale context from re-triggering echo forever. |
| `tts.py` | `DEFAULT_VOICE` (`"ko-KR-InJoonNeural"`) | Edge TTS voice. Alternatives: SunHi (female), BongJin, Hyunsu. |
| `train_main_yolo.py` | `ROBOFLOW_*`, `EPOCHS` (50), `IMGSZ` (640), `PATIENCE` (15) | Main YOLO fine-tuning. Edit constants at the top to retarget a new Roboflow project/version. |
| `vectorizer.py` | `YOLO_VOCAB`, `FEATURE_DIM` (= `SBERT_FEATURE_DIM` = 801), `BASELINE_FEATURE_DIM` (33) | Feature space — changing requires GRU retrain |
| `sbert_encoder.py` | `MODEL_NAME` (`"jhgan/ko-sroberta-multitask"`), `SBERT_DIM` (768) | SBERT model. Switching model → `FEATURE_DIM` changes → GRU retrain + Dockerfile pre-download update. |
| `gru_model.py` | `PHASES` | Output classes — changing requires retrain |
| `report_generator.py` | `MODEL_NAME` | Groq model — only `openai/gpt-oss-{20b,120b}` support strict json_schema |
| `frontend/src/pages/LoadingPage.tsx` | `sampleFps: 1` (in `uploadAndAnalyze` call) | Backend frame sample rate |
| `frontend/src/pages/RecordPage.tsx` | `SEND_INTERVAL_MS`, `REALTIME_STT_CHUNK_MS` (6000), `TTS_MIN_INTERVAL_MS` (6000) | WS analyze cadence, MediaRecorder timeslice for STT chunks, TTS playback throttle. |

### Frontend integration patterns

- **`src/api/lablog.ts` is the single HTTP integration point.** All other files import types and helpers from here; no other file does `fetch`. **Exception**: `RecordPage.tsx` opens its two `/ws/analyze` and `/ws/transcribe` connections directly via `new WebSocket(...)` (only importing the `WS_BASE` constant from `lablog.ts`) — it bypasses the module for WebSocket I/O specifically.
- **Background analysis pattern**: `uploadAndAnalyze()` is fire-and-forget — creates a `pending` draft when XHR bytes finish uploading, lets server processing continue detached from React lifecycle, updates draft via `xhr.onload`. AnalysisDetailPage polls every 2s while `status === 'pending'` or `audio_pending === true`.
- **Drafts vs Archive split is by report presence, not by source**:
  - `DraftsPage` (`/drafts`) → `listDrafts().filter((d) => !d.report)` (보고서 미생성)
  - `ArchivePage` (`/archive`) → `listDrafts().filter((d) => !!d.report)` (보고서 생성 완료)
  - A draft moves from drafts → archive the moment `generateReport()` succeeds. Same for re-generation (`report` stays present).
- **Routing split** (don't conflate):
  - `/analysis/:id` — raw record cards (real draft data only). Editable: title, info (ExperimentInfoForm controlled), and each record's OCR/speech (textarea, 저장/취소).
  - `/report/:id` — checks `getDraft(id)?.report` first; falls back to hardcoded `MOCK_REPORTS` (IDs `1`-`5`, `archive-1` … `archive-6`). **Both paths render through the same `ReportView` component** so the mock and real-draft layouts match exactly. Real-draft view passes `onTitleChange` + `analysisEditHref` to enable inline title edit and the "분석편집" Link; mocks omit both. `MOCK_REPORTS` is stored as `Record<string, { cardTitle, report: GeneratedReport }>` — i.e. mocks are pre-shaped in the **same `GeneratedReport` schema** the LLM emits.
- **Auto file naming**: `generateDraftTitle()` returns `YYMMDD-N` (e.g. `260605-1`, `260605-2`). Day counter persisted under `lablog:titleSequenceByDay` so deletions don't recycle numbers. `addDraft()` and `uploadAndAnalyze()` both use it when no title is given. User can rename via `updateDraftTitle()` (inline edit on both Analysis & Report pages).
- **ExperimentInfoForm component** has dual mode:
  - **Uncontrolled** (no `value`/`onChange` props): reads/writes `lablog:experimentInfo` localStorage directly. Used on UploadPage/RecordPage pre-analysis.
  - **Controlled** (`value`+`onChange` given): no localStorage touch. Used on AnalysisDetailPage to edit `draft.info` after the fact.
  - On `addDraft`/`uploadAndAnalyze` the pending value is **copied** into the draft (via `info` option), then `clearPendingExperimentInfo()` wipes the form.
- **TTS playback** (RecordPage): on each `assist?: string[]` from `/ws/analyze`, the first message is sent through `synthesizeTTS()` → mp3 Blob → `Audio` element, throttled by `TTS_MIN_INTERVAL_MS=6000` to avoid speech overlap.
- **localStorage keys**:
  - `lablog:drafts` — Draft[]
  - `lablog:hidden-mocks` — string[] of mock IDs the user has "deleted"
  - `lablog:preferredCameraId` — RecordPage's last-used camera deviceId
  - `lablog:experimentInfo` — pending ExperimentInfo form values (form-side only; cleared when copied into a draft)
  - `lablog:titleSequenceByDay` — `{ "YYMMDD": N }` map for `generateDraftTitle`
  - `lablog:ttsEnabled` — RecordPage TTS on/off toggle
  - `lablog:seen-drafts` — string[] of draft IDs already viewed (drives "new" badge state)
- **RecordPage flow** (the most complex page):
  - Two parallel WebSockets: `/ws/analyze` (JPEG frames @ `SEND_INTERVAL_MS`) and `/ws/transcribe` (6s WebM/Opus chunks from a separate MediaRecorder restarted every chunk).
  - **Pause/resume** (not on/off): uses MediaRecorder native `pause()` for batch recorder, full stop+restart for the realtime STT cycle. `pausedRef` guards against stale callbacks.
  - **Camera selector**: enumerates `navigator.mediaDevices.enumerateDevices()` after permission grant. Dropdown only renders when >1 videoinput available, disabled while recording. Persists to `lablog:preferredCameraId`.
  - **Aspect-ratio adaptive preview**: `videoAspect` state set via `onLoadedMetadata` and applied to the preview container, so 16:9 and 9:16 cameras both display without letterboxing.
  - **Assist overlay**: subscribes to `record.assist?: string[]` on each `/ws/analyze` message; banner latches for 4s after each receipt to avoid flicker.

### Data flow (upload path)

```
UploadPage  ─file→  LoadingPage
                      └→ uploadAndAnalyze() (fire-and-forget XHR)
                          POST /api/analyze/video
                            ├→ ThreadPool: extract_and_transcribe (parallel)
                            └→ cv2.VideoCapture loop @ sample_fps=1
                                 detect_objects → YOLO persistence filter
                                 → conditional detect_text_in_regions:
                                     • 7-seg YOLO on full frame (one call)
                                     • EasyOCR on each YOLO bbox crop (OTSU-binarized)
                                     • dedup by exact text match, join with " | "
                                 → OCR persistence filter (text must repeat ≥2/4)
                                 → dedup save on yolo_changed ∨ valid_ocr_changed
                          response.records → updateDraft({status:'complete', data})
AnalysisDetailPage (polling)
   ├→ "GRU로 단계 분류" → POST /api/classify → updateDraft({phases})
   └→ "Claude로 보고서 생성" (label predates Groq switch)
        → POST /api/report/generate (records + phases)
        → Groq parse(response_format=GeneratedReport)
        → updateDraft({report})
ReportDetailPage → renders GeneratedReportView (same table styles as mock view)
```

The RecordPage flow is analogous but kicks off audio transcription separately (`audio_pending` field) and the records come from the WebSocket stream instead of the HTTP response.

## Recurring foot-guns

- **Changing `PHASES` or `FEATURE_DIM`** invalidates `gru_weights.pt`. The size-mismatch error from `state_dict` is caught with a clear "run `python train_gru.py`" message — don't silently `try/except` around it. This also fires when swapping the SBERT model (different `SBERT_DIM`) or reverting to baseline (`use_sbert=False` requires a separately-trained 33-dim weights file — currently the production path always includes SBERT).
- **Only EasyOCR is YOLO-gated** — 7-segment YOLO runs on the full frame. So a stopwatch is captured even with zero `best.pt` detections, but a printed label like "NaOH 1M" on a YOLO-missed bottle is lost. Mitigation: broaden `best.pt` vocab via `train_main_yolo.py` (in progress on `chemistry-lab-object-detection-topas`). Do **not** reintroduce a full-frame EasyOCR fallback without discussing — the previous `OCR_FALLBACK_THRESHOLD` was deliberately removed.
- **`ocr` field is multi-text joined.** When several detections each yield text, results are concatenated with `" | "` into one string. The `vectorizer.py` digit-ratio / unit-pattern features still work because `re.search` is order-agnostic. Downstream consumers parsing `ocr` must split on `" | "`.
- **Groq model selection** for report generation: `parse()` requires strict json_schema → must use `openai/gpt-oss-*`. Other models 400. TPM limits differ (gpt-oss-120b: 8K TPM free tier; gpt-oss-20b ≈30K). Default is **gpt-oss-20b** (more TPM headroom). If you ever switch to 120b for quality and hit TPM caps on long videos: (a) ensure phases are passed (triggers `_format_records` compression), (b) further compact the formatter, or (c) drop to non-strict (`response_format={"type":"json_object"}` + manual `json.loads`).
- **Strict json_schema constraints**: all Pydantic fields must be required (no `Optional`); object types get `additionalProperties: false` automatically; streaming + tool use incompatible with strict structured output.
- **Whisper hallucination defense is layered — don't remove a layer in isolation.** Three independent defenses combine: (1) RMS VAD pre-filter (`is_speech_in_chunk` / `REALTIME_RMS_THRESHOLD`) stops silent chunks ever reaching Whisper; (2) `temperature=0.0` + regex repetition + sentence-repetition filters catch model-side hallucinations; (3) `_is_prompt_echo` + 3-strike `running_context` reset catches prompt feedback loops. Each tuning knob (VAD threshold, `REALTIME_CONTEXT_MAX_CHARS`, chunk size, `CONTEXT_RESET_AFTER`) was settled empirically after observed regressions — see the Tunable constants table for the trade-off on each before changing.
- **Windows Store Python venv**: `uvicorn` on PATH may resolve to global Store Python (different env, missing deps). Use `.\.venv\Scripts\python.exe -m uvicorn ...` to be safe.
- **VS Code Python interpreter**: IDE may complain "package not installed" against the wrong interpreter. Select via `Ctrl+Shift+P` → "Python: Select Interpreter" → `.venv\Scripts\python.exe`.
- **Persistent env vars not visible in existing PowerShell windows**: must open a new window OR pull manually via `$env:X = [Environment]::GetEnvironmentVariable('X', 'User')`.
- **Korean LF/CRLF warnings on Windows git**: harmless; ignore.
- **`backend/README.md` is stale** — it still documents a removed `camera_prototype.py` demo entry point, describes YOLO as unfine-tuned COCO 80-class, and says "STT / GRU / Claude API: 이번 범위 밖" (out of scope). All of that has since shipped. Don't trust it over this file or the actual source; it needs a rewrite but hasn't been prioritized.
- **No automated tests exist** (no pytest files, no `*.test.ts`/`*.spec.ts`, no vitest/jest config) anywhere in the repo. Verification is manual (`npm run build`/`lint`, manual backend smoke-testing via `/docs`, `evaluate_gru.py` for the ML pipeline).

## Operating constraints

- **UI ownership** (re-stated inline for portability across environments):
  - User-designed pages have **frozen visible markup/styles**: `HomePage`, `UploadPage`, `LoadingPage` (the ring is theirs), `RecordPage` (buttons/notes are theirs — added overlays like camera selector / assist banner came from explicit user request), `ArchivePage`, `DraftsPage`, the **mock-data view** of `ReportDetailPage`.
  - Modify internal logic (state, callbacks, effects, navigate, refs) freely; treat rendered JSX and CSS class additions as needing user consent.
  - Claude-authored pages where free modification is fine: `AnalysisDetailPage`, the `GeneratedReportView` block inside `ReportDetailPage`.
- **No fake implementations**: when a feature can't be honestly implemented (e.g., real progress without a backend signal), state so and ask. Do not substitute fake/placeholder values (random-progress bars, hardcoded mock data presented as real, decorative animations implying measurement). Honest empty/loading states are preferred to dishonest filled ones.
- **Korean responses** by default. The user is a Korean high-school student; collaborator-level Korean, code-heavy answers acceptable.

## Recently removed (don't reintroduce)

The following were intentionally deleted in a cleanup pass — re-adding them without discussing is a regression:

- **`backend/camera_prototype.py`** — standalone webcam demo. Server flow doesn't depend on it; the file's `AnalysisResult` constructor was also stale (missing `detections`/`frame_width`/`frame_height`/`speech` fields).
- **`backend/extract_frames.py`** — manual frame-sampling helper for Roboflow labeling. Roboflow Universe + the `train_main_yolo.py` Roboflow SDK path made it unused.
- **`backend/train_yolo_local.py`** and **`backend/train_yolo.ipynb`** — older training scripts superseded by `train_main_yolo.py` + [KAGGLE_TRAINING.md](backend/KAGGLE_TRAINING.md).
- **`backend/7SEGMENT_SETUP.md`** — described the abandoned single-digit CNN approach. The current YOLOv8 detection path is documented in KAGGLE_TRAINING.md.
- ~~**`backend/gcp-credentials.json`** + `google-cloud-speech` package — Google Cloud STT was replaced by Groq Whisper.~~ **(되돌림)** — 배치 STT를 Google STT로 다시 도입해 Groq TPM 한도 부담을 분산. 자격증명은 `GCP_CREDENTIALS_JSON` env(JSON 통째) 또는 `GOOGLE_APPLICATION_CREDENTIALS`(파일 경로)로 받음. 디스크의 `gcp-credentials.json`은 여전히 두지 않음 (git ignore).
- **`analyzer.detect_text()`** (full-frame OCR) and **`analyzer.needs_assist()`** — the only callers were `camera_prototype.py`. `detect_text_in_regions` is now the sole OCR path; `get_assist_messages` is the sole assist path.

## In-progress work

- **Main YOLO re-fine-tuning** to broaden vocabulary beyond the 25 lab-equipment classes (stopwatch was the first missing class the user noticed). Source: Roboflow project `s-workspace-qeozq/chemistry-lab-object-detection-topas` v1. The chosen strategy is **A: replace `best.pt` fresh-fine-tuned from `yolov8n.pt`** on this dataset only (not a merge with the original 25-class data — that source is unavailable). Local `train_main_yolo.py` runs on CPU (6+ hours); the user is offloading to Kaggle GPU per [backend/KAGGLE_TRAINING.md](backend/KAGGLE_TRAINING.md). When the new `best.pt` lands, **server must be restarted** — `LabLogAnalyzer.__init__` loads the model once at startup. The pre-replacement `best.pt` is auto-backed up to `best.backup_YYYYMMDD_HHMMSS.pt` by `train_main_yolo.py` (a `best.backup_20260603_202741.pt` is already present, indicating one round of replacement has occurred).
- **7-segment detector** is **active**: `seven_segment_weights.pt` exists alongside `best.pt`, so `seven_segment_classifier.is_available()` returns True and `detect_text_in_regions` runs the 7-seg YOLO on every full frame. Built on ultralytics YOLOv8 (not the originally-planned single-digit CNN — the chosen Roboflow project was object-detection not classification, which actually upgrades the design to support multi-digit displays). Original setup notes in [backend/7SEGMENT_SETUP.md](backend/7SEGMENT_SETUP.md) describe the now-abandoned CNN path — refer to [backend/KAGGLE_TRAINING.md](backend/KAGGLE_TRAINING.md) for the current YOLO training approach instead.

## Repository

Hosted at `https://github.com/idjun725/LabLog` (force-pushed). Commit author: `정현준 <idjun725@gmail.com>`. The user prefers **repo-local** git config over global — only set `user.name`/`user.email` in `.git/config`, not via `--global`. Root `.gitignore` belt-and-suspenders duplicates subfolder ignores; `.claude/` is gitignored; `chemistry_lab_data/`, `seven_segment_data/`, `runs/`, `*.pt`, `gcp-credentials.json` all blocked.
