# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LabLog is a multimodal AI system that records / uploads a science experiment video and produces a structured Korean high-school lab report. The pipeline:

1. **Video frame analysis** — Three text/object models with explicit role separation:
   - **YOLO object detection** (`best.pt`, fine-tuned 25 lab equipment classes, 2026-05; retrain in progress on `chemistry-lab-object-detection-topas` to broaden coverage incl. stopwatches etc.).
   - **EasyOCR** for general labels/measurements — runs **only on YOLO bbox crops** (each OTSU-binarized for accuracy).
   - **7-segment YOLOv8 detector** (`seven_segment_weights.pt`, lazy-loaded if present) for digital displays — runs **on the full frame** (fast, catches digits regardless of whether the device is in `best.pt` vocab).
   - Two temporal-consistency filters reject "flashing" detections: one for YOLO labels, one for OCR text — both `PERSISTENCE_WINDOW=4` / `PERSISTENCE_MIN_VOTES=2`. **Trade-off**: text on un-detected non-digit objects (e.g., a chemical bottle label YOLO misses) is lost — the user accepted this in exchange for noise reduction, with the plan to broaden `best.pt` vocab.
2. **Audio analysis** — Google Cloud STT (post-hoc batch transcription via ffmpeg → wav → 55s chunks). Realtime streaming STT is **not** implemented; RecordPage records audio with MediaRecorder and uploads on stop.
3. **Vector conversion** — multimodal `records` → 23-dim feature vectors (15 YOLO vocab one-hot + 6 OCR + 2 STT).
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

# Train the GRU stage classifier (~30s on CPU, writes gru_weights.pt)
.\.venv\Scripts\python.exe train_gru.py

# Train the 7-segment YOLOv8 detector (CPU OK for small dataset, writes seven_segment_weights.pt)
.\.venv\Scripts\python.exe train_seven_segment.py

# Train/refresh the MAIN YOLO from Roboflow (CPU = hours; offload to Kaggle GPU recommended).
# Backs up existing best.pt to best.backup_YYYYMMDD_HHMMSS.pt before overwriting.
.\.venv\Scripts\python.exe train_main_yolo.py

# Standalone webcam loop (no server)
.\.venv\Scripts\python.exe camera_prototype.py
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

`GOOGLE_APPLICATION_CREDENTIALS` and ffmpeg are both **optional** — STT degrades silently if either is absent. `GROQ_API_KEY` is **required** for `/api/report/generate` (returns 503 otherwise). `ROBOFLOW_API_KEY` is only used by the training scripts (`train_seven_segment.py`, `train_main_yolo.py`) — never read at runtime.

## Architecture

### Surface — endpoints and routes

**Backend FastAPI** (`backend/server.py`):

| Verb | Path | Purpose |
|---|---|---|
| GET  | `/api/health` | model_loaded check |
| POST | `/api/analyze/video` | upload video → records (HTTP batch) |
| POST | `/api/classify` | records → GRU phase labels |
| POST | `/api/record/transcribe` | webm audio + records → records w/ speech filled |
| POST | `/api/report/generate` | records (+ optional phases) → GeneratedReport via Groq |
| WS   | `/ws/analyze` | binary JPEG frames → JSON AnalysisRecord per frame |

CORS allows `localhost`/`127.0.0.1` on any port via regex (dev convenience).

**Frontend routes** (`src/App.tsx`):
`/`, `/upload`, `/record`, `/archive`, `/drafts`, `/loading`, `/analysis/:id`, `/report/:id`, `*` → redirect to `/`.

### Backend module boundaries (intentionally flat — no package)

- **`analyzer.py`** — `AnalysisResult` dataclass is the canonical record schema (its fields appear unchanged in WS/HTTP JSON). `analyze_video()` runs STT in a `ThreadPoolExecutor` parallel to the frame loop and merges segments by timestamp afterwards. Implements:
  - **YOLO temporal-consistency filter**: `PERSISTENCE_WINDOW=4` / `PERSISTENCE_MIN_VOTES=2`. A label must appear in ≥2 of last 4 samples **and** be present in the current frame to count.
  - **OCR temporal-consistency filter**: same `PERSISTENCE_WINDOW`/`MIN_VOTES` as YOLO, over a `recent_ocr_texts` deque of joined OCR strings; text must repeat ≥2 times before a record is saved on `ocr_changed`.
  - **`detect_text_in_regions(frame, detections)`** — the production OCR pipeline. Runs **two models with explicit roles**:
    1. **7-segment YOLO on the FULL frame** (single call, fast) via `seven_segment_classifier.detect_digits()`. Catches digital displays regardless of whether the device is in `best.pt` vocab.
    2. **EasyOCR on each YOLO bbox crop** (OTSU-binarized first). Catches printed labels/measurements on detected equipment.
    - Results from both sources are deduped by exact text match, then joined with `OCR_TEXT_SEPARATOR` (`" | "`) into the single `ocr` field.
    - **No full-frame EasyOCR fallback** — the user removed it after switching to full-frame seg7, accepting the trade-off that text on undetected non-digit objects is missed.
  - **OCR noise filter (`_filter_ocr_raw`)**, shared by `detect_text` and `detect_text_in_regions`: `OCR_MIN_CONFIDENCE=0.5`, `OCR_MIN_LENGTH=2`, `OCR_MIN_DIGIT_RATIO=0.25`, plus `NUMERIC_PATTERN` (must contain a digit).
  - The original full-frame `detect_text()` method is still defined and **still called by `camera_prototype.py`** — do not delete.
- **`seven_segment_classifier.py`** — Lazy-loading wrapper around an ultralytics `YOLO(seven_segment_weights.pt)`. `is_available()` is a cheap path-existence check safe to call per-frame; `detect_digits(crop, conf_threshold)` returns `(joined_digits_string, avg_confidence) | None` with digits sorted left-to-right by bbox x-coord. Grayscale input is auto-converted to BGR (YOLO requires 3 channels). `_load_failed` flag prevents repeated load-failure attempts. If the weights file is absent, `is_available()` returns False and `detect_text_in_regions` skips the seg7 call entirely — graceful degradation.
- **`train_seven_segment.py`** / **`train_main_yolo.py`** — CLI training scripts following the same pattern: read `ROBOFLOW_API_KEY`, download the named Roboflow project/version as `yolov8` format (reused if `data.yaml` already exists), train via `ultralytics.YOLO.train()`, copy the resulting `runs/<name>/weights/best.pt` to the canonical location (`seven_segment_weights.pt` / `best.pt`). `train_main_yolo.py` additionally **backs up the existing `best.pt`** to `best.backup_YYYYMMDD_HHMMSS.pt` before overwriting — do not skip this when modifying the script.
- **`stt.py`** — `extract_and_transcribe()` is the unified helper that runs ffmpeg → wav → Google STT chunks (55s each). On ffmpeg failure the **last 5 lines** of stderr are shown (not the head, since ffmpeg's banner fills the first 1KB). `find_speech_at(t_sec, segments)` does the timestamp matching.
- **`vectorizer.py`** — `YOLO_VOCAB` ordering is the feature-index order. **Changing it requires retraining the GRU** (state_dict shape changes).
- **`gru_model.py`** — `PHASES` list order = output class index. **Changing it requires retraining** (`gru_weights.pt` becomes incompatible; `gru_classifier.py` catches state_dict mismatch and surfaces a retrain prompt).
- **`gru_classifier.py`** — lazy-loads weights into a module-level `_model` singleton.
- **`report_generator.py`** — Groq via `openai>=1.50` SDK with `base_url="https://api.groq.com/openai/v1"`. **Only `openai/gpt-oss-20b` and `openai/gpt-oss-120b` support strict `json_schema`** (required by `client.beta.chat.completions.parse(response_format=PydanticModel)`). Other Groq models (llama, qwen) return 400. `generate_report(records, filename, phases=None)` — when `phases` provided, `_format_records` groups consecutive same-phase records into one block per phase and dedupes objects/OCR/speech texts within the block (token compression for long videos). When `phases=None`, it falls back to one line per record. The Groq `parse()` returns the parsed Pydantic instance on `response.choices[0].message.parsed`; `None` triggers a `RuntimeError` (no auto-fallback).
- **`server.py`** — FastAPI endpoints. The realtime WebSocket (`/ws/analyze`) calls `analyze_frame()` per-frame and does **not** apply the persistence filter or STT (those need batch context), but **does** use cropping-based OCR (same `detect_text_in_regions` as batch). `get_assist_messages()` is called per WS frame and the resulting `string[]` is attached to the payload as `assist?` (for RecordPage's overlay banner).

### Tunable constants (where each lives)

| File | Constant | Effect |
|---|---|---|
| `analyzer.py` | `OCR_MIN_CONFIDENCE` (0.5), `OCR_MIN_LENGTH` (2), `OCR_MIN_DIGIT_RATIO` (0.25), `NUMERIC_PATTERN` | OCR noise filter — raise to be more aggressive |
| `analyzer.py` | `PERSISTENCE_WINDOW` (4), `PERSISTENCE_MIN_VOTES` (2) | YOLO **and** OCR temporal-consistency (same constants) — raise to filter more flicker |
| `analyzer.py` | `get_assist_messages()` thresholds | YOLO 0.4, OCR 0.65, brightness 40 |
| `analyzer.py` | `DEFAULT_YOLO_WEIGHTS` (`"best.pt"`) | Main YOLO weights file. Drop in `yolov8n.pt` for COCO classes, or a new `best.pt` after re-fine-tuning. API contract unchanged. |
| `analyzer.py` | `SEG7_MIN_CONFIDENCE` (0.5) | Confidence floor for the 7-segment detector. Raise to reduce false positives from non-digit textures. |
| `analyzer.py` | `OCR_TEXT_SEPARATOR` (`" \| "`) | Join string for multi-source OCR results. Downstream consumers must split on this exact value. |
| `train_main_yolo.py` | `ROBOFLOW_*`, `EPOCHS` (50), `IMGSZ` (640), `PATIENCE` (15) | Main YOLO fine-tuning. Edit constants at the top to retarget a new Roboflow project/version. |
| `vectorizer.py` | `YOLO_VOCAB`, `FEATURE_DIM` | Feature space — changing requires GRU retrain |
| `gru_model.py` | `PHASES` | Output classes — changing requires retrain |
| `report_generator.py` | `MODEL_NAME` | Groq model — only `openai/gpt-oss-{20b,120b}` support strict json_schema |
| `frontend/src/pages/LoadingPage.tsx` | `sampleFps: 1` (in `uploadAndAnalyze` call) | Backend frame sample rate |

### Frontend integration patterns

- **`src/api/lablog.ts` is the single integration point.** All other files import types and helpers from here; no other file does `fetch`.
- **Background analysis pattern**: `uploadAndAnalyze()` is fire-and-forget — creates a `pending` draft when XHR bytes finish uploading, lets server processing continue detached from React lifecycle, updates draft via `xhr.onload`. AnalysisDetailPage polls every 2s while `status === 'pending'` or `audio_pending === true`.
- **Routing split** (don't conflate):
  - `/analysis/:id` — raw record cards (real draft data only)
  - `/report/:id` — checks `getDraft(id)?.report` first; falls back to hardcoded `MOCK_REPORTS` (IDs `1`-`5`, `archive-1` … `archive-6`)
- **localStorage keys**:
  - `lablog:drafts` — Draft[]
  - `lablog:hidden-mocks` — string[] of mock IDs the user has "deleted"
  - `lablog:preferredCameraId` — RecordPage's last-used camera deviceId
- **RecordPage camera selector**: enumerates `navigator.mediaDevices.enumerateDevices()` after permission grant. Dropdown only renders when >1 videoinput available, disabled while recording. Selection persists to `lablog:preferredCameraId`.
- **Assist overlay**: RecordPage subscribes to `record.assist?: string[]` on each WS message; banner latches for 4s after each receipt to avoid flicker.

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

- **Changing `PHASES` or `FEATURE_DIM`** invalidates `gru_weights.pt`. The size-mismatch error from `state_dict` is caught with a clear "run `python train_gru.py`" message — don't silently `try/except` around it.
- **Only EasyOCR is YOLO-gated** — 7-segment YOLO runs on the full frame. So a stopwatch is captured even with zero `best.pt` detections, but a printed label like "NaOH 1M" on a YOLO-missed bottle is lost. Mitigation: broaden `best.pt` vocab via `train_main_yolo.py` (in progress on `chemistry-lab-object-detection-topas`). Do **not** reintroduce a full-frame EasyOCR fallback without discussing — the previous `OCR_FALLBACK_THRESHOLD` was deliberately removed.
- **Don't delete `detect_text()`** in `analyzer.py`. It's no longer called by `analyze_frame`/`analyze_video` but `camera_prototype.py` still uses it for the no-YOLO-context full-frame OCR case.
- **`ocr` field is multi-text joined.** When several detections each yield text, results are concatenated with `" | "` into one string. The `vectorizer.py` digit-ratio / unit-pattern features still work because `re.search` is order-agnostic. Downstream consumers parsing `ocr` must split on `" | "`.
- **Groq model selection** for report generation: `parse()` requires strict json_schema → must use `openai/gpt-oss-*`. Other models 400. TPM limits differ (gpt-oss-120b: 8K TPM free tier; gpt-oss-20b ≈30K). Default is **gpt-oss-20b** (more TPM headroom). If you ever switch to 120b for quality and hit TPM caps on long videos: (a) ensure phases are passed (triggers `_format_records` compression), (b) further compact the formatter, or (c) drop to non-strict (`response_format={"type":"json_object"}` + manual `json.loads`).
- **Strict json_schema constraints**: all Pydantic fields must be required (no `Optional`); object types get `additionalProperties: false` automatically; streaming + tool use incompatible with strict structured output.
- **Windows Store Python venv**: `uvicorn` on PATH may resolve to global Store Python (different env, missing deps). Use `.\.venv\Scripts\python.exe -m uvicorn ...` to be safe.
- **VS Code Python interpreter**: IDE may complain "package not installed" against the wrong interpreter. Select via `Ctrl+Shift+P` → "Python: Select Interpreter" → `.venv\Scripts\python.exe`.
- **Persistent env vars not visible in existing PowerShell windows**: must open a new window OR pull manually via `$env:X = [Environment]::GetEnvironmentVariable('X', 'User')`.
- **Korean LF/CRLF warnings on Windows git**: harmless; ignore.

## Operating constraints

- **UI ownership** (re-stated inline for portability across environments):
  - User-designed pages have **frozen visible markup/styles**: `HomePage`, `UploadPage`, `LoadingPage` (the ring is theirs), `RecordPage` (buttons/notes are theirs — added overlays like camera selector / assist banner came from explicit user request), `ArchivePage`, `DraftsPage`, the **mock-data view** of `ReportDetailPage`.
  - Modify internal logic (state, callbacks, effects, navigate, refs) freely; treat rendered JSX and CSS class additions as needing user consent.
  - Claude-authored pages where free modification is fine: `AnalysisDetailPage`, the `GeneratedReportView` block inside `ReportDetailPage`.
- **No fake implementations**: when a feature can't be honestly implemented (e.g., real progress without a backend signal), state so and ask. Do not substitute fake/placeholder values (random-progress bars, hardcoded mock data presented as real, decorative animations implying measurement). Honest empty/loading states are preferred to dishonest filled ones.
- **Korean responses** by default. The user is a Korean high-school student; collaborator-level Korean, code-heavy answers acceptable.

## In-progress work

- **Main YOLO re-fine-tuning** to broaden vocabulary beyond the 25 lab-equipment classes (stopwatch was the first missing class the user noticed). Source: Roboflow project `s-workspace-qeozq/chemistry-lab-object-detection-topas` v1. The chosen strategy is **A: replace `best.pt` fresh-fine-tuned from `yolov8n.pt`** on this dataset only (not a merge with the original 25-class data — that source is unavailable). Local `train_main_yolo.py` runs on CPU (6+ hours); the user is offloading to Kaggle GPU per [backend/KAGGLE_TRAINING.md](backend/KAGGLE_TRAINING.md). When the new `best.pt` lands, **server must be restarted** — `LabLogAnalyzer.__init__` loads the model once at startup. The pre-replacement `best.pt` is auto-backed up to `best.backup_YYYYMMDD_HHMMSS.pt` by `train_main_yolo.py`.
- **7-segment detector** is **already implemented** (`seven_segment_classifier.py`, `train_seven_segment.py`) using ultralytics YOLOv8 (not the originally-planned single-digit CNN — the chosen Roboflow project was object-detection not classification, which actually upgrades the design to support multi-digit displays). Inactive until `seven_segment_weights.pt` exists (graceful no-op). Original setup notes in [backend/7SEGMENT_SETUP.md](backend/7SEGMENT_SETUP.md) describe the now-abandoned CNN path — refer to [backend/KAGGLE_TRAINING.md](backend/KAGGLE_TRAINING.md) for the current YOLO training approach instead.

## Repository

Hosted at `https://github.com/idjun725/LabLog` (force-pushed 2026-05-14 to replace a different LabLog MVP that lived on `main`). Commit author: `정현준 <idjun725@gmail.com>` via global git config. Root `.gitignore` belt-and-suspenders duplicates subfolder ignores; `.claude/` is gitignored.
