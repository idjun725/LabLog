# LabLog AI Coding Agent Guide

## Purpose
This file helps AI coding agents work productively in the LabLog repository.
It points to the main docs, explains the architecture, and highlights project-specific conventions and pitfalls.

## Core docs
- `CLAUDE.md` — primary design and implementation guidance for this project.
- `backend/README.md` — backend setup, API contract, and optional STT configuration.
- `frontend/README.md` — frontend tooling and Vite/React notes.

## Build / run commands
### Frontend
From `frontend/`:
- `npm run dev` — start Vite dev server
- `npm run build` — compile TypeScript and produce `dist/`
- `npm run lint` — run ESLint
- `npm run preview` — preview built app

### Backend
From `backend/`:
- `python -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`
- `pip install -r requirements.txt`
- `.\.venv\Scripts\python.exe -m uvicorn server:app --reload --port 8000`
- `.\.venv\Scripts\python.exe train_gru.py`

## Architecture overview
- Backend is a flat FastAPI app in `backend/`.
- Frontend is a Vite + React app in `frontend/`.
- There is no database; drafts are stored in browser `localStorage`.
- `src/api/lablog.ts` is the single frontend HTTP/WS integration point. Other frontend code should not call `fetch` directly.

## Key backend boundaries
- `analyzer.py` contains the canonical analysis logic.
- `server.py` exposes HTTP and WebSocket endpoints.
- `vectorizer.py`, `gru_model.py`, `gru_classifier.py`, and `report_generator.py` form the ML pipeline.
- `stt.py` depends on optional `ffmpeg` and Google Cloud STT.

## Important conventions and cautions
- Do not change `backend/gru_model.py` PHASES order or `backend/vectorizer.py` `YOLO_VOCAB` feature order without retraining `gru_weights.pt`.
- `report_generator.py` uses Groq strict `json_schema`; only `openai/gpt-oss-20b` or `openai/gpt-oss-120b` are valid models.
- `GOOGLE_APPLICATION_CREDENTIALS` and `GROQ_API_KEY` are environment variables. `GROQ_API_KEY` is required for `/api/report/generate`.
- The realtime WebSocket path `/ws/analyze` does not apply the batch persistence filter or STT; only HTTP batch analysis does.
- Avoid fake/hardcoded UI behavior. If backend status is unknown, preserve honest loading and error states.

## Frontend ownership guidelines
- UI markup/style is stable for `HomePage`, `UploadPage`, `LoadingPage`, `RecordPage`, `ArchivePage`, `DraftsPage`, and the mock view of `ReportDetailPage`.
- `AnalysisDetailPage` and the generated report view inside `ReportDetailPage` are safe to modify more freely.

## Recommended behavior for agents
- Prefer small, local changes over broad refactors.
- Preserve existing user-visible UI structure unless the user explicitly asks for visual changes.
- Use the project docs (`CLAUDE.md`, `backend/README.md`, `frontend/README.md`) rather than repeating detailed architecture.
- If a requested feature depends on external services (Groq, STT, ffmpeg), call it out explicitly.

## Useful paths
- `backend/server.py`
- `backend/analyzer.py`
- `backend/vectorizer.py`
- `backend/gru_model.py`
- `backend/report_generator.py`
- `frontend/src/api/lablog.ts`
- `frontend/src/pages/RecordPage.tsx`
- `frontend/src/pages/AnalysisDetailPage.tsx`
- `frontend/src/pages/ReportDetailPage.tsx`

## AI 에이전트 헬퍼 파일
- 리포지토리에서 AI 에이전트의 빠른 진입을 돕기 위한 파일:
	- [ .github/copilot-instructions.md ](.github/copilot-instructions.md) — 빠른 실행 커맨드, 핵심 제약, 에이전트 행동 원칙을 요약합니다.
	- [CLAUDE.md](CLAUDE.md) — 상세 아키텍처·운영 규칙(권장 참조).
