# GitHub Copilot / AI Agent 안내 (간략)

이 파일은 리포지토리에서 AI 코딩 에이전트가 빠르게 생산성 있게 작업하도록 돕기 위한 요약 가이드입니다. 자세한 설계 및 운영 규칙은 [CLAUDE.md](CLAUDE.md)와 [AGENTS.md](AGENTS.md)를 참고하세요.

핵심 요약
- **빠른 실행**: 프론트엔드: `cd frontend && npm run dev`. 백엔드: `cd backend && .\.venv\Scripts\python.exe -m uvicorn server:app --reload --port 8000`.
- **주요 파일(변경 전 주의)**: [backend/gru_model.py](backend/gru_model.py), [backend/vectorizer.py](backend/vectorizer.py) — 이들의 상수/ordering 변경은 `gru_weights.pt` 재학습을 요구합니다.
- **프론트엔드 통합 지점**: 모든 HTTP/WS 통신은 `frontend/src/api/lablog.ts`를 통해 이뤄집니다. 다른 파일에서 직접 `fetch` 사용 금지.
- **UI 제약**: `HomePage`, `UploadPage`, `LoadingPage`, `RecordPage`, `ArchivePage`, `DraftsPage`, 및 `ReportDetailPage`의 모의(mock) 뷰는 시각적 마크업/스타일을 변경하지 마세요. 내부 로직(상태/콜백)은 요청 시 자유롭게 수정 가능합니다(자세한 내용은 [AGENTS.md](AGENTS.md)를 확인).
- **외부 서비스 주의**: 보고서 생성은 `GROQ_API_KEY`, 배치 STT는 `GCP_CREDENTIALS_JSON` 또는 `GOOGLE_APPLICATION_CREDENTIALS` 등이 필요합니다. 변경 전 관련 환경 변수와 부작용을 명시하세요.

에이전트 행동 원칙(간단)
- 작은, 국소적 변경을 선호하세요 — 대대적 리팩토링은 사용자 승인 필요.
- 모델/가중치 관련 파일을 건드리기 전에는 반드시 재학습(retrain) 필요 여부를 확인하고 사용자에게 알리세요.
- 문서가 이미 존재하면 복사하지 말고 링크로 참조하세요 (link, don't embed).

추가 리소스
- 전체 정책/아키텍처: [CLAUDE.md](CLAUDE.md)
- 에이전트 전용 가이드(더 상세): [AGENTS.md](AGENTS.md)
