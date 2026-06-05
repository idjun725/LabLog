# LabLog Backend

기획서 1단계(멀티모달 분석) 중 **영상 파이프라인**(YOLO 객체 탐지 + EasyOCR 텍스트 인식).
실행 모드는 두 가지입니다.

| 모드 | 진입점 | 용도 |
|---|---|---|
| 로컬 웹캠 데모 | `camera_prototype.py` | 백엔드 동작 검증, 카메라 미리보기 |
| API 서버 | `server.py` (uvicorn) | 프론트엔드와 연동 |

추론 로직은 [analyzer.py](analyzer.py)에 캡슐화되어 두 모드가 같은 코드를 공유합니다.

## 파일 구조

```
backend/
├── analyzer.py             # YOLO + EasyOCR 래퍼 (단일 진실 공급원)
├── camera_prototype.py     # 로컬 웹캠 데모
├── server.py               # FastAPI 서버
├── requirements.txt
├── README.md
├── .gitignore
├── logs/                   # 실행 시 자동 생성 (JSONL 분석 로그)
└── snapshots/              # 실행 시 자동 생성 (s 키 스냅샷)
```

## 설치

```powershell
cd c:\Coding\lablog\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### (선택) 음성 분석 — ffmpeg + Google Cloud STT

영상에서 음성을 추출해 Google Cloud Speech-to-Text로 전사하는 기능은 다음 두 가지가
설치/설정되어 있을 때만 동작합니다. 둘 다 없어도 시스템은 정상 동작하며, 단지
records의 `speech` 필드만 빈 문자열이 됩니다.

**1) ffmpeg (오디오 추출)**

```powershell
# scoop 사용자
scoop install ffmpeg

# chocolatey 사용자
choco install ffmpeg

# 또는 https://www.gyan.dev/ffmpeg/builds/ 에서 "release essentials" 다운로드 후
# 압축 풀어 시스템 환경변수 Path에 추가
```
확인: `ffmpeg -version`

**2) Google Cloud STT 자격증명**

1. <https://console.cloud.google.com> 접속, 새 프로젝트 생성 (무료 크레딧 $300 + STT 월 60분 무료)
2. APIs & Services → Library → "Cloud Speech-to-Text API" 검색 → **Enable**
3. IAM & Admin → Service Accounts → **Create Service Account**, Role: `Cloud Speech Client`
4. 만들어진 서비스 계정 → Keys → Add Key → Create new key → JSON → 다운로드
5. JSON을 `backend/gcp-credentials.json`으로 저장 (이미 .gitignore에 포함됨)
6. 백엔드 띄울 때마다 환경변수 지정:
   ```powershell
   $env:GOOGLE_APPLICATION_CREDENTIALS = "c:\Coding\lablog\backend\gcp-credentials.json"
   uvicorn server:app --reload --port 8000
   ```

자격증명이 없을 때의 동작: 콘솔에 `[stt] GOOGLE_APPLICATION_CREDENTIALS 미설정 — 건너뜀` 한 줄만 찍히고 분석은 계속됩니다.

## 실행 — 로컬 웹캠 데모

```powershell
python camera_prototype.py
```

- `q` : 종료
- `s` : 현재 프레임 스냅샷 저장 (`snapshots/`)

## 실행 — API 서버

```powershell
uvicorn server:app --reload --port 8000
```

서버가 뜨면 자동 문서에서 모든 엔드포인트를 확인할 수 있습니다:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

## API 엔드포인트

### `GET /api/health`
모델 로드 여부 확인.
```json
{"status": "ok", "model_loaded": true}
```

### `POST /api/analyze/video`
업로드된 영상 파일을 일괄 분석. 프론트엔드의 `/upload` 페이지에서 사용.

- `file` (multipart) : 영상 파일
- `sample_fps` (query, 기본 1.0) : 초당 분석 프레임 수

응답:
```json
{
  "filename": "experiment.mp4",
  "frame_count": 42,
  "records": [
    {
      "timestamp": "00:00:01",
      "yolo": {"bottle": 0.91},
      "ocr": "4.0g",
      "ocr_confidence": 0.87,
      "avg_yolo_confidence": 0.78,
      "brightness": 132.4
    }
  ]
}
```

### `WebSocket /ws/analyze`
실시간 프레임 스트림 분석. 프론트엔드의 `/record` 페이지에서 사용.

- 클라이언트 → 서버 : 바이너리 메시지 (JPEG 인코딩 프레임)
- 서버 → 클라이언트 : JSON 메시지 (위 records와 동일 스키마 + 선택적 `assist` 필드)

`assist` 필드는 평균 신뢰도가 낮거나 화면이 어두워 시야 확보가 필요할 때만 포함됩니다 (기획서의 "실시간 영상 보조 기능").

## 프론트엔드 연동 예시

영상 업로드:
```ts
const form = new FormData()
form.append('file', file)
const res = await fetch('http://localhost:8000/api/analyze/video?sample_fps=1', {
  method: 'POST',
  body: form,
})
const { records } = await res.json()
```

실시간 스트림:
```ts
const ws = new WebSocket('ws://localhost:8000/ws/analyze')
ws.onmessage = (e) => {
  const record = JSON.parse(e.data)
  if (record.assist) showAssistMessage(record.assist)
}
// MediaRecorder/Canvas로 캡처한 JPEG 바이트를 ws.send(jpegBytes)로 전송
```

## 알려진 한계 및 다음 단계

- **YOLO 클래스**: 현재 `yolov8n.pt`는 COCO 80개 일반 클래스. 실험 기자재(비커, 전자저울 등) 탐지는 직접 라벨링한 데이터셋으로 파인튜닝 필요. 파인튜닝 후 [analyzer.py](analyzer.py)의 `DEFAULT_YOLO_WEIGHTS`만 `best.pt` 경로로 변경하면 됩니다. **API 계약은 그대로**이므로 프론트엔드 변경 불필요.
- **OCR 주기**: EasyOCR가 무거워 30프레임마다 실행합니다 (`camera_prototype.py`). WebSocket 분석은 클라이언트가 보내는 속도에 따라 결정됩니다.
- **STT / GRU / Claude API**: 이번 범위 밖. JSONL 스키마가 다음 단계의 입력이 됩니다.
- **한글 오버레이**: `cv2.putText`가 한글을 렌더링하지 못해 화면은 영문, 콘솔/JSON은 한글로 분리.
