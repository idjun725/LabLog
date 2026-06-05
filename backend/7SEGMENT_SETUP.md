# 7세그먼트 분류기 설정 가이드 (Roboflow + PyTorch)

## 1단계: Roboflow 공개 데이터셋 찾기

### A. Roboflow Universe 접속
- https://universe.roboflow.com 방문
- 검색어: `7-segment`, `digital display`, `seven segment`
- 추천 데이터셋:
  - **"7-Segment Display Digit Recognition"** (가장 인기)
  - **"MNIST 7-Segment"**
  - 최소 500장 이상의 라벨링된 이미지 있는지 확인

### B. 데이터셋 선택 기준
- ✓ 라벨: 0-9 숫자 분류 (10 클래스)
- ✓ 형식: 이미지 분류 (object detection 아님)
- ✓ 라이선스: 다운로드 가능 (Commercial use OK)
- ✓ 이미지 수: 최소 500장 이상 (train + val 나뉨)

---

## 2단계: Roboflow에서 모델 학습 (웹 기반)

### A. 데이터셋 Import
1. 선택한 데이터셋 페이지에서 **"Clone this Dataset"** 클릭
2. Roboflow 계정 로그인 (없으면 가입)
3. **"Clone"** → 자신의 workspace로 가져오기

### B. 데이터 전처리
1. "Preprocessing" 탭:
   - Auto-Orient: ✓ 체크
   - Resize to: 224x224 또는 128x128
   - Grayscale: ✓ 체크 (선택, 7세그는 흑백 데이터)

2. "Augmentation" 탭:
   - Rotation: ±15°
   - Brightness: ±20%
   - Flip: Vertical (선택)

### C. 모델 학습 (Roboflow 자체 학습기)
1. **"Train"** 탭 클릭
2. 모델 선택:
   - **YOLOv8 Classification** (권장) — 빠르고 정확
   - 또는 **PyTorch ResNet50**
3. **"Start Training"** 클릭 (무료 GPU, ~10-20분)

### D. 모델 평가
- 학습 완료 후 **"Results"** 탭에서:
  - Accuracy 확인 (90% 이상 목표)
  - Confusion Matrix 검토
  - 실패 케이스 분석

---

## 3단계: 모델 다운로드 (PyTorch 형식)

### A. Roboflow 웹 인터페이스에서
1. 학습 완료 후 **"Versions"** 탭
2. 최신 버전 선택
3. **"Export"** 버튼 클릭

### B. 내보내기 포맷 선택
- **"Roboflow Inference Server"** → PyTorch 클릭
- 또는 **"PyTorch"** 직접 선택

### C. 다운로드
```
# Option 1: Roboflow API 다운로드
pip install roboflow
python << 'EOF'
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("project-name")
version = project.version(1)
dataset = version.download("pytorch")
EOF
```

또는

```
# Option 2: 웹에서 직접 다운로드
# Roboflow 페이지의 "Export" → "PyTorch" → 다운로드 링크
```

---

## 4단계: LabLog 백엔드에 통합

### A. 모델 파일 배치
다운로드한 `.pt` 파일을 다음 위치에 저장:
```
backend/
  ├── 7segment_weights.pt     ← 다운로드한 가중치
  ├── 7segment_model.py       ← 모델 정의 (LabLog 생성)
  ├── 7segment_classifier.py  ← inference 래퍼 (LabLog 생성)
  └── train_7segment.py       ← fine-tuning 스크립트 (선택)
```

### B. 모델 정의 (`7segment_model.py`)
```python
import torch
import torch.nn as nn

DIGITS = [str(i) for i in range(10)]  # ["0", "1", ..., "9"]
NUM_DIGITS = len(DIGITS)

# Roboflow YOLOv8 Classification은 자동 export됨
# 이 파일은 호환성 래퍼만 제공
```

### C. Inference 래퍼 (`7segment_classifier.py`)
```python
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms

WEIGHTS_PATH = Path(__file__).parent / "7segment_weights.pt"
DIGITS = [str(i) for i in range(10)]

_model = None

def _load_model():
    global _model
    if _model is not None:
        return _model
    if not WEIGHTS_PATH.exists():
        raise RuntimeError(f"7segment 가중치 없음: {WEIGHTS_PATH}")
    _model = torch.hub.load('ultralytics/yolov8', 'custom', path=str(WEIGHTS_PATH), force_reload=False)
    _model.eval()
    return _model

def classify_7segment(image_array) -> str:
    """numpy array (crop) → 0-9 숫자 문자열"""
    model = _load_model()
    img = Image.fromarray(image_array)
    results = model(img)
    pred_class = results[0].probs.top1
    return DIGITS[pred_class]
```

### D. analyzer.py에 통합
```python
# analyzer.py에서:
from 7segment_classifier import classify_7segment

def _is_7segment_candidate(text: str) -> bool:
    return bool(re.match(r"^\d{1,2}$", text))

# detect_text_in_regions 내에서:
if _is_7segment_candidate(ocr_text):
    try:
        digit = classify_7segment(crop)  # numpy crop 전달
        return digit, 1.0  # 7segment 모델 신뢰도 = 1.0
    except Exception:
        return ocr_text, ocr_conf  # fallback to EasyOCR
```

---

## 5단계: 테스트

```bash
cd backend
.\.venv\Scripts\python.exe -c "from 7segment_classifier import classify_7segment; print('OK')"

# 테스트 영상 업로드 후:
# [analyze] 7-segment candidate detected: "8" → 7segment classifier
```

---

## Roboflow API Key 설정 (선택)

Roboflow에서 직접 추론하려면:
```powershell
$env:ROBOFLOW_API_KEY = "YOUR_API_KEY"
```

하지만 **로컬 .pt 모델 사용**을 권장 (더 빠름, 클라우드 의존 없음).

---

## 문제 해결

| 문제 | 해결 |
|---|---|
| `FileNotFoundError: 7segment_weights.pt` | 모델 파일을 `backend/` 폴더에 저장했는지 확인 |
| `ImportError: No module named 'ultralytics'` | pip install -r requirements.txt (이미 있음) |
| 낮은 정확도 (<80%) | Roboflow에서 더 많은 데이터로 재학습하거나, 기존 EasyOCR fallback 사용 |

---

## 다음 단계

1. ✅ Roboflow 데이터셋 선택 + 모델 학습 (웹에서 완료)
2. ✅ 모델 다운로드 (.pt 파일)
3. ⏳ LabLog 통합 코드 작성 (Claude가 도와드림)
4. ⏳ 테스트

준비되면 알려주세요!
