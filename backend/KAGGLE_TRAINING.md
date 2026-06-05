# Kaggle Notebooks로 YOLO 학습하기

CPU로 6시간 걸리는 학습이 Kaggle 무료 GPU(P100/T4)에서는 **20-40분**으로 줄어듭니다.

---

## Step 1: Kaggle 계정 + 휴대폰 인증 (최초 1회)

1. https://www.kaggle.com 접속 → 우측 상단 "Register" → Google 로그인 권장
2. **GPU 사용은 휴대폰 인증 필요**:
   - 우측 상단 프로필 → "Settings"
   - "Phone Verification" 섹션에서 휴대폰 인증
   - 인증 후 주 30시간 GPU 사용 가능

---

## Step 2: 새 Notebook 만들기

1. 좌측 메뉴에서 **"Code"** 클릭 → **"+ New Notebook"**
2. 우측 패널 **"Settings"** 펼치기:
   - **Accelerator**: `GPU P100` 선택 (T4도 OK, P100이 약간 더 빠름)
   - **Internet**: `On` (Roboflow API 호출에 필요)
   - **Persistence**: `No` (이번엔 일회성)

---

## Step 3: Roboflow API key를 Kaggle Secret으로 저장

코드에 API key를 하드코딩하지 않기 위해 Kaggle Secrets 사용:

1. Notebook 상단 메뉴 → **"Add-ons"** → **"Secrets"**
2. **"Add a new secret"** 클릭
3. 입력:
   - **Label**: `ROBOFLOW_API_KEY`
   - **Value**: 본인의 Roboflow API key
4. **Save** → 옆의 토글을 켜서 활성화

---

## Step 4: 학습 코드 (셀별로 복사해서 붙여넣기)

### 셀 1: 패키지 설치

```python
!pip install -q ultralytics roboflow
```

### 셀 2: API key 로드 + 환경 확인

```python
from kaggle_secrets import UserSecretsClient
import torch

user_secrets = UserSecretsClient()
ROBOFLOW_API_KEY = user_secrets.get_secret("ROBOFLOW_API_KEY")

print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음")
```

> ⚠️ "CUDA 사용 가능: False"가 나오면 Settings에서 Accelerator를 GPU로 다시 설정.

### 셀 3: Roboflow 데이터셋 다운로드

```python
from roboflow import Roboflow

ROBOFLOW_WORKSPACE = "s-workspace-qeozq"
ROBOFLOW_PROJECT = "chemistry-lab-object-detection-topas"
ROBOFLOW_VERSION = 1

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
version = project.version(ROBOFLOW_VERSION)
dataset = version.download("yolov8", location="/kaggle/working/chemistry_data")

print(f"\n데이터셋 위치: {dataset.location}")
```

### 셀 4: YOLO 학습 (GPU)

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,                # GPU 빠르니 충분히
    imgsz=640,
    device=0,                  # GPU 0번
    patience=20,
    project="/kaggle/working/runs",
    name="main_yolo",
    exist_ok=True,
)
```

학습 진행 상황이 출력됩니다. 끝나면 자동으로 다음 셀 실행 가능.

### 셀 5: best.pt를 다운로드 위치로 복사

```python
import shutil

src = "/kaggle/working/runs/main_yolo/weights/best.pt"
dst = "/kaggle/working/best.pt"
shutil.copy(src, dst)

import os
size_mb = os.path.getsize(dst) / 1024 / 1024
print(f"\nbest.pt 복사 완료: {size_mb:.1f}MB → {dst}")
print("우측 패널 'Output' 탭에서 다운로드 가능")
```

### 셀 6: (선택) 검증 결과 확인

```python
results_val = model.val()
print(f"\nmAP@50: {results_val.box.map50:.3f}")
print(f"mAP@50-95: {results_val.box.map:.3f}")
```

---

## Step 5: best.pt 다운로드 & 로컬 적용

1. **Kaggle 우측 패널** → **"Output"** 탭 클릭
2. **`best.pt`** 우클릭 → **Download**
3. 다운로드된 파일을 로컬로 복사:

```powershell
# PowerShell에서
# 다운로드 폴더 → backend 폴더로 이동
Move-Item "$env:USERPROFILE\Downloads\best.pt" `
          "c:\project\LabLog\backend\best.pt" -Force
```

4. **서버 재시작 (필수)** — `LabLogAnalyzer`가 시작 시 1회만 모델 로드:

```powershell
# 기존 uvicorn Ctrl+C로 종료 후
cd c:\project\LabLog\backend
.\.venv\Scripts\python.exe -m uvicorn server:app --reload --port 8000
```

5. 새 영상 업로드 → 새 클래스 탐지 확인

---

## ⚠ 알아둘 점

### GPU 할당 대기
무료 GPU는 가끔 대기열이 생깁니다. "Initializing... GPU 할당 중" 메시지가 길어지면 다른 시간에 재시도 (보통 대학생 시간대(저녁) 피하면 빠름).

### 세션 시간 제한
Kaggle 무료는 **9시간 연속** 세션 제한. 학습은 보통 20-40분이라 충분하지만, 데이터셋이 크거나 epochs를 너무 늘리면 주의.

### "이미 best.pt가 있어요" 경고
Kaggle은 매번 깨끗한 환경. 백업 걱정 없음. 로컬의 기존 best.pt는 train_main_yolo.py처럼 백업하지 않으므로, **수동 백업 권장**:

```powershell
Copy-Item c:\project\LabLog\backend\best.pt `
          c:\project\LabLog\backend\best.backup.pt
```

다운로드한 새 best.pt가 마음에 들지 않으면 backup.pt로 복원 가능.

### 학습 모니터링
Kaggle Notebook은 실시간 출력. 학습 중 다른 탭으로 가도 계속 진행됩니다. **브라우저 닫지 마세요** — 닫으면 세션 종료될 수 있습니다.

---

## 학습 후 단계

best.pt 적용 후:

- ✅ 더 많은 객체 탐지 → OCR 누락 감소
- ✅ 7-seg 전체 프레임 + EasyOCR bbox 구조 그대로 유지 (재구성 불필요)
- ✅ `OCR_FALLBACK_THRESHOLD` 더이상 필요 없음 (이미 제거됨)

테스트 영상 업로드해서 결과 확인해보시고, 추가 개선 필요하면 알려주세요!
