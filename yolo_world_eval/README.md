# YOLO-World 순수 정확도 측정 프로토타입

`backend`/`frontend`와 독립된 프로토타입 폴더입니다. 목적: **파인튜닝 없이 텍스트
프롬프트만으로 동작하는 순수 YOLO-World**가 실험기구를 실제로 얼마나 정확히
탐지하는지 정량적으로 측정합니다. `backend/open_vocab_detector.py`가 커스텀 클래스
탐지에 실제로 기대할 수 있는 신뢰도 수준을 가늠하는 게 목적입니다 — `best.pt`(파인튜닝
모델)를 대체하자는 게 아니라, 보조 탐지기로서의 실력을 숫자로 확인하는 것입니다.

## 평가 데이터셋

`best.pt` 학습에 쓴 것과 **같은** Roboflow 데이터셋(`s-workspace-qeozq/chemistry-lab-object-detection-topas` v1)을 그대로 씁니다. 이 데이터셋은 실제 실험기구 사진 + ground-truth
박스/클래스 라벨을 담고 있어, YOLO-World가 25개 클래스명을 프롬프트로 받았을 때
실제 사진에서 정확한 박스를 얼마나 잘 찾는지 mAP/precision/recall로 측정할 수 있습니다.

> **참고**: `best.pt`는 이 데이터셋으로 학습했으므로 여기서 `best.pt`의 점수를 함께
> 재보는 건 "이미 본 문제"를 다시 푸는 것이라 공정한 비교가 아닙니다. 이 프로토타입의
> 핵심은 **YOLO-World 단독 zero-shot 성능**을 숫자로 남기는 것이고, `compare_qualitative.py`는
> 그 옆에 `best.pt` 결과를 나란히 그려서 "감"을 잡기 위한 보조 도구일 뿐입니다.

## 실행 환경

별도 `.venv`를 새로 만들지 않고 **`backend/.venv`를 재사용**합니다 — `ultralytics`,
`roboflow`, `opencv-python`, `pyyaml`이 이미 다 설치돼 있어서(용량이 큰 torch를
중복 설치할 필요가 없음), 모든 스크립트를 이렇게 실행하면 됩니다:

```powershell
cd C:\LL\LabLog\yolo_world_eval
..\backend\.venv\Scripts\python.exe download_data.py
..\backend\.venv\Scripts\python.exe evaluate.py
..\backend\.venv\Scripts\python.exe compare_qualitative.py   # 선택 — 시각 비교
```

## 사전 준비: ROBOFLOW_API_KEY

데이터셋 다운로드에 필요합니다 (이 환경엔 아직 등록돼 있지 않습니다):

```powershell
$env:ROBOFLOW_API_KEY = '<your_key>'
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| `download_data.py` | Roboflow에서 라벨 포함 데이터셋을 `./data/`로 다운로드 (yolov8 포맷) |
| `evaluate.py` | 순수 `yolov8s-worldv2.pt`(파인튜닝 없음)에 25개 클래스명을 프롬프트로 주고 `./data`에 대해 공식 mAP/precision/recall 계산 → `./results/yolo_world_zeroshot/`에 confusion matrix·PR curve·`summary.json` 저장 |
| `compare_qualitative.py` | 검증 이미지 중 일부를 골라 `best.pt`(왼쪽) vs YOLO-World(오른쪽) 예측을 나란히 그려 `./results/qualitative/`에 저장 — 정성적 실패 사례 확인용 |

`data/`, `results/`는 실행 후 생성되는 산출물이라 `.gitignore`에 포함했습니다.

## 모델 선택

`evaluate.py`/`compare_qualitative.py` 둘 다 `yolov8s-worldv2.pt`를 씁니다 —
`backend/open_vocab_detector.py`가 실제 서비스에서 쓰는 것과 **동일한 모델**이라,
여기서 나온 mAP 숫자가 실제 배포 환경의 정확도를 그대로 대변합니다.
