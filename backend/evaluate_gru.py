"""GRU 단계 분류기 평가 — Baseline(33-dim) vs SBERT(801-dim) 비교.

Leave-One-Experiment-Out (LOEO) 교차검증:
- 실험 5개 → fold 5개. 각 fold에서 4개 실험으로 학습, 남은 1개로 평가.
- Baseline과 SBERT 두 특징 세트를 동일 하이퍼파라미터로 학습·평가.
- fold별 accuracy 쌍으로 scipy.stats.wilcoxon (Wilcoxon Signed-Rank Test) 수행.
- Macro F1도 함께 리포트 (클래스 불균형 대비).

사용:
  .\.venv\Scripts\python.exe evaluate_gru.py

주의:
- fold 수가 5개라 Wilcoxon 검정력은 매우 낮음. p-value 해석에 주의.
- 재현성 위해 torch/numpy seed 고정. SBERT encoding은 결정론적.
- SBERT 모델 다운로드(~420MB) 최초 1회 필요.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from scipy import stats

from gru_model import GRUStageClassifier, NUM_PHASES, PHASES
from vectorizer import BASELINE_FEATURE_DIM, SBERT_FEATURE_DIM, vectorize_records

DATA_PATH = Path(__file__).parent / "training_data.json"
EPOCHS = 200
LR = 1e-3
SEED = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_one_fold(
    train_X: list[np.ndarray],
    train_Y: list[np.ndarray],
    input_dim: int,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> GRUStageClassifier:
    """4개 실험(가변 길이)으로 GRU 학습. batch=len(train_X) 로 한 번에.

    train_gru.py의 로직을 축약. batch 크기가 4로 작아 padding + ignore_index로 처리.
    """
    from torch.nn.utils.rnn import pad_sequence

    xs = [torch.from_numpy(x) for x in train_X]
    ys = [torch.from_numpy(y) for y in train_Y]
    x_padded = pad_sequence(xs, batch_first=True)      # (B, T_max, D)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=-100)

    model = GRUStageClassifier(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for _ in range(epochs):
        logits = model(x_padded)
        loss = loss_fn(logits.reshape(-1, NUM_PHASES), y_padded.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _evaluate(model: GRUStageClassifier, x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """단일 실험 시퀀스 예측 → (accuracy, macro F1, preds)."""
    with torch.no_grad():
        logits = model(torch.from_numpy(x).unsqueeze(0))  # (1, T, C)
        preds = logits.argmax(dim=-1).squeeze(0).numpy()  # (T,)
    acc = float((preds == y).mean())
    macro_f1 = float(f1_score(y, preds, labels=list(range(NUM_PHASES)), average="macro", zero_division=0))
    return acc, macro_f1, preds


def _print_confusion(all_y: np.ndarray, all_pred: np.ndarray, tag: str) -> None:
    print(f"\n[{tag}] 혼동 행렬 (row=실제, col=예측)")
    header = "실제↓/예측→".ljust(14) + "".join(p.ljust(14) for p in PHASES)
    print(header)
    for i, phase in enumerate(PHASES):
        row = [int((all_pred == j)[all_y == i].sum()) for j in range(NUM_PHASES)]
        print(phase.ljust(14) + "".join(str(v).ljust(14) for v in row))


def main() -> int:
    if not DATA_PATH.exists():
        print(f"[eval] 학습 데이터 없음: {DATA_PATH}", file=sys.stderr)
        return 1

    _set_seed(SEED)

    with DATA_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)

    n_folds = len(items)
    print(f"[eval] 실험 {n_folds}개, 총 라벨 {sum(len(i['labels']) for i in items)}개")
    print(f"[eval] LOEO 5-fold — 각 fold에서 4개 실험 학습 / 1개 평가")
    print(f"[eval] Baseline dim={BASELINE_FEATURE_DIM}, SBERT dim={SBERT_FEATURE_DIM}")

    phase_to_idx = {p: i for i, p in enumerate(PHASES)}
    Y_all = [np.array([phase_to_idx[l] for l in it["labels"]], dtype=np.int64) for it in items]

    # 사전 벡터화 — SBERT 인코딩은 무겁지만 결정론적이므로 fold마다 재계산 불필요.
    print(f"[eval] 벡터화 (SBERT 임베딩 사전 계산)…", flush=True)
    t0 = time.monotonic()
    X_baseline = [vectorize_records(it["records"], use_sbert=False) for it in items]
    X_sbert = [vectorize_records(it["records"], use_sbert=True) for it in items]
    print(f"[eval] 벡터화 완료 ({time.monotonic() - t0:.1f}초)")

    baseline_acc, baseline_f1 = [], []
    sbert_acc, sbert_f1 = [], []
    baseline_y, baseline_pred = [], []
    sbert_y, sbert_pred = [], []

    for test_i in range(n_folds):
        train_indices = [i for i in range(n_folds) if i != test_i]
        exp_name = items[test_i].get("experiment", f"fold{test_i}")

        # Baseline
        _set_seed(SEED + test_i)  # fold별 다른 seed but 두 모드가 같은 seed 공유
        model_b = _train_one_fold(
            [X_baseline[i] for i in train_indices],
            [Y_all[i] for i in train_indices],
            input_dim=BASELINE_FEATURE_DIM,
        )
        acc_b, f1_b, pred_b = _evaluate(model_b, X_baseline[test_i], Y_all[test_i])
        baseline_acc.append(acc_b)
        baseline_f1.append(f1_b)
        baseline_y.append(Y_all[test_i])
        baseline_pred.append(pred_b)

        # SBERT
        _set_seed(SEED + test_i)
        model_s = _train_one_fold(
            [X_sbert[i] for i in train_indices],
            [Y_all[i] for i in train_indices],
            input_dim=SBERT_FEATURE_DIM,
        )
        acc_s, f1_s, pred_s = _evaluate(model_s, X_sbert[test_i], Y_all[test_i])
        sbert_acc.append(acc_s)
        sbert_f1.append(f1_s)
        sbert_y.append(Y_all[test_i])
        sbert_pred.append(pred_s)

        print(
            f"[fold {test_i + 1}/{n_folds}] {exp_name}  "
            f"baseline acc={acc_b:.3f} F1={f1_b:.3f}  |  "
            f"sbert acc={acc_s:.3f} F1={f1_s:.3f}"
        )

    baseline_acc_arr = np.array(baseline_acc)
    sbert_acc_arr = np.array(sbert_acc)
    baseline_f1_arr = np.array(baseline_f1)
    sbert_f1_arr = np.array(sbert_f1)

    print("\n" + "=" * 60)
    print("결과 요약 (mean ± std, n_folds=5)")
    print("=" * 60)
    print(f"Baseline  acc = {baseline_acc_arr.mean():.4f} ± {baseline_acc_arr.std(ddof=1):.4f}   "
          f"macroF1 = {baseline_f1_arr.mean():.4f} ± {baseline_f1_arr.std(ddof=1):.4f}")
    print(f"SBERT     acc = {sbert_acc_arr.mean():.4f} ± {sbert_acc_arr.std(ddof=1):.4f}   "
          f"macroF1 = {sbert_f1_arr.mean():.4f} ± {sbert_f1_arr.std(ddof=1):.4f}")
    print(f"차이(SBERT − Baseline)  acc = {(sbert_acc_arr - baseline_acc_arr).mean():+.4f}   "
          f"macroF1 = {(sbert_f1_arr - baseline_f1_arr).mean():+.4f}")

    # Wilcoxon Signed-Rank Test — 쌍으로 된 fold accuracy에 적용.
    # 차이가 모두 0이면 scipy가 예외를 던지므로 방어.
    diffs = sbert_acc_arr - baseline_acc_arr
    if np.all(diffs == 0):
        print("\n[wilcoxon] 두 모델의 fold accuracy가 모두 동일 — 검정 수행 불가")
    else:
        try:
            # zero_method='wilcox': 0을 제거하고 나머지에만 순위 부여 (표준적)
            # alternative='greater': SBERT > Baseline 검정
            stat, p_greater = stats.wilcoxon(sbert_acc_arr, baseline_acc_arr,
                                             zero_method="wilcox", alternative="greater")
            _, p_two_sided = stats.wilcoxon(sbert_acc_arr, baseline_acc_arr,
                                            zero_method="wilcox", alternative="two-sided")
            print(f"\n[wilcoxon] SBERT > Baseline  W={stat:.3f}  p={p_greater:.4f} (one-sided)")
            print(f"[wilcoxon] SBERT ≠ Baseline               p={p_two_sided:.4f} (two-sided)")
            print("           주의: n_folds=5이라 검정력이 매우 낮음. p 해석은 참고용.")
        except ValueError as e:
            print(f"\n[wilcoxon] 계산 불가: {e}")

    _print_confusion(np.concatenate(baseline_y), np.concatenate(baseline_pred), "Baseline")
    _print_confusion(np.concatenate(sbert_y), np.concatenate(sbert_pred), "SBERT")

    print("\n[eval] 성공 기준(PRD): SBERT accuracy ≥ 0.55")
    target = 0.55
    result = "PASS" if sbert_acc_arr.mean() >= target else "MISS"
    print(f"[eval] SBERT mean accuracy = {sbert_acc_arr.mean():.4f} → {result} (목표 {target})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
