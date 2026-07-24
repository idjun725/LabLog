"""GRU 단계 분류 모델 학습 CLI.

사용:
  .\.venv\Scripts\python.exe train_gru.py

training_data.json을 읽어 학습하고 gru_weights.pt로 저장한다.
샘플 데이터는 작으므로 데모용. 실제 정확도를 높이려면 학습 데이터를 추가해야 한다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from gru_model import GRUStageClassifier, NUM_PHASES, PHASES
from vectorizer import FEATURE_DIM, vectorize_records

DATA_PATH = Path(__file__).parent / "training_data.json"
WEIGHTS_PATH = Path(__file__).parent / "gru_weights.pt"
PAD_LABEL = -100  # CrossEntropyLoss ignore_index


class StageDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.phase_to_idx = {p: i for i, p in enumerate(PHASES)}
        # 검증
        for item in items:
            if len(item["records"]) != len(item["labels"]):
                raise ValueError(
                    f"records/labels 길이 불일치: {item.get('experiment', '?')}: "
                    f"{len(item['records'])} vs {len(item['labels'])}"
                )
            for label in item["labels"]:
                if label not in self.phase_to_idx:
                    raise ValueError(
                        f"미지의 라벨 '{label}' — 허용: {PHASES}"
                    )

        # vectorize_records()(SBERT 포함)를 여기서 한 번만 실행해 캐싱 — __getitem__마다
        # 다시 계산하면 epoch마다 같은 텍스트를 SBERT로 재인코딩하게 되어 (200 epoch ×
        # 실험 수) 배로 느려진다. 입력이 학습 중 바뀌지 않으므로 미리 계산해도 결과는 동일.
        print(f"[train] SBERT 벡터화 사전 계산 중 (실험 {len(items)}개, 1회만)...")
        self._cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for item in items:
            x = vectorize_records(item["records"])  # (T, D)
            y = np.array(
                [self.phase_to_idx[label] for label in item["labels"]], dtype=np.int64
            )  # (T,)
            self._cache.append((torch.from_numpy(x), torch.from_numpy(y)))
        print("[train] 벡터화 완료")

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, i: int):
        return self._cache[i]


def collate(batch):
    """가변 길이 시퀀스 패딩 — 라벨 패딩은 PAD_LABEL로 채워 loss 무시."""
    xs, ys = zip(*batch)
    x_padded = pad_sequence(xs, batch_first=True)  # (B, T_max, D)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=PAD_LABEL)
    return x_padded, y_padded


def main():
    if not DATA_PATH.exists():
        print(f"[train] 학습 데이터 없음: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    with DATA_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"[train] 학습 샘플 {len(items)}개 로드 ({DATA_PATH.name})")

    dataset = StageDataset(items)
    total_labels = sum(len(item["labels"]) for item in items)
    print(f"[train] 총 라벨 {total_labels}개, FEATURE_DIM={FEATURE_DIM}, 클래스 {NUM_PHASES}개")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate)
    model = GRUStageClassifier(input_dim=FEATURE_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)

    epochs = 200
    t0 = time.monotonic()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            logits = model(x)  # (B, T, C)
            loss = loss_fn(logits.reshape(-1, NUM_PHASES), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = y != PAD_LABEL
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()

        if epoch % 20 == 0 or epoch == epochs - 1:
            acc = correct / total if total else 0.0
            avg_loss = total_loss / max(len(loader), 1)
            print(
                f"[train] epoch {epoch + 1:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  acc={acc:.3f}"
            )

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(
        f"[train] 가중치 저장: {WEIGHTS_PATH.name} "
        f"(총 {time.monotonic() - t0:.1f}초)"
    )
    print("[train] 추론은 server.py의 /api/classify 엔드포인트로 가능합니다.")


if __name__ == "__main__":
    main()
