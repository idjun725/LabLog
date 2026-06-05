"""기획서 3단계 — GRU 기반 실험 단계 분류 모델.

입력: 시계열 특징 벡터 (B, T, D)
출력: 각 시점의 단계 클래스 로짓 (B, T, 5)

5개 단계: 준비 / 측정 / 반응 / 관찰 / 정리

작은 모델로 시작 — 학습 데이터가 적으면 오버피팅 위험.
hidden_dim=32, 1-layer bidirectional GRU. 데이터가 늘면 키워도 됨.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 라벨 정의 — 인덱스 순서가 모델 출력 클래스 순서.
# 변경 시 training_data.json의 라벨도 동시에 업데이트하고 재학습해야 함.
# 기존 gru_weights.pt는 출력 차원이 달라 호환되지 않으므로 학습 후 덮어쓰기 필수.
PHASES = ["준비", "반응", "측정 및 관찰", "정리"]
NUM_PHASES = len(PHASES)


class GRUStageClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim, NUM_PHASES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out, _ = self.gru(x)
        return self.classifier(out)  # (B, T, NUM_PHASES)
