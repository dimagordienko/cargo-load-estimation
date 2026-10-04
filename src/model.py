"""
Модель регрессии загрузки кузова: предобученный CNN-backbone (timm) +
регрессионная голова. Выход гарантированно лежит в [0, 100] за счёт
sigmoid на последнем слое.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Generalized Mean Pooling. При p=1 эквивалентен average pooling,
    при p->inf эквивалентен max pooling. Обучаемый p сам находит баланс
    между "усреднить всю картинку" и "выделить самые заметные (занятые) зоны" -
    именно это и нужно для оценки ЧАСТИЧНОЙ занятости площади пола,
    где average pooling размывает сигнал.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) - фича-карта ДО глобального пулинга
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x.flatten(1)


class CargoLoadRegressor(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True,
                 dropout: float = 0.3, pooling: str = "gem"):
        super().__init__()
        import timm

        assert pooling in ("avg", "gem")
        self.pooling_type = pooling

        # num_classes=0, global_pool="" -> backbone отдаёт фича-карту (B,C,H,W) БЕЗ пулинга,
        # чтобы мы могли применить свой (GeM) пулинг вместо стандартного average
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )
        feat_dim = self.backbone.num_features
        self.pool = GeM() if pooling == "gem" else nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)          # (B, C, H, W)
        feats = self.pool(feat_map)          # (B, C) - GeM или avg
        if self.pooling_type == "avg":
            feats = feats.flatten(1)
        out = self.head(feats)
        # sigmoid -> [0,1] -> масштаб на 100 -> гарантированный диапазон [0,100]
        return torch.sigmoid(out).squeeze(1) * 100.0


def build_model(backbone_name: str = "efficientnet_b0", pretrained: bool = True,
                 pooling: str = "gem") -> nn.Module:
    return CargoLoadRegressor(backbone_name=backbone_name, pretrained=pretrained, pooling=pooling)


if __name__ == "__main__":
    # быстрая самопроверка форм тензоров
    model = build_model()
    dummy = torch.randn(4, 3, 384, 384)
    out = model(dummy)
    print("output shape:", out.shape, "range:", out.min().item(), out.max().item())
