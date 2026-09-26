"""
RSNA Knee MRI Abnormality Detection - Self-Contained Model Architecture.
Hierarchical Multi-Series Architecture for Kaggle Inference.
No internet access or online pretrained downloads required.
"""

import math
from typing import Dict, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SliceEncoder(nn.Module):
    """
    Backbone encoder mapping single-channel MRI slice [N, 1, H, W] -> [N, feature_dim].
    Initializes architecture with custom in_channels=1 without downloading online weights.
    """
    def __init__(
        self,
        encoder_name: str = "resnet18",
        in_channels: int = 1,
        feature_dim: int = 512
    ):
        super().__init__()
        self.encoder_name = encoder_name.lower()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        
        if "resnet18" in self.encoder_name:
            base_model = models.resnet18(weights=None)
            self.feature_dim = 512
        elif "resnet34" in self.encoder_name:
            base_model = models.resnet34(weights=None)
            self.feature_dim = 512
        elif "resnet50" in self.encoder_name:
            base_model = models.resnet50(weights=None)
            self.feature_dim = 2048
        elif "efficientnet_b0" in self.encoder_name:
            base_model = models.efficientnet_b0(weights=None)
            self.feature_dim = 1280
        elif "densenet121" in self.encoder_name:
            base_model = models.densenet121(weights=None)
            self.feature_dim = 1024
        else:
            base_model = models.resnet18(weights=None)
            self.feature_dim = 512

        if "resnet" in self.encoder_name:
            if in_channels != 3:
                old_conv = base_model.conv1
                base_model.conv1 = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False
                )
            self.backbone = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
                base_model.avgpool
            )
        elif "efficientnet" in self.encoder_name:
            if in_channels != 3:
                old_conv = base_model.features[0][0]
                base_model.features[0][0] = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False
                )
            self.backbone = nn.Sequential(
                base_model.features,
                base_model.avgpool
            )
        elif "densenet" in self.encoder_name:
            if in_channels != 3:
                old_conv = base_model.features.conv0
                base_model.features.conv0 = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False
                )
            self.backbone = nn.Sequential(
                base_model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return torch.flatten(feats, 1)

class MeanMaxPool(nn.Module):
    """Combines Mean and Max pooling along the sequence dimension."""
    def __init__(self, in_features: int, project_back: bool = True):
        super().__init__()
        self.project_back = project_back
        if project_back:
            self.proj = nn.Linear(in_features * 2, in_features)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, K, D] where K is number of items (slices or series)
        mask: [B, K] where 1.0 = valid item, 0.0 = padded item
        """
        if mask is not None:
            mask_exp = mask.unsqueeze(-1)  # [B, K, 1]
            sum_x = torch.sum(x * mask_exp, dim=1)
            count = torch.sum(mask_exp, dim=1).clamp(min=1.0)
            mean_pool = sum_x / count
            
            fill_val = -1e4 if x.dtype == torch.float16 else -1e9
            masked_x = x.masked_fill(mask_exp == 0, fill_val)
            max_pool = torch.max(masked_x, dim=1)[0]
            max_pool = torch.where(count > 0, max_pool, torch.zeros_like(max_pool))
        else:
            mean_pool = torch.mean(x, dim=1)
            max_pool = torch.max(x, dim=1)[0]
            
        combined = torch.cat([mean_pool, max_pool], dim=-1)
        if self.project_back:
            return self.proj(combined)
        return combined

class GatedAttentionPool(nn.Module):
    """Gated Attention Pooling (Ilse et al.)."""
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc_v = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh())
        self.fc_u = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid())
        self.fc_w = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        v = self.fc_v(x)
        u = self.fc_u(x)
        gated = v * u
        scores = self.fc_w(gated).squeeze(-1)
        if mask is not None:
            fill_val = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask == 0, fill_val)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)

def build_aggregator(agg_type: str, in_features: int, hidden_dim: int = 128) -> nn.Module:
    agg_type = agg_type.lower()
    if agg_type in ["mean_max", "meanmax"]:
        return MeanMaxPool(in_features=in_features, project_back=True)
    elif agg_type == "mean":
        return MeanMaxPool(in_features=in_features, project_back=False)
    elif agg_type == "gated_attention":
        return GatedAttentionPool(in_features=in_features, hidden_dim=hidden_dim)
    else:
        return MeanMaxPool(in_features=in_features, project_back=True)

class SeriesMetadataEmbedder(nn.Module):
    """Embeds raw 10-dimensional series metadata."""
    def __init__(self, raw_meta_dim: int = 10, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(raw_meta_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HierarchicalKneeMRIModel(nn.Module):
    """
    End-to-End Hierarchical Knee MRI Abnormality Detection Architecture.
    Inference forward: [B, S, N, 1, H, W] -> [B, 12] logits.
    """
    def __init__(
        self,
        encoder_name: str = "resnet18",
        feature_dim: int = 512,
        slice_pooling: str = "mean_max",
        series_pooling: str = "mean_max",
        use_metadata: bool = True,
        metadata_dim: int = 64,
        hidden_dim: int = 256,
        num_classes: int = 12,
        dropout: float = 0.3
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.feature_dim = feature_dim
        self.use_metadata = use_metadata
        self.num_classes = num_classes
        
        # 1. Slice Encoder
        self.slice_encoder = SliceEncoder(
            encoder_name=encoder_name,
            in_channels=1,
            feature_dim=feature_dim
        )
        self.feature_dim = self.slice_encoder.feature_dim
        
        # 2. Slice Aggregator
        self.slice_aggregator = build_aggregator(
            agg_type=slice_pooling,
            in_features=self.feature_dim,
            hidden_dim=hidden_dim
        )
        
        # 3. Series Metadata Fusion
        if self.use_metadata:
            self.metadata_embedder = SeriesMetadataEmbedder(
                raw_meta_dim=10,
                embed_dim=metadata_dim
            )
            self.series_fusion = nn.Sequential(
                nn.Linear(self.feature_dim + metadata_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.GELU()
            )
            
        # 4. Series Aggregator
        self.series_aggregator = build_aggregator(
            agg_type=series_pooling,
            in_features=self.feature_dim,
            hidden_dim=hidden_dim
        )
        
        # 5. Study-Level Multi-Label Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        images: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        series_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        images: [B, S, N, C, H, W]
        metadata: [B, S, 10]
        series_mask: [B, S]
        Returns: [B, 12] logits
        """
        B, S, N, C, H, W = images.shape
        x_flat = images.view(B * S * N, C, H, W)
        
        slice_feats = self.slice_encoder(x_flat)
        slice_feats = slice_feats.view(B * S, N, self.feature_dim)
        
        series_feats = self.slice_aggregator(slice_feats)
        series_feats = series_feats.view(B, S, self.feature_dim)
        
        if self.use_metadata and metadata is not None:
            meta_embed = self.metadata_embedder(metadata)
            fused = torch.cat([series_feats, meta_embed], dim=-1)
            series_feats = self.series_fusion(fused)
            
        study_feats = self.series_aggregator(series_feats, mask=series_mask)
        logits = self.classifier(study_feats)
        return logits
