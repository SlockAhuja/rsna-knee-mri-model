"""
RSNA Knee MRI Abnormality Detection - High-Performance Kaggle Inference Engine.
Self-contained, zero internet dependencies, automatic CUDA acceleration with CPU fallback.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import torch

from .model import HierarchicalKneeMRIModel
from .preprocessing import build_study_input_tensor

logger = logging.getLogger("RSNA_Knee.Inference")

DEFAULT_TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture"
]

class KneeMRIInferenceEngine:
    """
    Production-ready Kaggle Inference Engine for RSNA Knee MRI Abnormality Detection.
    """
    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir
        
        # Load configuration
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.target_columns = self.config.get("target_columns", DEFAULT_TARGETS)
        self.data_cfg = self.config.get("data", {})
        self.model_cfg = self.config.get("model", {})
        
        # Load thresholds
        thresh_path = os.path.join(model_dir, "thresholds.json")
        if os.path.exists(thresh_path):
            with open(thresh_path, "r") as f:
                self.thresholds = json.load(f)
        else:
            self.thresholds = {col: 0.5 for col in self.target_columns}
            
        # Determine device (automatic CUDA with CPU fallback)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        else:
            self.device = device
            
        # Build model architecture (weights=None, zero internet dependency)
        self.model = HierarchicalKneeMRIModel(
            encoder_name=self.model_cfg.get("encoder_name", "resnet18"),
            feature_dim=self.model_cfg.get("feature_dim", 512),
            slice_pooling=self.model_cfg.get("slice_pooling", "mean_max"),
            series_pooling=self.model_cfg.get("series_pooling", "mean_max"),
            use_metadata=self.model_cfg.get("use_metadata", True),
            metadata_dim=self.model_cfg.get("metadata_dim", 64),
            hidden_dim=self.model_cfg.get("hidden_dim", 256),
            num_classes=len(self.target_columns),
            dropout=0.0
        )
        
        # Load trained weights
        ckpt_path = os.path.join(model_dir, "best_model.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(model_dir, "best_model.pt")
            
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Trained model checkpoint not found in {model_dir}")
            
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_tensor(
        self,
        images: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        series_mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Inference on pre-assembled batch tensors.
        images: [B, S, N, 1, H, W]
        Returns: [B, 12] probabilities in [0.0, 1.0]
        """
        images = images.to(self.device)
        if metadata is not None:
            metadata = metadata.to(self.device)
        if series_mask is not None:
            series_mask = series_mask.to(self.device)
            
        logits = self.model(images, metadata=metadata, series_mask=series_mask)
        assert torch.isfinite(logits).all(), "FATAL: Non-finite logits generated during inference!"
        probs = torch.sigmoid(logits).cpu().numpy()
        assert not np.isnan(probs).any(), "FATAL: NaN generated in predicted probabilities!"
        return probs

    def predict_study(
        self,
        study_dir: str,
        series_metadata_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Full end-to-end inference on a single Study directory.
        Returns probabilities, binary thresholded predictions, and StudyInstanceUID.
        """
        study_inputs = build_study_input_tensor(
            study_dir=study_dir,
            series_metadata_df=series_metadata_df,
            image_size=self.data_cfg.get("image_size", 224),
            num_slices=self.data_cfg.get("num_slices_per_series", 16),
            max_series=self.data_cfg.get("max_series_per_study", 6),
            intensity_norm=self.data_cfg.get("intensity_norm", "percentile"),
            p_min=self.data_cfg.get("p_min", 1.0),
            p_max=self.data_cfg.get("p_max", 99.0)
        )
        
        probs = self.predict_tensor(
            images=study_inputs["images"],
            metadata=study_inputs["metadata"],
            series_mask=study_inputs["series_mask"]
        )[0]
        
        prob_dict = {col: float(probs[i]) for i, col in enumerate(self.target_columns)}
        binary_dict = {col: int(probs[i] >= self.thresholds.get(col, 0.5)) for i, col in enumerate(self.target_columns)}
        
        return {
            "StudyInstanceUID": study_inputs["study_uid"],
            "probabilities": prob_dict,
            "binary_predictions": binary_dict,
            "raw_probabilities_array": probs
        }

    def predict_batch_studies(
        self,
        study_dirs: List[str],
        series_metadata_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Runs batch prediction over a list of study directories and returns a DataFrame.
        """
        rows = []
        for s_dir in study_dirs:
            res = self.predict_study(s_dir, series_metadata_df=series_metadata_df)
            row = {"StudyInstanceUID": res["StudyInstanceUID"]}
            row.update(res["probabilities"])
            rows.append(row)
        return pd.DataFrame(rows)
