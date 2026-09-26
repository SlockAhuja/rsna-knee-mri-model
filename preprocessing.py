"""
RSNA Knee MRI Preprocessing and Volume Assembly for Kaggle Inference.
Transforms raw study DICOMs and series metadata into standardized model input tensors.
"""

import os
import glob
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import torch

from .dicom_loader import order_series_slices, sample_slice_indices, load_single_dicom_slice

def encode_series_metadata(row: Optional[Union[pd.Series, Dict[str, Any]]]) -> np.ndarray:
    """
    Encodes series metadata into a 10-dimensional vector:
    - Fluid_Sensitive: [is_0, is_1, is_unknown] (indices 0..2)
    - Fat_Suppression: [is_0, is_1, is_unknown] (indices 3..5)
    - Anatomical_Plane: [axial, coronal, sagittal, unknown] (indices 6..9)
    """
    vec = np.zeros(10, dtype=np.float32)
    if row is None:
        vec[2] = 1.0  # unknown fluid
        vec[5] = 1.0  # unknown fat
        vec[9] = 1.0  # unknown plane
        return vec
        
    # Fluid Sensitive
    fluid = row.get("Fluid_Sensitive", -1) if isinstance(row, dict) else row.get("Fluid_Sensitive", -1)
    if fluid == 1 or fluid is True or str(fluid).lower() in ["1", "true", "yes"]:
        vec[1] = 1.0
    elif fluid == 0 or fluid is False or str(fluid).lower() in ["0", "false", "no"]:
        vec[0] = 1.0
    else:
        vec[2] = 1.0
        
    # Fat Suppression
    fat = row.get("Fat_Suppression", -1) if isinstance(row, dict) else row.get("Fat_Suppression", -1)
    if fat == 1 or fat is True or str(fat).lower() in ["1", "true", "yes"]:
        vec[4] = 1.0
    elif fat == 0 or fat is False or str(fat).lower() in ["0", "false", "no"]:
        vec[3] = 1.0
    else:
        vec[5] = 1.0
        
    # Plane
    plane = str(row.get("Anatomical_Plane", "") if isinstance(row, dict) else row.get("Anatomical_Plane", "")).lower().strip()
    if "ax" in plane:
        vec[6] = 1.0
    elif "cor" in plane:
        vec[7] = 1.0
    elif "sag" in plane:
        vec[8] = 1.0
    else:
        vec[9] = 1.0
        
    return vec

def build_study_input_tensor(
    study_dir: str,
    series_metadata_df: Optional[pd.DataFrame] = None,
    image_size: int = 224,
    num_slices: int = 16,
    max_series: int = 6,
    intensity_norm: str = "percentile",
    p_min: float = 1.0,
    p_max: float = 99.0
) -> Dict[str, torch.Tensor]:
    """
    Processes all series in a given study folder and constructs:
    - images: [1, max_series, num_slices, 1, image_size, image_size]
    - metadata: [1, max_series, 10]
    - series_mask: [1, max_series]
    """
    study_uid = os.path.basename(study_dir.rstrip("/\\"))
    
    # Locate all series subdirectories
    series_dirs = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]
    
    # If no subdirectories, look for DICOMs directly
    if not series_dirs:
        dcms = glob.glob(os.path.join(study_dir, "*.dcm"))
        if dcms:
            series_dirs = ["single_series"]
            
    # Index series metadata if dataframe passed
    series_meta_lookup = {}
    if series_metadata_df is not None:
        if "StudyInstanceUID" in series_metadata_df.columns:
            sub_df = series_metadata_df[series_metadata_df["StudyInstanceUID"] == study_uid]
            for _, r in sub_df.iterrows():
                series_meta_lookup[str(r.get("SeriesInstanceUID", ""))] = r
        elif "SeriesInstanceUID" in series_metadata_df.columns:
            for _, r in series_metadata_df.iterrows():
                series_meta_lookup[str(r.get("SeriesInstanceUID", ""))] = r

    study_volumes = []
    study_meta = []
    series_valid = []
    
    for s_name in series_dirs[:max_series]:
        s_path = os.path.join(study_dir, s_name) if s_name != "single_series" else study_dir
        if os.path.isdir(s_path):
            dcm_files = glob.glob(os.path.join(s_path, "*.dcm"))
            if not dcm_files:
                dcm_files = glob.glob(os.path.join(s_path, "*"))
        else:
            dcm_files = glob.glob(os.path.join(study_dir, "*.dcm"))
            
        ordered_paths = order_series_slices(dcm_files)
        sampled_indices = sample_slice_indices(len(ordered_paths), num_samples=num_slices)
        
        vol_slices = []
        if len(ordered_paths) > 0:
            for s_idx in sampled_indices:
                slice_arr = load_single_dicom_slice(
                    ordered_paths[s_idx],
                    target_size=image_size,
                    intensity_norm=intensity_norm,
                    p_min=p_min,
                    p_max=p_max
                )
                vol_slices.append(slice_arr)
        else:
            for _ in range(num_slices):
                vol_slices.append(np.zeros((1, image_size, image_size), dtype=np.float32))
                
        vol_tensor = torch.from_numpy(np.stack(vol_slices, axis=0))  # [num_slices, 1, H, W]
        meta_row = series_meta_lookup.get(s_name, None)
        meta_vec = encode_series_metadata(meta_row)
        
        study_volumes.append(vol_tensor)
        study_meta.append(torch.from_numpy(meta_vec))
        series_valid.append(1.0)
        
    # Pad to max_series
    while len(study_volumes) < max_series:
        dummy_vol = torch.zeros((num_slices, 1, image_size, image_size), dtype=torch.float32)
        dummy_meta = torch.from_numpy(encode_series_metadata(None))
        study_volumes.append(dummy_vol)
        study_meta.append(dummy_meta)
        series_valid.append(0.0)
        
    final_images = torch.stack(study_volumes, dim=0).unsqueeze(0)    # [1, max_series, num_slices, 1, H, W]
    final_meta = torch.stack(study_meta, dim=0).unsqueeze(0)         # [1, max_series, 10]
    final_series_mask = torch.tensor(series_valid, dtype=torch.float32).unsqueeze(0) # [1, max_series]
    
    return {
        "images": final_images,
        "metadata": final_meta,
        "series_mask": final_series_mask,
        "study_uid": study_uid
    }
