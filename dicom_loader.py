"""
RSNA Knee MRI DICOM Loader and 3D Spatial Geometry Ordering.
Robust against corrupt headers, missing fields, and varied transfer syntaxes.
"""

import os
import re
import math
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2

try:
    import pydicom
    try:
        from pydicom.pixels import apply_voi_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

logger = logging.getLogger("RSNA_Knee.DICOM")

def natural_sort_key(s: str) -> List[Union[int, str]]:
    """Helper for natural alphanumeric sorting (e.g. '10.dcm' after '9.dcm')."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(s))]

def compute_slice_spatial_position(dcm: Any) -> Optional[float]:
    """
    Computes 3D spatial position along the normal vector of the acquisition plane.
    Normal vector = ImageOrientationPatient[0:3] x ImageOrientationPatient[3:6]
    Position = dot(ImagePositionPatient, Normal)
    """
    try:
        if hasattr(dcm, "ImageOrientationPatient") and hasattr(dcm, "ImagePositionPatient"):
            iop = [float(v) for v in dcm.ImageOrientationPatient]
            ipp = [float(v) for v in dcm.ImagePositionPatient]
            if len(iop) == 6 and len(ipp) == 3:
                row_dir = np.array(iop[0:3], dtype=np.float64)
                col_dir = np.array(iop[3:6], dtype=np.float64)
                normal = np.cross(row_dir, col_dir)
                norm_val = np.linalg.norm(normal)
                if norm_val > 1e-6:
                    normal = normal / norm_val
                    pos_val = float(np.dot(np.array(ipp, dtype=np.float64), normal))
                    return pos_val
    except Exception:
        pass
        
    try:
        if hasattr(dcm, "SliceLocation") and dcm.SliceLocation is not None:
            return float(dcm.SliceLocation)
    except Exception:
        pass
        
    try:
        if hasattr(dcm, "InstanceNumber") and dcm.InstanceNumber is not None:
            return float(dcm.InstanceNumber)
    except Exception:
        pass
        
    return None

def order_series_slices(dcm_paths: List[str]) -> List[str]:
    """
    Robustly orders DICOM slice filepaths for an MRI series using spatial geometry hierarchy.
    Fallback: Spatial Normal Projection -> SliceLocation -> InstanceNumber -> Natural filename sort.
    """
    if len(dcm_paths) <= 1:
        return dcm_paths
        
    if not PYDICOM_AVAILABLE:
        return sorted(dcm_paths, key=natural_sort_key)
        
    slice_records = []
    has_spatial_info = False
    
    for path in dcm_paths:
        try:
            hdr = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            pos = compute_slice_spatial_position(hdr)
            inst = None
            if hasattr(hdr, "InstanceNumber") and hdr.InstanceNumber is not None:
                try:
                    inst = int(hdr.InstanceNumber)
                except (ValueError, TypeError):
                    inst = None
            if pos is not None:
                has_spatial_info = True
            slice_records.append({
                "path": path,
                "position": pos,
                "instance": inst,
                "filename_key": natural_sort_key(path)
            })
        except Exception:
            slice_records.append({
                "path": path,
                "position": None,
                "instance": None,
                "filename_key": natural_sort_key(path)
            })
            
    if has_spatial_info:
        slice_records.sort(key=lambda r: (
            r["position"] is None,
            r["position"] if r["position"] is not None else 0.0,
            r["instance"] is None,
            r["instance"] if r["instance"] is not None else 0,
            r["filename_key"]
        ))
    else:
        slice_records.sort(key=lambda r: (
            r["instance"] is None,
            r["instance"] if r["instance"] is not None else 0,
            r["filename_key"]
        ))
        
    return [r["path"] for r in slice_records]

def sample_slice_indices(total_slices: int, num_samples: int = 16) -> List[int]:
    """Uniform linear spacing slice sampler for inference."""
    if total_slices <= 0:
        return [0] * num_samples
    if total_slices <= num_samples:
        indices = list(range(total_slices))
        while len(indices) < num_samples:
            indices.append(total_slices - 1)
        return indices
    return np.linspace(0, total_slices - 1, num_samples, dtype=int).tolist()

def load_single_dicom_slice(
    path: str,
    target_size: int = 224,
    intensity_norm: str = "percentile",
    p_min: float = 1.0,
    p_max: float = 99.0
) -> np.ndarray:
    """Loads and normalizes a single DICOM slice to a [1, H, W] float32 numpy array in [0, 1]."""
    if not os.path.exists(path):
        return np.zeros((1, target_size, target_size), dtype=np.float32)
        
    img_array = None
    if PYDICOM_AVAILABLE:
        try:
            dcm = pydicom.dcmread(path, force=True)
            try:
                pixel_data = apply_voi_lut(dcm.pixel_array, dcm)
            except Exception:
                pixel_data = dcm.pixel_array
            img_array = pixel_data.astype(np.float32)
            if hasattr(dcm, "PhotometricInterpretation") and str(dcm.PhotometricInterpretation).strip() == "MONOCHROME1":
                img_array = np.max(img_array) - img_array
        except Exception:
            img_array = None
            
    if img_array is None:
        try:
            fallback_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if fallback_img is not None:
                if len(fallback_img.shape) == 3:
                    fallback_img = cv2.cvtColor(fallback_img, cv2.COLOR_BGR2GRAY)
                img_array = fallback_img.astype(np.float32)
        except Exception:
            img_array = None
            
    if img_array is None or img_array.size == 0:
        return np.zeros((1, target_size, target_size), dtype=np.float32)
        
    h, w = img_array.shape[:2]
    if h != target_size or w != target_size:
        img_array = cv2.resize(
            img_array,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA if (h > target_size) else cv2.INTER_LINEAR
        )
        
    # Percentile intensity normalization
    if intensity_norm == "percentile":
        low = np.percentile(img_array, p_min)
        high = np.percentile(img_array, p_max)
        if high > low + 1e-4:
            img_array = np.clip(img_array, low, high)
            img_array = (img_array - low) / (high - low)
        else:
            max_val = np.max(img_array)
            img_array = img_array / (max_val + 1e-6) if max_val > 0 else np.zeros_like(img_array)
    elif intensity_norm == "min_max":
        min_v, max_v = np.min(img_array), np.max(img_array)
        if max_v > min_v + 1e-4:
            img_array = (img_array - min_v) / (max_v - min_v)
        else:
            img_array = np.zeros_like(img_array)
            
    img_array = np.clip(img_array, 0.0, 1.0).astype(np.float32)
    return np.expand_dims(img_array, axis=0)
