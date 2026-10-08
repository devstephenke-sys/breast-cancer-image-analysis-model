"""Preprocessing pipeline for breast cancer images."""

import pydicom
import numpy as np
from typing import Tuple, Optional
import torch
from monai.transforms import (
    Compose,
    LoadImage,
    NormalizeIntensity,
    Resize,
    RandAffine,
    RandFlip,
    RandRotate,
    RandZoom,
    ToTensor,
)


class DICOMPreprocessor:
    """Preprocess DICOM images for breast cancer classification."""

    def __init__(
        self,
        target_size: Tuple[int, int, int] = (256, 256, 32),
        normalize: bool = True,
        augment: bool = False,
    ):
        """Initialize preprocessor.

        Args:
            target_size: Target spatial size (H, W, D) for 3D or (H, W) for 2D
            normalize: Apply intensity normalization
            augment: Apply data augmentation
        """
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment

        # Build transform pipeline
        transforms = [
            LoadImage(image_only=True),
        ]

        if len(target_size) == 3:
            # 3D transforms for MRI
            transforms.extend([
                Resize(spatial_size=target_size),
            ])
        else:
            # 2D transforms for mammography
            transforms.extend([
                Resize(spatial_size=target_size),
            ])

        if normalize:
            transforms.append(NormalizeIntensity())

        if augment:
            transforms.extend([
                RandAffine(
                    rotate_range=np.pi / 12,
                    translate_range=0.1,
                    scale_range=0.1,
                    prob=0.5,
                ),
                RandFlip(spatial_axis=0, prob=0.5),
                RandFlip(spatial_axis=1, prob=0.5),
            ])

        transforms.append(ToTensor())

        self.transform = Compose(transforms)

    def __call__(self, image_path: str) -> torch.Tensor:
        """Preprocess a single image.

        Args:
            image_path: Path to DICOM file

        Returns:
            Preprocessed tensor
        """
        return self.transform(image_path)

    def load_dicom(self, dicom_path: str) -> np.ndarray:
        """Load DICOM file and return pixel array.

        Args:
            dicom_path: Path to DICOM file

        Returns:
            Pixel array as numpy array
        """
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array

        # Apply rescale slope and intercept if present
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

        return pixel_array.astype(np.float32)

    def normalize_window(self, image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
        """Apply windowing to DICOM image.

        Args:
            image: Input image
            window_center: Window center
            window_width: Window width

        Returns:
            Windowed image
        """
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2

        image = np.clip(image, window_min, window_max)
        image = (image - window_min) / (window_max - window_min)

        return image


class BreastMRI3DPreprocessor(DICOMPreprocessor):
    """Specialized preprocessor for 3D breast MRI."""

    def __init__(
        self,
        target_size: Tuple[int, int, int] = (256, 256, 32),
        normalize: bool = True,
        augment: bool = False,
    ):
        """Initialize 3D MRI preprocessor."""
        super().__init__(target_size, normalize, augment)

    def load_mri_series(self, series_paths: list) -> np.ndarray:
        """Load an MRI series as a 3D volume.

        Args:
            series_paths: List of DICOM file paths in the series

        Returns:
            3D volume array
        """
        # Sort by slice location or instance number
        slices = []
        for path in series_paths:
            ds = pydicom.dcmread(path)
            if hasattr(ds, 'SliceLocation'):
                slices.append((ds.SliceLocation, path))
            elif hasattr(ds, 'InstanceNumber'):
                slices.append((ds.InstanceNumber, path))
            else:
                slices.append((0, path))

        slices.sort(key=lambda x: x[0])

        # Load and stack slices
        volume = np.stack([self.load_dicom(path) for _, path in slices], axis=0)

        return volume


class Mammography2DPreprocessor(DICOMPreprocessor):
    """Specialized preprocessor for 2D mammography."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
        augment: bool = False,
    ):
        """Initialize 2D mammography preprocessor."""
        super().__init__(target_size, normalize, augment)

    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input image
            clip_limit: CLAHE clip limit

        Returns:
            Contrast-enhanced image
        """
        try:
            import cv2
            # Convert to uint8 for OpenCV
            image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image_uint8)
            return enhanced.astype(np.float32) / 255.0
        except ImportError:
            # Fallback to simple normalization if OpenCV not available
            return (image - image.min()) / (image.max() - image.min())


def get_preprocessor(
    modality: str,
    target_size: Optional[Tuple] = None,
    augment: bool = False,
) -> DICOMPreprocessor:
    """Get appropriate preprocessor for modality.

    Args:
        modality: Imaging modality (MR, MG, etc.)
        target_size: Target size
        augment: Whether to apply augmentation

    Returns:
        Preprocessor instance
    """
    if modality == "MR":
        if target_size is None:
            target_size = (256, 256, 32)
        return BreastMRI3DPreprocessor(target_size, augment=augment)
    elif modality == "MG":
        if target_size is None:
            target_size = (512, 512)
        return Mammography2DPreprocessor(target_size, augment=augment)
    else:
        if target_size is None:
            target_size = (256, 256)
        return DICOMPreprocessor(target_size, augment=augment)
