from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
from skimage import color, io, transform
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


REQUIRED_FIELDS = [
    "image_name",
    "original_image_path",
    "compressed_image_path",
    "format",
    "original_size",
    "compressed_size",
]

SCORE_DEFINITION = {
    "formula": "0.4*ssim + 0.3*normalized_psnr + 0.3*normalized_compression_ratio",
    "purpose": "analysis_only",
    "note": "This score is used for reporting and comparison only. It does not choose the compression format.",
}


def get_logger(logger_name: str = "Agent4Evaluator") -> logging.Logger:
    """
    Create and return a logger.
    """
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def validate_input(item: Dict[str, Any]) -> Optional[str]:
    """
    Validate required input fields.

    Returns:
    - None if input is valid
    - an error message if something is missing or invalid
    """
    if not isinstance(item, dict):
        return "Input item must be a dictionary"

    missing_fields = []

    for field in REQUIRED_FIELDS:
        if field not in item:
            missing_fields.append(field)

    if missing_fields:
        return f"Missing required fields: {', '.join(missing_fields)}"

    if not str(item.get("image_name", "")).strip():
        return "image_name is empty"

    if not str(item.get("original_image_path", "")).strip():
        return "original_image_path is empty"

    if not str(item.get("compressed_image_path", "")).strip():
        return "compressed_image_path is empty"

    if not str(item.get("format", "")).strip():
        return "format is empty"

    return None


def parse_size(value: Any, field_name: str) -> int:
    """
    Convert a size value to int and validate it.
    """
    try:
        parsed_value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer")

    if parsed_value < 0:
        raise ValueError(f"{field_name} must be >= 0")

    return parsed_value


def merge_warning_messages(*messages: Optional[str]) -> Optional[str]:
    """
    Merge warning messages into one string.
    """
    valid_messages = [message for message in messages if message]
    if not valid_messages:
        return None
    return " | ".join(valid_messages)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to [0, 1].

    Handles:
    - integer images (uint8, uint16, ...)
    - float images
    """
    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        return image.astype(np.float64) / float(max_value)

    image = image.astype(np.float64)

    if image.size == 0:
        return image

    image_min = image.min()
    image_max = image.max()

    if image_max == image_min:
        if image_max > 1.0:
            return image / image_max
        return image

    if image_min < 0.0 or image_max > 1.0:
        image = (image - image_min) / (image_max - image_min)

    return image


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image and convert it to float values in [0, 1].

    Supports:
    - grayscale
    - RGB
    - RGBA
    """
    image = io.imread(image_path)

    if image.ndim == 2:
        return normalize_image(image)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = normalize_image(image)
            image = color.rgba2rgb(image)
            return image

        if image.shape[2] == 3:
            return normalize_image(image)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def prepare_images(
    original_image: np.ndarray,
    compressed_image: np.ndarray,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Make images compatible:
    - same color mode
    - same size

    Returns:
    - prepared original image
    - prepared compressed image
    - processing notes
    """
    original = original_image.copy()
    compressed = compressed_image.copy()

    processing_notes = {
        "original_shape": list(original.shape),
        "compressed_shape_before": list(compressed.shape),
        "compressed_shape_after": None,
        "resized_for_comparison": False,
        "color_conversion_applied": None,
    }

    if original.ndim == 2 and compressed.ndim == 3:
        compressed = color.rgb2gray(compressed)
        processing_notes["color_conversion_applied"] = "rgb_to_gray"

    elif original.ndim == 3 and compressed.ndim == 2:
        compressed = np.stack([compressed] * 3, axis=-1)
        processing_notes["color_conversion_applied"] = "gray_to_rgb"

    if original.shape != compressed.shape:
        logger.info(
            "Image size mismatch detected. Resizing compressed image from %s to %s",
            compressed.shape,
            original.shape,
        )
        compressed = transform.resize(
            compressed,
            original.shape,
            preserve_range=True,
            anti_aliasing=True,
        )
        processing_notes["resized_for_comparison"] = True

    processing_notes["compressed_shape_after"] = list(compressed.shape)

    return original, compressed, processing_notes


def compute_mse(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Compute MSE.
    """
    return float(mean_squared_error(original, compressed))


def compute_mae(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Compute MAE (Mean Absolute Error).
    This is used as an additional quality metric.
    """
    return float(np.mean(np.abs(original - compressed)))


def compute_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Compute PSNR.
    """
    return float(peak_signal_noise_ratio(original, compressed, data_range=1.0))


def compute_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Compute SSIM.
    """
    if original.ndim == 2:
        return float(structural_similarity(original, compressed, data_range=1.0))

    return float(
        structural_similarity(
            original,
            compressed,
            channel_axis=-1,
            data_range=1.0,
        )
    )


def compute_compression_ratio(
    original_size: int,
    compressed_size: int,
    logger: logging.Logger,
) -> float:
    """
    Compute compression ratio:
    t = (1 - compressed_size / original_size) * 100
    """
    if original_size <= 0:
        logger.warning("Original size is <= 0. Compression ratio set to 0.")
        return 0.0

    ratio = (1 - (compressed_size / original_size)) * 100
    return float(ratio)


def normalize_psnr(psnr_value: float, max_psnr: float = 50.0) -> float:
    """
    Normalize PSNR into [0, 1].
    """
    if np.isinf(psnr_value):
        return 1.0
    return float(np.clip(psnr_value / max_psnr, 0.0, 1.0))


def normalize_compression_ratio(compression_ratio: float) -> float:
    """
    Normalize compression ratio into [0, 1].
    Negative values are clipped to 0.
    """
    return float(np.clip(compression_ratio / 100.0, 0.0, 1.0))


def compute_quality_size_score(
    ssim_value: float,
    psnr_value: float,
    compression_ratio: float,
) -> float:
    """
    Compute a simple quality-size score for analysis only.

    Weights:
    - 40% SSIM
    - 30% normalized PSNR
    - 30% normalized compression ratio
    """
    normalized_psnr = normalize_psnr(psnr_value)
    normalized_ratio = normalize_compression_ratio(compression_ratio)

    score = (
        0.4 * ssim_value
        + 0.3 * normalized_psnr
        + 0.3 * normalized_ratio
    )
    return float(score)


def determine_quality_status(
    evaluation_success: bool,
    ssim_value: Optional[float],
    ssim_threshold: float,
) -> str:
    """
    Return:
    - 'invalid' if evaluation failed
    - 'low' if evaluation succeeded but quality is below threshold
    - 'good' if evaluation succeeded and quality is acceptable
    """
    if not evaluation_success:
        return "invalid"

    if ssim_value is None:
        return "invalid"

    if ssim_value < ssim_threshold:
        return "low"

    return "good"


def build_result_template(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an empty result structure.
    """
    return {
        "image_name": item.get("image_name"),
        "category": item.get("category"),
        "original_image_path": item.get("original_image_path"),
        "compressed_image_path": item.get("compressed_image_path"),
        "format": item.get("format"),
        "quality": item.get("quality"),
        "original_size": item.get("original_size"),
        "compressed_size": item.get("compressed_size"),
        "extracted_features": item.get("extracted_features"),
        "llm_recommendation": item.get("llm_recommendation"),
        "llm_justification": item.get("llm_justification"),
        "mse": None,
        "mae": None,
        "psnr": None,
        "ssim": None,
        "compression_ratio": None,
        "quality_size_score": None,
        "evaluation_success": False,
        "quality_status": "invalid",
        "is_valid": False,
        "warning": None,
        "processing_notes": {},
        "score_definition": SCORE_DEFINITION,
    }


def evaluate_one(
    item: Dict[str, Any],
    ssim_threshold: float = 0.80,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Evaluate one compressed image result.

    Expected input example:
    {
        "image_name": "image1",
        "category": "photo",
        "original_image_path": "data/originals/image1.png",
        "compressed_image_path": "outputs/compressed/image1_q85.jpg",
        "format": "jpeg",
        "quality": 85,
        "original_size": 245678,
        "compressed_size": 75432,
        "extracted_features": {...},      # optional
        "llm_recommendation": {...},      # optional
        "llm_justification": "..."        # optional
    }
    """
    if logger is None:
        logger = get_logger()

    result = build_result_template(item)

    validation_error = validate_input(item)
    if validation_error is not None:
        result["warning"] = validation_error
        result["processing_notes"] = {
            "validation_passed": False,
            "resized_for_comparison": False,
            "color_conversion_applied": None,
        }
        logger.error(validation_error)
        return result

    try:
        original_path = Path(item["original_image_path"])
        compressed_path = Path(item["compressed_image_path"])

        if not original_path.exists():
            message = f"Original image not found: {original_path}"
            result["warning"] = message
            logger.error(message)
            return result

        if not compressed_path.exists():
            message = f"Compressed image not found: {compressed_path}"
            result["warning"] = message
            logger.error(message)
            return result

        original_image = load_image(str(original_path))
        compressed_image = load_image(str(compressed_path))

        original_image, compressed_image, processing_notes = prepare_images(
            original_image,
            compressed_image,
            logger,
        )

        mse_value = compute_mse(original_image, compressed_image)
        mae_value = compute_mae(original_image, compressed_image)
        psnr_value = compute_psnr(original_image, compressed_image)
        ssim_value = compute_ssim(original_image, compressed_image)

        original_size = parse_size(item["original_size"], "original_size")
        compressed_size = parse_size(item["compressed_size"], "compressed_size")

        compression_ratio = compute_compression_ratio(
            original_size,
            compressed_size,
            logger,
        )

        score = compute_quality_size_score(
            ssim_value,
            psnr_value,
            compression_ratio,
        )

        warning_message = None

        if ssim_value < ssim_threshold:
            warning_message = merge_warning_messages(
                warning_message,
                "recompress with higher quality",
            )
            logger.warning(
                "Low quality detected for image '%s' (SSIM=%.4f)",
                item.get("image_name"),
                ssim_value,
            )

        if compression_ratio < 0:
            warning_message = merge_warning_messages(
                warning_message,
                "compressed file is larger than original file",
            )
            logger.warning(
                "Compressed file is larger than original for image '%s'",
                item.get("image_name"),
            )

        quality_status = determine_quality_status(
            evaluation_success=True,
            ssim_value=ssim_value,
            ssim_threshold=ssim_threshold,
        )

        processing_notes["validation_passed"] = True

        result.update(
            {
                "mse": round(mse_value, 4),
                "mae": round(mae_value, 4),
                "psnr": round(psnr_value, 4) if np.isfinite(psnr_value) else float("inf"),
                "ssim": round(ssim_value, 4),
                "compression_ratio": round(compression_ratio, 4),
                "quality_size_score": round(score, 4),
                "evaluation_success": True,
                "quality_status": quality_status,
                "is_valid": quality_status == "good",
                "warning": warning_message,
                "processing_notes": processing_notes,
            }
        )

        logger.info(
            "Evaluation completed for image '%s' with status '%s'",
            item.get("image_name"),
            quality_status,
        )

        return result

    except Exception as error:
        message = f"Evaluation error: {str(error)}"
        result["warning"] = message
        result["evaluation_success"] = False
        result["quality_status"] = "invalid"
        result["is_valid"] = False
        result["processing_notes"] = {
            "validation_passed": True,
            "resized_for_comparison": False,
            "color_conversion_applied": None,
        }
        logger.exception(message)
        return result


def evaluate_batch(
    items: List[Dict[str, Any]],
    ssim_threshold: float = 0.80,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple compressed image results.
    """
    if logger is None:
        logger = get_logger()

    results: List[Dict[str, Any]] = []

    for item in items:
        results.append(
            evaluate_one(
                item=item,
                ssim_threshold=ssim_threshold,
                logger=logger,
            )
        )

    return results