from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import json
import logging
import math

import matplotlib.pyplot as plt


REQUIRED_RESULT_FIELDS = [
    "image_name",
    "format",
    "evaluation_success",
    "quality_status",
]


def get_logger(logger_name: str = "Agent5Reporter") -> logging.Logger:
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


def safe_category_name(category: Optional[str]) -> str:
    """
    Convert category to a safe folder name.
    """
    if not category:
        return "uncategorized"

    safe_name = str(category).strip().lower()
    safe_name = safe_name.replace(" ", "_")
    safe_name = safe_name.replace("/", "_")
    safe_name = safe_name.replace("\\", "_")
    return safe_name or "uncategorized"


def ensure_output_directories(
    reports_dir: str = "outputs/reports",
    graphs_dir: str = "outputs/graphs",
) -> Dict[str, Path]:
    """
    Create base output directories if they do not exist.
    """
    reports_path = Path(reports_dir)
    graphs_path = Path(graphs_dir)

    reports_path.mkdir(parents=True, exist_ok=True)
    graphs_path.mkdir(parents=True, exist_ok=True)

    return {
        "reports_dir": reports_path,
        "graphs_dir": graphs_path,
    }


def ensure_category_directories(
    category: str,
    reports_dir: Path,
    graphs_dir: Path,
) -> Dict[str, Path]:
    """
    Create category-specific directories.
    """
    safe_category = safe_category_name(category)

    category_reports_dir = reports_dir / safe_category
    category_graphs_dir = graphs_dir / safe_category

    category_reports_dir.mkdir(parents=True, exist_ok=True)
    category_graphs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "category_reports_dir": category_reports_dir,
        "category_graphs_dir": category_graphs_dir,
    }


def validate_evaluation_results(
    evaluation_results: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Validate evaluation results structure.
    """
    warnings = []

    if not isinstance(evaluation_results, list):
        return False, ["evaluation_results must be a list"]

    for index, result in enumerate(evaluation_results):
        if not isinstance(result, dict):
            warnings.append(f"Result at index {index} is not a dictionary")
            continue

        missing_fields = []
        for field in REQUIRED_RESULT_FIELDS:
            if field not in result:
                missing_fields.append(field)

        if missing_fields:
            warnings.append(
                f"Result at index {index} is missing fields: {', '.join(missing_fields)}"
            )

    return len(warnings) == 0, warnings


def validate_image_metadata(
    image_metadata: Dict[str, Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Validate image metadata structure.
    """
    warnings = []

    if not isinstance(image_metadata, dict):
        return False, ["image_metadata must be a dictionary"]

    for image_name, metadata in image_metadata.items():
        if not isinstance(metadata, dict):
            warnings.append(f"Metadata for image '{image_name}' is not a dictionary")

    return len(warnings) == 0, warnings


def group_results_by_image(
    evaluation_results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group evaluation results by image_name.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for result in evaluation_results:
        image_name = result.get("image_name", "unknown_image")
        if image_name not in grouped:
            grouped[image_name] = []
        grouped[image_name].append(result)

    return grouped


def is_number(value: Any) -> bool:
    """
    Check if a value is a real finite number.

    Accepts:
    - int
    - float

    Rejects:
    - None
    - NaN
    - +inf
    - -inf
    """
    if not isinstance(value, (int, float)):
        return False

    return math.isfinite(float(value))


def safe_average(values: List[Any]) -> Optional[float]:
    """
    Compute average of finite numeric values only.
    Returns None if there are no valid values.
    """
    numeric_values = [float(value) for value in values if is_number(value)]

    if not numeric_values:
        return None

    return round(sum(numeric_values) / len(numeric_values), 4)


def get_image_metadata(
    image_name: str,
    results: List[Dict[str, Any]],
    image_metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge metadata from image_metadata and fallback values from results.
    """
    metadata = image_metadata.get(image_name, {}).copy()

    if not metadata and results:
        first_result = results[0]
        metadata = {
            "category": first_result.get("category"),
            "original_image_path": first_result.get("original_image_path"),
            "original_size_bytes": first_result.get("original_size"),
            "extracted_features": first_result.get("extracted_features"),
            "llm_recommendation": first_result.get("llm_recommendation"),
            "llm_justification": first_result.get("llm_justification"),
        }

    return metadata


def build_format_comparison_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a summary of the best observed formats by metric.
    This is only descriptive reporting, not a decision step.
    """
    valid_results = [item for item in results if item.get("evaluation_success")]

    if not valid_results:
        return {
            "best_psnr_format": None,
            "best_ssim_format": None,
            "best_compression_ratio_format": None,
            "best_quality_size_score_format": None,
        }

    def best_format(metric_name: str) -> Optional[str]:
        metric_results = [
            item for item in valid_results if is_number(item.get(metric_name))
        ]
        if not metric_results:
            return None
        best_item = max(metric_results, key=lambda item: item.get(metric_name, 0))
        return best_item.get("format")

    return {
        "best_psnr_format": best_format("psnr"),
        "best_ssim_format": best_format("ssim"),
        "best_compression_ratio_format": best_format("compression_ratio"),
        "best_quality_size_score_format": best_format("quality_size_score"),
    }


def build_recommendation_relevance(
    llm_recommendation: Optional[Dict[str, Any]],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare LLM recommendation with tested/evaluated results when possible.
    """
    if not isinstance(llm_recommendation, dict):
        return {
            "recommended_format": None,
            "tested_formats": [],
            "match": None,
            "comment": "No LLM recommendation provided.",
        }

    recommended_format = llm_recommendation.get("recommended_format")
    tested_formats = sorted(
        {
            str(result.get("format"))
            for result in results
            if result.get("format") is not None
        }
    )

    if recommended_format is None:
        return {
            "recommended_format": None,
            "tested_formats": tested_formats,
            "match": None,
            "comment": "LLM recommendation exists but recommended_format is missing.",
        }

    match = recommended_format in tested_formats

    if match:
        comment = "The recommended format was tested and evaluated."
    else:
        comment = "The recommended format was not found in the evaluated results."

    return {
        "recommended_format": recommended_format,
        "tested_formats": tested_formats,
        "match": match,
        "comment": comment,
    }


def build_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build the analysis section of the final JSON report.
    """
    warnings = []
    success_count = 0
    good_count = 0
    low_count = 0
    invalid_count = 0

    psnr_values = []
    ssim_values = []
    compression_values = []
    quality_size_scores = []

    for result in results:
        if result.get("evaluation_success"):
            success_count += 1

        status = result.get("quality_status")
        if status == "good":
            good_count += 1
        elif status == "low":
            low_count += 1
        else:
            invalid_count += 1

        if result.get("warning"):
            warnings.append(result["warning"])

        psnr_values.append(result.get("psnr"))
        ssim_values.append(result.get("ssim"))
        compression_values.append(result.get("compression_ratio"))
        quality_size_scores.append(result.get("quality_size_score"))

    if success_count == 0:
        summary = "No valid evaluation result was generated"
    else:
        summary = "Compression evaluated successfully"

    return {
        "summary": summary,
        "warnings": warnings,
        "evaluation_count": len(results),
        "successful_evaluations": success_count,
        "good_quality_count": good_count,
        "low_quality_count": low_count,
        "invalid_count": invalid_count,
        "avg_psnr": safe_average(psnr_values),
        "avg_ssim": safe_average(ssim_values),
        "avg_compression_ratio": safe_average(compression_values),
        "avg_quality_size_score": safe_average(quality_size_scores),
        "notes": (
            "This report contains evaluation results only. "
            "Compression format selection was done by previous agents."
        ),
    }


def build_image_report(
    image_name: str,
    results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the final JSON report for one image.
    """
    llm_recommendation = metadata.get("llm_recommendation")
    extracted_features = metadata.get("extracted_features", {})
    llm_justification = metadata.get("llm_justification")

    score_definition = None
    if results:
        score_definition = results[0].get("score_definition")

    return {
        "image_name": image_name,
        "category": metadata.get("category"),
        "original_image": metadata.get("original_image_path", ""),
        "original_size_bytes": metadata.get("original_size_bytes", 0),
        "extracted_features": extracted_features,
        "llm_recommendation": llm_recommendation,
        "llm_justification": llm_justification,
        "llm_recommendation_relevance": build_recommendation_relevance(
            llm_recommendation,
            results,
        ),
        "format_comparison_summary": build_format_comparison_summary(results),
        "score_definition": score_definition,
        "results": results,
        "analysis": build_analysis(results),
    }


def save_json_report(
    report_data: Dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
    force: bool = True,
) -> Path:
    """
    Save one JSON report to disk.
    """
    image_name = report_data.get("image_name", "unknown_image")
    output_path = output_dir / f"{image_name}_final_report.json"

    if output_path.exists() and not force:
        logger.info("JSON report already exists, skipping: %s", output_path)
        return output_path

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report_data, file, indent=2)

    logger.info("JSON report saved: %s", output_path)
    return output_path


def save_csv_file(
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
    output_path: Path,
    logger: logging.Logger,
    force: bool = True,
) -> Path:
    """
    Generic helper to save CSV files.
    """
    if output_path.exists() and not force:
        logger.info("CSV file already exists, skipping: %s", output_path)
        return output_path

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    logger.info("CSV file saved: %s", output_path)
    return output_path


def save_summary_csv(
    evaluation_results: List[Dict[str, Any]],
    reports_dir: Path,
    logger: logging.Logger,
    filename: str = "summary.csv",
    force: bool = True,
) -> Path:
    """
    Save a global CSV summary of all evaluation results.
    """
    rows = []

    for result in evaluation_results:
        rows.append(
            {
                "image_name": result.get("image_name"),
                "category": result.get("category"),
                "format": result.get("format"),
                "quality": result.get("quality"),
                "mse": result.get("mse"),
                "mae": result.get("mae"),
                "psnr": result.get("psnr"),
                "ssim": result.get("ssim"),
                "compression_ratio": result.get("compression_ratio"),
                "quality_size_score": result.get("quality_size_score"),
                "evaluation_success": result.get("evaluation_success"),
                "quality_status": result.get("quality_status"),
                "is_valid": result.get("is_valid"),
                "warning": result.get("warning"),
            }
        )

    fieldnames = [
        "image_name",
        "category",
        "format",
        "quality",
        "mse",
        "mae",
        "psnr",
        "ssim",
        "compression_ratio",
        "quality_size_score",
        "evaluation_success",
        "quality_status",
        "is_valid",
        "warning",
    ]

    return save_csv_file(
        rows=rows,
        fieldnames=fieldnames,
        output_path=reports_dir / filename,
        logger=logger,
        force=force,
    )


def save_all_tests_summary_csv(
    evaluation_results: List[Dict[str, Any]],
    reports_dir: Path,
    logger: logging.Logger,
    filename: str = "all_tests_summary.csv",
    force: bool = True,
) -> Path:
    """
    Save a CSV file with one line per test result.
    """
    rows = []

    for index, result in enumerate(evaluation_results, start=1):
        rows.append(
            {
                "test_id": index,
                "image_name": result.get("image_name"),
                "category": result.get("category"),
                "format": result.get("format"),
                "quality": result.get("quality"),
                "quality_status": result.get("quality_status"),
                "evaluation_success": result.get("evaluation_success"),
                "psnr": result.get("psnr"),
                "ssim": result.get("ssim"),
                "compression_ratio": result.get("compression_ratio"),
                "quality_size_score": result.get("quality_size_score"),
                "warning": result.get("warning"),
            }
        )

    fieldnames = [
        "test_id",
        "image_name",
        "category",
        "format",
        "quality",
        "quality_status",
        "evaluation_success",
        "psnr",
        "ssim",
        "compression_ratio",
        "quality_size_score",
        "warning",
    ]

    return save_csv_file(
        rows=rows,
        fieldnames=fieldnames,
        output_path=reports_dir / filename,
        logger=logger,
        force=force,
    )


def aggregate_by_category(
    evaluation_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Build a summary per image category.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for result in evaluation_results:
        category = safe_category_name(result.get("category"))
        grouped.setdefault(category, []).append(result)

    summary: Dict[str, Dict[str, Any]] = {}

    for category, results in grouped.items():
        summary[category] = {
            "count": len(results),
            "avg_psnr": safe_average([item.get("psnr") for item in results]),
            "avg_ssim": safe_average([item.get("ssim") for item in results]),
            "avg_compression_ratio": safe_average(
                [item.get("compression_ratio") for item in results]
            ),
            "avg_quality_size_score": safe_average(
                [item.get("quality_size_score") for item in results]
            ),
            "good_count": sum(item.get("quality_status") == "good" for item in results),
            "low_count": sum(item.get("quality_status") == "low" for item in results),
            "invalid_count": sum(
                item.get("quality_status") == "invalid" for item in results
            ),
        }

    return summary


def aggregate_by_format(
    evaluation_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Build a summary per compression format.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for result in evaluation_results:
        format_name = result.get("format") or "unknown"
        grouped.setdefault(format_name, []).append(result)

    summary: Dict[str, Dict[str, Any]] = {}

    for format_name, results in grouped.items():
        summary[format_name] = {
            "count": len(results),
            "avg_psnr": safe_average([item.get("psnr") for item in results]),
            "avg_ssim": safe_average([item.get("ssim") for item in results]),
            "avg_compression_ratio": safe_average(
                [item.get("compression_ratio") for item in results]
            ),
            "avg_quality_size_score": safe_average(
                [item.get("quality_size_score") for item in results]
            ),
            "good_count": sum(item.get("quality_status") == "good" for item in results),
            "low_count": sum(item.get("quality_status") == "low" for item in results),
            "invalid_count": sum(
                item.get("quality_status") == "invalid" for item in results
            ),
        }

    return summary


def save_summary_dict_json(
    summary_data: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
    force: bool = True,
) -> Path:
    """
    Save a dictionary summary to JSON.
    """
    if output_path.exists() and not force:
        logger.info("JSON summary already exists, skipping: %s", output_path)
        return output_path

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary_data, file, indent=2)

    logger.info("JSON summary saved: %s", output_path)
    return output_path


def save_summary_dict_csv(
    summary_data: Dict[str, Dict[str, Any]],
    first_column_name: str,
    output_path: Path,
    logger: logging.Logger,
    force: bool = True,
) -> Path:
    """
    Save a dictionary summary to CSV.
    """
    rows = []

    for key, value in summary_data.items():
        row = {first_column_name: key}
        row.update(value)
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [first_column_name]

    return save_csv_file(
        rows=rows,
        fieldnames=fieldnames,
        output_path=output_path,
        logger=logger,
        force=force,
    )


def build_failures_summary(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a summary of failures and warnings.
    """
    warnings = []
    low_quality_results = 0
    invalid_results = 0
    larger_than_original_count = 0

    for result in evaluation_results:
        if result.get("quality_status") == "low":
            low_quality_results += 1

        if result.get("quality_status") == "invalid":
            invalid_results += 1

        warning_message = result.get("warning")
        if warning_message:
            warnings.append(warning_message)

        compression_ratio = result.get("compression_ratio")
        if is_number(compression_ratio) and compression_ratio < 0:
            larger_than_original_count += 1

    return {
        "invalid_results": invalid_results,
        "low_quality_results": low_quality_results,
        "compressed_larger_than_original_count": larger_than_original_count,
        "warnings": warnings,
    }


def build_master_report(
    evaluation_results: List[Dict[str, Any]],
    image_metadata: Dict[str, Dict[str, Any]],
    category_summary: Dict[str, Dict[str, Any]],
    format_summary: Dict[str, Dict[str, Any]],
    failures_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a global master report.
    """
    image_names = sorted(
        {
            str(result.get("image_name", "unknown_image"))
            for result in evaluation_results
        }
    )

    return {
        "total_results": len(evaluation_results),
        "images": image_names,
        "image_metadata": image_metadata,
        "evaluation_results": evaluation_results,
        "category_summary": category_summary,
        "format_summary": format_summary,
        "failures_summary": failures_summary,
    }


def build_visual_comparison_manifest(
    evaluation_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build a manifest for before/after visual comparison.
    """
    manifest = []

    for result in evaluation_results:
        manifest.append(
            {
                "image_name": result.get("image_name"),
                "category": result.get("category"),
                "original_image_path": result.get("original_image_path"),
                "compressed_image_path": result.get("compressed_image_path"),
                "format": result.get("format"),
                "quality": result.get("quality"),
                "psnr": result.get("psnr"),
                "ssim": result.get("ssim"),
                "compression_ratio": result.get("compression_ratio"),
                "quality_status": result.get("quality_status"),
            }
        )

    return manifest


def build_presentation_summary(
    evaluation_results: List[Dict[str, Any]],
    max_items: int = 5,
) -> List[Dict[str, Any]]:
    """
    Build a small summary file useful for presentation slides.
    """
    valid_results = [
        result for result in evaluation_results
        if result.get("evaluation_success") is True
    ]

    sorted_results = sorted(
        valid_results,
        key=lambda item: item.get("quality_size_score", -1)
        if is_number(item.get("quality_size_score")) else -1,
        reverse=True,
    )

    selected = sorted_results[:max_items]

    presentation_data = []
    for result in selected:
        presentation_data.append(
            {
                "image_name": result.get("image_name"),
                "category": result.get("category"),
                "format": result.get("format"),
                "quality": result.get("quality"),
                "psnr": result.get("psnr"),
                "ssim": result.get("ssim"),
                "compression_ratio": result.get("compression_ratio"),
                "quality_status": result.get("quality_status"),
                "original_image_path": result.get("original_image_path"),
                "compressed_image_path": result.get("compressed_image_path"),
            }
        )

    return presentation_data


def create_metric_graph(
    image_name: str,
    results: List[Dict[str, Any]],
    metric_name: str,
    graphs_dir: Path,
    logger: logging.Logger,
    output_filename: str,
    include_low_quality: bool = True,
    force: bool = True,
) -> Path:
    """
    Create a simple bar chart for one metric.

    Notes:
    - ignores invalid results
    - ignores None, NaN and infinite values
    """
    output_path = graphs_dir / output_filename

    if output_path.exists() and not force:
        logger.info("Graph already exists, skipping: %s", output_path)
        return output_path

    selected_results = []

    for item in results:
        if item.get("evaluation_success") is not True:
            continue

        metric_value = item.get(metric_name)
        if not is_number(metric_value):
            logger.warning(
                "Skipping invalid value for metric '%s' in image '%s': %s",
                metric_name,
                item.get("image_name"),
                metric_value,
            )
            continue

        quality_status = item.get("quality_status")
        if quality_status == "good":
            selected_results.append(item)
        elif quality_status == "low" and include_low_quality:
            selected_results.append(item)

    labels = []
    values = []

    for item in selected_results:
        format_name = item.get("format", "unknown")
        quality = item.get("quality")
        status = item.get("quality_status", "unknown")

        if quality is None:
            label = f"{format_name}-{status}"
        else:
            label = f"{format_name}-{quality}-{status}"

        labels.append(label)
        values.append(float(item.get(metric_name)))

    plt.figure(figsize=(8, 5))

    if labels:
        plt.bar(labels, values)
    else:
        plt.text(0.5, 0.5, "No valid data", ha="center", va="center")

    plt.title(f"{metric_name.upper()} comparison - {image_name}")
    plt.xlabel("Compressed result")
    plt.ylabel(metric_name.upper())
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    logger.info("Graph saved: %s", output_path)
    return output_path


def create_aggregate_graph_by_format(
    format_summary: Dict[str, Dict[str, Any]],
    metric_name: str,
    graphs_dir: Path,
    logger: logging.Logger,
    output_filename: str,
    force: bool = True,
) -> Path:
    """
    Create a global graph showing average metric by format.

    Notes:
    - ignores None, NaN and infinite values
    """
    output_path = graphs_dir / output_filename

    if output_path.exists() and not force:
        logger.info("Aggregate graph already exists, skipping: %s", output_path)
        return output_path

    labels = []
    values = []

    for format_name, summary in format_summary.items():
        metric_value = summary.get(metric_name)

        if not is_number(metric_value):
            logger.warning(
                "Skipping invalid aggregate value for metric '%s' and format '%s': %s",
                metric_name,
                format_name,
                metric_value,
            )
            continue

        labels.append(format_name)
        values.append(float(metric_value))

    plt.figure(figsize=(8, 5))

    if labels:
        plt.bar(labels, values)
    else:
        plt.text(0.5, 0.5, "No valid data", ha="center", va="center")

    plt.title(f"Average {metric_name.upper()} by format")
    plt.xlabel("Format")
    plt.ylabel(metric_name.upper())
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    logger.info("Aggregate graph saved: %s", output_path)
    return output_path


def generate_graphs_for_image(
    image_name: str,
    results: List[Dict[str, Any]],
    graphs_dir: Path,
    logger: logging.Logger,
    force: bool = True,
) -> Dict[str, str]:
    """
    Generate PSNR, SSIM and compression ratio graphs for one image.
    """
    psnr_path = create_metric_graph(
        image_name=image_name,
        results=results,
        metric_name="psnr",
        graphs_dir=graphs_dir,
        logger=logger,
        output_filename=f"{image_name}_psnr.png",
        force=force,
    )

    ssim_path = create_metric_graph(
        image_name=image_name,
        results=results,
        metric_name="ssim",
        graphs_dir=graphs_dir,
        logger=logger,
        output_filename=f"{image_name}_ssim.png",
        force=force,
    )

    ratio_path = create_metric_graph(
        image_name=image_name,
        results=results,
        metric_name="compression_ratio",
        graphs_dir=graphs_dir,
        logger=logger,
        output_filename=f"{image_name}_compression_ratio.png",
        force=force,
    )

    return {
        "psnr_graph": str(psnr_path),
        "ssim_graph": str(ssim_path),
        "compression_ratio_graph": str(ratio_path),
    }


def process_results(
    evaluation_results: List[Dict[str, Any]],
    image_metadata: Dict[str, Dict[str, Any]],
    reports_dir: str = "outputs/reports",
    graphs_dir: str = "outputs/graphs",
    logger: Optional[logging.Logger] = None,
    force: bool = True,
) -> Dict[str, Any]:
    """
    Main reporting pipeline.
    """
    if logger is None:
        logger = get_logger()

    results_valid, result_warnings = validate_evaluation_results(evaluation_results)
    metadata_valid, metadata_warnings = validate_image_metadata(image_metadata)

    for warning in result_warnings + metadata_warnings:
        logger.warning(warning)

    paths = ensure_output_directories(reports_dir=reports_dir, graphs_dir=graphs_dir)
    reports_path = paths["reports_dir"]
    graphs_path = paths["graphs_dir"]

    grouped = group_results_by_image(evaluation_results)

    saved_reports = []
    generated_graphs = {}

    for image_name, results in grouped.items():
        metadata = get_image_metadata(image_name, results, image_metadata)

        category = metadata.get("category") or (
            results[0].get("category") if results else "uncategorized"
        ) or "uncategorized"

        category_paths = ensure_category_directories(
            category=category,
            reports_dir=reports_path,
            graphs_dir=graphs_path,
        )

        report = build_image_report(
            image_name=image_name,
            results=results,
            metadata=metadata,
        )

        report_path = save_json_report(
            report_data=report,
            output_dir=category_paths["category_reports_dir"],
            logger=logger,
            force=force,
        )
        saved_reports.append(str(report_path))

        graphs = generate_graphs_for_image(
            image_name=image_name,
            results=results,
            graphs_dir=category_paths["category_graphs_dir"],
            logger=logger,
            force=force,
        )
        generated_graphs[image_name] = graphs

    summary_csv_path = save_summary_csv(
        evaluation_results=evaluation_results,
        reports_dir=reports_path,
        logger=logger,
        force=force,
    )

    all_tests_summary_csv_path = save_all_tests_summary_csv(
        evaluation_results=evaluation_results,
        reports_dir=reports_path,
        logger=logger,
        force=force,
    )

    category_summary = aggregate_by_category(evaluation_results)
    category_summary_json_path = save_summary_dict_json(
        summary_data=category_summary,
        output_path=reports_path / "category_summary.json",
        logger=logger,
        force=force,
    )
    category_summary_csv_path = save_summary_dict_csv(
        summary_data=category_summary,
        first_column_name="category",
        output_path=reports_path / "category_summary.csv",
        logger=logger,
        force=force,
    )

    format_summary = aggregate_by_format(evaluation_results)
    format_summary_json_path = save_summary_dict_json(
        summary_data=format_summary,
        output_path=reports_path / "format_summary.json",
        logger=logger,
        force=force,
    )
    format_summary_csv_path = save_summary_dict_csv(
        summary_data=format_summary,
        first_column_name="format",
        output_path=reports_path / "format_summary.csv",
        logger=logger,
        force=force,
    )

    failures_summary = build_failures_summary(evaluation_results)
    failures_summary_json_path = save_summary_dict_json(
        summary_data=failures_summary,
        output_path=reports_path / "failures_summary.json",
        logger=logger,
        force=force,
    )

    master_report = build_master_report(
        evaluation_results=evaluation_results,
        image_metadata=image_metadata,
        category_summary=category_summary,
        format_summary=format_summary,
        failures_summary=failures_summary,
    )
    master_report_json_path = save_summary_dict_json(
        summary_data=master_report,
        output_path=reports_path / "master_report.json",
        logger=logger,
        force=force,
    )

    visual_manifest = build_visual_comparison_manifest(evaluation_results)
    visual_manifest_json_path = save_summary_dict_json(
        summary_data={"items": visual_manifest},
        output_path=reports_path / "visual_comparison_manifest.json",
        logger=logger,
        force=force,
    )

    presentation_summary = build_presentation_summary(evaluation_results)
    presentation_summary_json_path = save_summary_dict_json(
        summary_data={"items": presentation_summary},
        output_path=reports_path / "presentation_summary.json",
        logger=logger,
        force=force,
    )

    global_graphs = {
        "avg_psnr_by_format": str(
            create_aggregate_graph_by_format(
                format_summary=format_summary,
                metric_name="avg_psnr",
                graphs_dir=graphs_path,
                logger=logger,
                output_filename="avg_psnr_by_format.png",
                force=force,
            )
        ),
        "avg_ssim_by_format": str(
            create_aggregate_graph_by_format(
                format_summary=format_summary,
                metric_name="avg_ssim",
                graphs_dir=graphs_path,
                logger=logger,
                output_filename="avg_ssim_by_format.png",
                force=force,
            )
        ),
        "avg_compression_ratio_by_format": str(
            create_aggregate_graph_by_format(
                format_summary=format_summary,
                metric_name="avg_compression_ratio",
                graphs_dir=graphs_path,
                logger=logger,
                output_filename="avg_compression_ratio_by_format.png",
                force=force,
            )
        ),
    }

    return {
        "validation": {
            "evaluation_results_valid": results_valid,
            "image_metadata_valid": metadata_valid,
            "warnings": result_warnings + metadata_warnings,
        },
        "json_reports": saved_reports,
        "summary_csv": str(summary_csv_path),
        "all_tests_summary_csv": str(all_tests_summary_csv_path),
        "category_summary_json": str(category_summary_json_path),
        "category_summary_csv": str(category_summary_csv_path),
        "format_summary_json": str(format_summary_json_path),
        "format_summary_csv": str(format_summary_csv_path),
        "failures_summary_json": str(failures_summary_json_path),
        "master_report_json": str(master_report_json_path),
        "visual_comparison_manifest_json": str(visual_manifest_json_path),
        "presentation_summary_json": str(presentation_summary_json_path),
        "graphs": generated_graphs,
        "global_graphs": global_graphs,
    }