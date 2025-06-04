# filepath: /Users/quatton/Documents/GitHub/hato/hato/eval/visualize.py
import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Ward dictionary for 23 special wards in Tokyo (Japanese to English)
TOKYO_WARD_DICT = {
    "千代田": "chiyoda",
    "中央": "chuo",
    "港": "minato",
    "新宿": "shinjuku",
    "文京": "bunkyo",
    "台東": "taito",
    "墨田": "sumida",
    "江東": "koto",
    "品川": "shinagawa",
    "目黒": "meguro",
    "大田": "ota",
    "世田谷": "setagaya",
    "渋谷": "shibuya",
    "中野": "nakano",
    "杉並": "suginami",
    "豊島": "toshima",
    "北": "kita",
    "荒川": "arakawa",
    "板橋": "itabashi",
    "練馬": "nerima",
    "足立": "adachi",
    "葛飾": "katsushika",
    "江戸川": "edogawa",
}


def translate_ward_jp_to_en(ward_name: str) -> str:
    """
    Translate Japanese ward name to English.
    Trims 区 suffix then matches with the ward dictionary (lowercase).
    """
    if not ward_name:
        return ""

    # Remove 区 suffix if present
    trimmed = ward_name.replace("区", "").strip()

    # Look up in dictionary and return lowercase
    # If not found in dictionary, normalize the input as English
    if trimmed in TOKYO_WARD_DICT:
        return TOKYO_WARD_DICT[trimmed].lower()
    else:
        return normalize_ward_name(ward_name)


def normalize_ward_name(ward_name: str) -> str:
    """
    Normalize ward name by removing common suffixes and converting to lowercase.
    Removes: -ku, ku, ward suffixes and strips whitespace.
    Also removes Japanese 区 suffix.
    """
    if not ward_name:
        return ""

    normalized = ward_name.strip().lower()

    # Remove Japanese 区 suffix first
    normalized = normalized.replace("区", "")
    normalized = normalized.replace("-", "")  # Replace hyphens with spaces

    # Remove common English suffixes
    suffixes_to_remove = ["ku", "ward"]
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix) and normalized != "shinjuku":
            normalized = normalized[: -len(suffix)].strip()

    return normalized.strip()


def load_verified_results(file_path: str = "out/verified_output.json") -> Dict:
    """Load verified results from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Verified output file {file_path} not found!")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_accuracy_with_normalization(verified_data: Dict) -> None:
    """Calculate and compare accuracy metrics using normalization."""
    results = verified_data["results"]
    total_results = len(results)

    # Original judgment counts
    original_ward_correct = sum(1 for r in results if r["judgment"]["ward"])

    # Normalized judgment counts
    normalized_ward_correct = 0

    for result in results:
        predicted_ward = result["answer"]["ward"]
        actual_ward = result["actual_address"]["city"]["name"]

        # Normalize both predicted and actual ward names
        predicted_normalized = normalize_ward_name(predicted_ward)
        actual_normalized = normalize_ward_name(translate_ward_jp_to_en(actual_ward))

        # Check if normalized versions match
        if predicted_normalized == actual_normalized:
            normalized_ward_correct += 1

    # Calculate percentages
    original_ward_pct = (original_ward_correct / total_results) * 100
    normalized_ward_pct = (normalized_ward_correct / total_results) * 100

    print(f"Analysis of {total_results} verified results:\n")
    print("Ward Accuracy:")
    print(
        f"  Original judgment: {original_ward_correct}/{total_results} ({original_ward_pct:.1f}%)"
    )
    print(
        f"  With normalization: {normalized_ward_correct}/{total_results} ({normalized_ward_pct:.1f}%)"
    )
    print(
        f"  Improvement: +{normalized_ward_correct - original_ward_correct} ({normalized_ward_pct - original_ward_pct:+.1f}%)\n"
    )


def plot_confusion_matrix(
    y_true_normalized: List[str],
    y_pred_normalized: List[str],
    ward_dict: Dict,
    output_path: str = "out/confusion_matrix.png",
    json_output_path: str = "out/confusion_matrix.json",
) -> None:
    """Plot and save confusion matrix as heatmap, and export data to JSON."""
    # Expect already normalized input data

    # Create confusion matrix with normalized labels
    labels = sorted(list(set(ward_dict.values())))  # Use English ward names as labels
    label_to_index = {label: i for i, label in enumerate(labels)}

    cm_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_norm, pred_norm in zip(y_true_normalized, y_pred_normalized):
        if true_norm in label_to_index and pred_norm in label_to_index:
            true_index = label_to_index[true_norm]
            pred_index = label_to_index[pred_norm]
            cm_matrix[true_index, pred_index] += 1

    # Normalize by rows (actual wards) to show ratios
    cm_normalized = np.zeros_like(cm_matrix, dtype=float)
    for i in range(len(labels)):
        row_sum = cm_matrix[i].sum()
        if row_sum > 0:
            cm_normalized[i] = cm_matrix[i] / row_sum

    # Convert to percentages
    cm_percentage = cm_normalized * 100

    # Export confusion matrix data to JSON
    confusion_matrix_data = {
        "labels": labels,
        "raw_counts": cm_matrix.tolist(),
        "normalized_percentages": cm_percentage.tolist(),
        "total_samples": len(y_true_normalized),
        "metadata": {
            "description": "Confusion matrix for Tokyo ward prediction",
            "format": {
                "raw_counts": "Integer counts of predictions",
                "normalized_percentages": "Percentage normalized by actual ward (rows sum to 100%)",
                "labels": "Ward names in alphabetical order (both rows and columns)",
            },
        },
    }

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(confusion_matrix_data, f, indent=2, ensure_ascii=False)

    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_percentage,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Percentage (%)"},
    )
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Ward")
    plt.ylabel("Actual Ward")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path)
    plt.close()


def analyze_ward_difficulty(verified_data: Dict) -> None:
    """Analyze which wards are most difficult to predict correctly."""
    results = verified_data["results"]

    ward_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for result in results:
        actual_ward = result["actual_address"]["city"]["name"]
        predicted_ward = result["answer"]["ward"]

        # Normalize both
        actual_normalized = normalize_ward_name(translate_ward_jp_to_en(actual_ward))
        predicted_normalized = normalize_ward_name(predicted_ward)

        ward_stats[actual_normalized]["total"] += 1
        if actual_normalized == predicted_normalized:
            ward_stats[actual_normalized]["correct"] += 1

    # Calculate accuracy for each ward and sort by difficulty (lowest accuracy first)
    ward_accuracy = []
    for ward, stats in ward_stats.items():
        if stats["total"] > 0:  # Only include wards that appear in the data
            accuracy = stats["correct"] / stats["total"]
            ward_accuracy.append((ward, accuracy, stats["correct"], stats["total"]))

    # Sort by accuracy (most difficult first)
    ward_accuracy.sort(key=lambda x: x[1])

    print("\nWard Prediction Difficulty Analysis:")
    print("(Most difficult wards listed first)\n")
    print(f"{'Ward':<12} {'Accuracy':<10} {'Correct':<8} {'Total':<8}")
    print("-" * 40)

    for ward, accuracy, correct, total in ward_accuracy:
        print(f"{ward:<12} {accuracy * 100:>7.1f}% {correct:>7}/{total:<7}")


def main():
    """Main function to process verified_output.json and calculate accuracy."""

    # Load and analyze verified results
    try:
        verified_data = load_verified_results()
        calculate_accuracy_with_normalization(verified_data)

        # Use all results for confusion matrix
        results = verified_data["results"]

        # Normalize ward names once
        y_true_normalized = []
        y_pred_normalized = []

        for result in results:
            actual_ward = result["actual_address"]["city"]["name"]
            predicted_ward = result["answer"]["ward"]

            # Normalize actual ward (translate Japanese to English, then normalize)
            true_normalized = normalize_ward_name(translate_ward_jp_to_en(actual_ward))
            y_true_normalized.append(true_normalized)

            # Normalize predicted ward
            pred_normalized = normalize_ward_name(predicted_ward)
            y_pred_normalized.append(pred_normalized)

        print(f"\nGenerating confusion matrix for {len(results)} results...")
        plot_confusion_matrix(y_true_normalized, y_pred_normalized, TOKYO_WARD_DICT)
        print("Confusion matrix saved as 'out/confusion_matrix.png'")
        print("Confusion matrix data exported to 'out/confusion_matrix.json'")

        # Analyze ward prediction difficulty
        analyze_ward_difficulty(verified_data)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error processing verified results: {e}")


if __name__ == "__main__":
    main()
