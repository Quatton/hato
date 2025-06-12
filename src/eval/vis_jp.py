import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches


FILES = [
    "lora_jp_sampled",
    # "lora_7b_jp_sampled",
    # "qwen3b_jp_sampled",
    # "qwen7b_jp_sampled",
    # "gpt41_jp_sampled",
    # "gpt41_jp_sampled_p_02",
    # "gpt41_jp_sampled_p_02_no_tokyo",
    # "gpt41_jp_sampled_p_02_top_3",
    "lora_jp_sampled_no_tokyo",
]


# Calculate region boundaries based on prefecture order
def calculate_region_boundaries():
    """Calculate the start and end indices for each region in the prefecture list."""
    region_boundaries = {}
    current_index = 0

    # Group prefectures by region in order
    regions_order = [
        "hokkaido",
        "tohoku",
        "kanto",
        "chubu",
        "kinki",
        "chugoku",
        "shikoku",
        "kyushu",
        "okinawa",
    ]

    for region in regions_order:
        region_prefs = [pref for pref in PREFECTURE if REGION[pref] == region]
        start_idx = current_index
        end_idx = current_index + len(region_prefs) - 1
        region_boundaries[region] = (start_idx, end_idx + 1)
        current_index += len(region_prefs)

    return region_boundaries


# Generate confusion matrix
def generate_confusion_matrix(results):
    labels = PREFECTURE
    label_to_index = {label: i for i, label in enumerate(labels)}

    cm_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for result in results:
        true_ward = result["prefecture"].lower()
        pred_ward = result["guess_prefecture"]
        pred_wards = [pred_ward] if isinstance(pred_ward, str) else pred_ward

        # if pred_ward == "tokyo":
        #     continue

        for pred_ward in pred_wards:
            pred_ward = pred_ward.lower()
            if true_ward in label_to_index and pred_ward in label_to_index:
                true_index = label_to_index[true_ward]
                pred_index = label_to_index[pred_ward]
                cm_matrix[true_index, pred_index] += 1

    # Normalize by rows
    cm_normalized = np.zeros_like(cm_matrix, dtype=float)
    for i in range(len(labels)):
        row_sum = cm_matrix[i].sum()
        if row_sum > 0:
            cm_normalized[i] = cm_matrix[i] / row_sum

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(24, 20))
    sns.heatmap(
        cm_normalized * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmax=100,
        vmin=0,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Percentage (%)"},
        ax=ax,
    )

    for region, (start, end) in REGION_BOUNDARIES.items():
        color = REGION_COLOR[region]
        width = end - start
        height = end - start

        # Draw rectangle border for each region
        # Ensure the rectangle exactly surrounds the grid cells
        rect = patches.Rectangle(
            (start, start),
            width,
            height,
            linewidth=5,
            edgecolor=color,
            facecolor="none",
            linestyle="-",
            clip_on=False,  # Ensure the rectangle is not clipped
        )
        ax.add_patch(rect)

    plt.title("Confusion Matrix Heatmap with Region Boundaries")
    plt.xlabel("Predicted Ward")
    plt.ylabel("Actual Ward")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"out/confusion_matrix_{FILE}.png")
    plt.close()


PREFECTURE = [
    "hokkaido",
    "aomori",
    "iwate",
    "miyagi",
    "akita",
    "yamagata",
    "fukushima",
    "ibaraki",
    "tochigi",
    "gunma",
    "saitama",
    "chiba",
    "tokyo",
    "kanagawa",
    "niigata",
    "toyama",
    "ishikawa",
    "fukui",
    "yamanashi",
    "nagano",
    "gifu",
    "shizuoka",
    "aichi",
    "mie",
    "shiga",
    "kyoto",
    "osaka",
    "hyogo",
    "nara",
    "wakayama",
    "tottori",
    "shimane",
    "okayama",
    "hiroshima",
    "yamaguchi",
    "tokushima",
    "kagawa",
    "ehime",
    "kochi",
    "fukuoka",
    "saga",
    "nagasaki",
    "kumamoto",
    "oita",
    "miyazaki",
    "kagoshima",
    "okinawa",
]

REGION = {
    "hokkaido": "hokkaido",
    "aomori": "tohoku",
    "iwate": "tohoku",
    "miyagi": "tohoku",
    "akita": "tohoku",
    "yamagata": "tohoku",
    "fukushima": "tohoku",
    "ibaraki": "kanto",
    "tochigi": "kanto",
    "gunma": "kanto",
    "saitama": "kanto",
    "chiba": "kanto",
    "tokyo": "kanto",
    "kanagawa": "kanto",
    "niigata": "chubu",
    "toyama": "chubu",
    "ishikawa": "chubu",
    "fukui": "chubu",
    "yamanashi": "chubu",
    "nagano": "chubu",
    "gifu": "chubu",
    "shizuoka": "chubu",
    "aichi": "chubu",
    "mie": "kinki",
    "shiga": "kinki",
    "kyoto": "kinki",
    "osaka": "kinki",
    "hyogo": "kinki",
    "nara": "kinki",
    "wakayama": "kinki",
    "tottori": "chugoku",
    "shimane": "chugoku",
    "okayama": "chugoku",
    "hiroshima": "chugoku",
    "yamaguchi": "chugoku",
    "tokushima": "shikoku",
    "kagawa": "shikoku",
    "ehime": "shikoku",
    "kochi": "shikoku",
    "fukuoka": "kyushu",
    "saga": "kyushu",
    "nagasaki": "kyushu",
    "kumamoto": "kyushu",
    "oita": "kyushu",
    "miyazaki": "kyushu",
    "kagoshima": "kyushu",
    "okinawa": "okinawa",
}

REGION_COLOR = {
    "hokkaido": "#FF6B6B",
    "tohoku": "#4ECDC4",
    "kanto": "#45B7D1",
    "chubu": "#96CEB4",
    "kinki": "#FFEAA7",
    "chugoku": "#DDA0DD",
    "shikoku": "#98D8C8",
    "kyushu": "#F7DC6F",
    "okinawa": "#BB8FCE",
}

PREFECTURE_ALPHABETICAL = sorted(PREFECTURE)

# Calculate region boundaries after constants are defined
REGION_BOUNDARIES = calculate_region_boundaries()

for FILE in FILES:
    with open(
        os.path.join("out/", f"eval_results_{FILE}.json"), "r", encoding="utf-8"
    ) as f:
        results = json.load(f)

    score = 0
    total = 0
    for result in results:
        # if result["guess_prefecture"] == "tokyo":
        #     continue
        total += 1
        id = result["id"]

        guess_prefecture = result["guess_prefecture"]
        guess_prefectures = (
            [guess_prefecture]
            if isinstance(guess_prefecture, str)
            else guess_prefecture
        )

        for guess in guess_prefectures:
            if result["prefecture"].lower() == guess.lower():
                score += 1
                break

    print(f"Score: {score}/{total} ({(score / total) * 100:.2f}%)")

    # Call confusion matrix generation
    generate_confusion_matrix(results)
