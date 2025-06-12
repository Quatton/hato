import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILE = "base_7b_cheat"

with open(
    os.path.join("out/", f"eval_results_{FILE}.json"), "r", encoding="utf-8"
) as f:
    results = json.load(f)

score = 0
total = 0
for result in results:
    total += 1
    id = result["id"]
    if result["match"]:
        score += 1

print(f"Score: {score}/{total} ({(score / total) * 100:.2f}%)")

ward_dict = {
    "世田谷": "setagaya",
    "練馬": "nerima",
    "中野": "nakano",
    "豊島": "toshima",
    "目黒": "meguro",
    "杉並": "suginami",
    "葛飾": "katsushika",
    "足立": "adachi",
    "大田": "ota",
    "江戸川": "edogawa",
    "北": "kita",
    "荒川": "arakawa",
    "板橋": "itabashi",
    "品川": "shinagawa",
    "江東": "koto",
    "墨田": "sumida",
    "台東": "taito",
    "文京": "bunkyo",
    "港": "minato",
    "新宿": "shinjuku",
    "渋谷": "shibuya",
    "中央": "chuo",
    "千代田": "chiyoda",
}


# Generate confusion matrix
def generate_confusion_matrix(results, ward_dict):
    labels = list(ward_dict.values())
    label_to_index = {label: i for i, label in enumerate(labels)}

    cm_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for result in results:
        if "guess_ward" in result:
            true_ward = result["ward"]
            pred_ward = result["guess_ward"]
        else:
            true_ward = result["expected"]["ward"]
            pred_ward = result["predicted"]["ward"]

        if true_ward not in labels or pred_ward not in labels:
            continue

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
    plt.figure(figsize=(12, 10))
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
    )
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Ward")
    plt.ylabel("Actual Ward")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"out/sorted_confusion_matrix_{FILE}.png")
    plt.close()


# Call confusion matrix generation
generate_confusion_matrix(results, ward_dict)
