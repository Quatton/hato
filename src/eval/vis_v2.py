import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


with open("out/output_gpt.json", "r", encoding="utf-8") as f:
    answers = json.load(f)["results"]

FILE = "base_lamp"

with open(
    os.path.join("out/", f"eval_results_{FILE}.json"), "r", encoding="utf-8"
) as f:
    results = json.load(f)

score = 0
total = 0
for result in results:
    total += 1
    id = result["id"]
    answer = answers[id]

    if answer["index"] != id:
        print(f"Index mismatch for ID {id}: expected {answer['index']}, got {id}")
        continue

    response = result["predicted"]

    if response.get("ward") is None:
        continue

    if response["ward"] == answer["answer"]["ward"]:
        score += 1

print(f"Score: {score}/{total} ({(score / total) * 100:.2f}%)")

ward_dict = {
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


# Generate confusion matrix
def generate_confusion_matrix(results, answers, ward_dict):
    labels = sorted(list(ward_dict.values()))
    label_to_index = {label: i for i, label in enumerate(labels)}

    cm_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for result in results:
        id = result["id"]
        answer = answers[id]

        true_ward = answer["answer"].get("ward", "").lower()
        pred_ward = result["predicted"].get("ward", "").lower()

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
        vmax=60,
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
    plt.savefig(f"out/confusion_matrix_{FILE}.png")
    plt.close()


# Call confusion matrix generation
generate_confusion_matrix(results, answers, ward_dict)
