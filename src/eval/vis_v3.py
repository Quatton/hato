import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


with open("out/output_gpt.json", "r", encoding="utf-8") as f:
    answers = json.load(f)["results"]

FILE = "lora_3b_64"

with open(
    os.path.join("out/", f"eval_results_{FILE}.json"), "r", encoding="utf-8"
) as f:
    results = json.load(f)


# ward_dict = {
#     "千代田": "chiyoda",
#     "中央": "chuo",
#     "港": "minato",
#     "新宿": "shinjuku",
#     "文京": "bunkyo",
#     "台東": "taito",
#     "墨田": "sumida",
#     "江東": "koto",
#     "品川": "shinagawa",
#     "目黒": "meguro",
#     "大田": "ota",
#     "世田谷": "setagaya",
#     "渋谷": "shibuya",
#     "中野": "nakano",
#     "杉並": "suginami",
#     "豊島": "toshima",
#     "北": "kita",
#     "荒川": "arakawa",
#     "板橋": "itabashi",
#     "練馬": "nerima",
#     "足立": "adachi",
#     "葛飾": "katsushika",
#     "江戸川": "edogawa",
# }

ward_dict = {
    "chiyoda": 0,
    "chuo": 0,
    "minato": 0,
    "shinjuku": 1,
    "toshima": 1,
    "bunkyo": 2,
    "taito": 2,
    "sumida": 2,
    "koto": 2,
    "shinagawa": 3,
    "ota": 3,
    "shibuya": 4,
    "meguro": 4,
    "setagaya": 5,
    "arakawa": 6,
    "itabashi": 6,
    "kita": 6,
    "nakano": 6,
    "suginami": 6,
    "nerima": 6,
    "adachi": 7,
    "katsushika": 7,
    "edogawa": 7,
}

ward_class_name = {
    0: "Chiyoda, Chuo, Minato",
    1: "Shinjuku, Toshima",
    2: "Bunkyo, Taito, Sumida, Koto",
    3: "Shinagawa, Ota",
    4: "Shibuya, Meguro",
    5: "Setagaya",
    6: "Arakawa, Itabashi, Kita, Nakano, Suginami, Nerima",
    7: "Adachi, Katsushika, Edogawa",
    8: "Unknown",
}


# Generate confusion matrix
def generate_confusion_matrix(results, answers, ward_dict):
    labels = list(ward_class_name.values())

    cm_matrix = np.zeros((len(labels) - 1, len(labels)), dtype=int)

    for result in results:
        id = result["id"]
        answer = answers[id]

        true_ward = answer["answer"].get("ward", "").lower()
        pred_ward = result["predicted"].get("ward", "").lower()

        true_index = ward_dict.get(true_ward, 8)

        if true_index == 8:
            continue

        pred_index = ward_dict.get(pred_ward, 8)

        cm_matrix[true_index, pred_index] += 1

    # Normalize by rows
    cm_normalized = np.zeros_like(cm_matrix, dtype=float)
    for i in range(len(labels) - 1):
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
        yticklabels=labels[:-1],
        cbar_kws={"label": "Percentage (%)"},
    )
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Ward")
    plt.ylabel("Actual Ward")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"out/confusion_matrix_class_{FILE}.png")
    plt.close()


# Call confusion matrix generation
generate_confusion_matrix(results, answers, ward_dict)

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

    if ward_dict.get(response["ward"].lower(), 8) == ward_dict.get(
        answer["answer"]["ward"].lower(), 8
    ):
        score += 1

print(f"Score: {score}/{total} ({(score / total) * 100:.2f}%)")
