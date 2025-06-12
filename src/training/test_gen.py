from functools import reduce
import json


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

PREFECTURE_SAMPLED_ID: dict[str, list[str]] = {pref: [] for pref in PREFECTURE}

with open("out/japan-5k-compact.json", "r", encoding="utf-8") as f:
    samples = json.load(f)[1000:]

for sample in samples:
    prefecture = sample["prefecture"]
    if len(PREFECTURE_SAMPLED_ID[prefecture]) < 20:
        PREFECTURE_SAMPLED_ID[prefecture].append(sample)

with open("out/japan-5k-sampled.json", "w", encoding="utf-8") as f:
    for prefecture, samples in PREFECTURE_SAMPLED_ID.items():
        print(f"{prefecture}: {len(samples)} samples")

    dataset = reduce(lambda x, y: x + y, list(PREFECTURE_SAMPLED_ID.values()))

    print(f"Dataset saved with {len(dataset)} entries.")

    json.dump(
        dataset,
        f,
        indent=2,
        ensure_ascii=False,
    )
