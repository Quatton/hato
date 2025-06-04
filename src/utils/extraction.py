import re


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
  

def extract_info(text):
    observation_match = re.search(rf"<observation>\s*(.*?)\s*</observation>", text, re.DOTALL)
    reasoning_match = re.search(rf"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL)
    ward_match = re.search(rf"<ward>\s*(.*?)\s*</ward>", text)
    town_match = re.search(rf"<town>\s*(.*?)\s*</town>", text)

    observation = observation_match.group(1).strip() if observation_match else None
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    ward = ward_match.group(1).strip() if ward_match else None
    town = town_match.group(1).strip() if town_match else None

    answer = {}
    if ward is not None:
      answer["ward"] = ward
    if town is not None:
      answer["town"] = town

    return {
        "observation": observation,
        "reasoning": reasoning,
        "answer": answer
    }