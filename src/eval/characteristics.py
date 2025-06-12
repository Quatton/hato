import json
import pyperclip
import os

FILE = "out/eval_results_gpt41_jp_5k.json"
OUT_FILE = "out/prefecture_characteristics.json"
TEMPFILE = "TEMPFILE.txt"  # Define a fixed temporary file in the current directory

# Load existing characteristics if the file exists
try:
    with open(OUT_FILE, "r", encoding="utf-8") as f:
        prefecture_characteristics = json.load(f)
except FileNotFoundError:
    prefecture_characteristics = {}

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


with open(FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

# combine observations by prefecture
prefecture_results: dict[str, list[str]] = {pref: [] for pref in PREFECTURE}
for result in results:
    prefecture = result["prefecture"]
    if prefecture in prefecture_results:
        prefecture_results[prefecture].append(result["observation"])

for pref, obs in prefecture_results.items():
    print(f"{pref}: {len(obs)} observations")

PROMPT_TEMPLATE = """\
  Following is an observation from a prefecture {prefecture} in Japan. Please summarize the overall characteristics of this prefecture based on the observation. Format it in a code block for easy copy and paste.
  
  Observation: {observation}
""".strip()

for pref, obs in prefecture_results.items():
    if pref in prefecture_characteristics:
        print(f"Skipping {pref} as it is already processed.")
        continue

    print(f"\n{pref} ({len(obs)} observations):")
    observation = "\n".join(obs)
    prompt = PROMPT_TEMPLATE.format(prefecture=pref, observation=observation)
    pyperclip.copy(prompt)  # Copy the prompt to the clipboard
    print(
        "Prompt is automatically injected in your system clipboard. Please paste it in your browser to see the result. When you are done, paste the result into the temporary file that will open. Close the file to continue..."
    )

    # Create or overwrite the temporary file for pasting the result
    with open(TEMPFILE, "w", encoding="utf-8") as tmp_file:
        tmp_file.write("Paste your result here and save the file.\n")

    # Open the temporary file in the default editor
    os.system(f"{os.getenv('EDITOR', 'code --wait')} {TEMPFILE}")

    # Read the result from the temporary file
    with open(TEMPFILE, "r", encoding="utf-8") as tmp_file:
        result = tmp_file.read().strip()

    prefecture_characteristics[pref] = result

    # Save the updated results incrementally
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(prefecture_characteristics, f, ensure_ascii=False, indent=2)

    # Optionally delete the temporary file after processing
    os.remove(TEMPFILE)
