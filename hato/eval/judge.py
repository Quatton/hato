import asyncio
import json
import os
from typing import Dict, Set, Tuple
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from dotenv import load_dotenv

from hato.model.result import Results, ResultEntry

load_dotenv()

oai = AsyncAzureOpenAI(api_version="2025-01-01-preview")

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "out/output.json")
VERIFIED_OUTPUT_PATH = os.getenv("VERIFIED_OUTPUT_PATH", "out/verified_output.json")
CACHE_OUTPUT_PATH = os.getenv("CACHE_OUTPUT_PATH", "out/cache.json")


class JudgmentResult(BaseModel):
    ward: bool
    town: bool


class VerifiedResultEntry(BaseModel):
    index: int
    panoid: str
    answer: dict
    actual_address: dict
    judgment: JudgmentResult


class VerifiedResults(BaseModel):
    results: list[VerifiedResultEntry]


class LLMVerificationResponse(BaseModel):
    ward_match: bool
    town_match: bool
    wouldve_matched_with: str | None = None
    wouldve_matched_with_town: str | None = None


def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing common variations."""
    if not text:
        return ""

    # Remove common prefixes/suffixes and normalize
    text = text.strip()
    text = text.replace("区", "").replace("町", "").replace("丁目", "")
    text = text.replace("-", "").replace("−", "").replace("ー", "")
    text = text.replace(" ", "").replace("　", "")
    return text.lower()


def exact_match_check(predicted: str, actual: str) -> bool:
    """Check if two location names match exactly after normalization."""
    if not predicted or not actual:
        return False

    normalized_predicted = normalize_text(predicted)
    normalized_actual = normalize_text(actual)

    return normalized_predicted == normalized_actual


async def llm_verify_location(
    predicted_ward: str, predicted_town: str | None, actual_ward: str, actual_town: str
) -> LLMVerificationResponse | None:
    """Use LLM to verify if predicted locations match actual locations."""
    try:
        response = await oai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a Tokyo geography expert. Compare predicted and actual Tokyo locations.

Consider these as matches:
- Different spellings/romanizations of the same place
- Alternative names for the same area
- Administrative boundary changes
- Common abbreviations or full names

For wards (区), be strict - they should refer to the same administrative ward.
For towns/districts, be more flexible - consider neighboring areas or historical name changes.

If the result is not matched, provide a "would have matched with" field indicating what the model should have predicted instead.
E.g. "actual: 渋谷区, predicted: Meguro, would have matched with would be "Shibuya".

Respond with whether each location type matches and provide reasoning.""",
                },
                {
                    "role": "user",
                    "content": f"""Compare these Tokyo locations:

Predicted:
- Ward: {predicted_ward}
- Town: {predicted_town or "Not specified"}

Actual:
- Ward: {actual_ward}
- Town: {actual_town or "Not specified"}

Do they match?""",
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "verification_result",
                    "schema": LLMVerificationResponse.model_json_schema(),
                },
            },
        )

        return LLMVerificationResponse.model_validate_json(
            response.choices[0].message.content or ""
        )

    except Exception as e:
        print(f"Error in LLM verification: {e}")
        return None


def load_cache() -> Dict[str, Set[str]]:
    """Load existing cache from file."""
    if os.path.exists(CACHE_OUTPUT_PATH):
        try:
            with open(CACHE_OUTPUT_PATH, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                # Convert lists back to sets
                return {k: set(v) for k, v in cache_data.items()}
        except Exception as e:
            print(f"Error loading cache: {e}")

    return {}


def save_cache(cache: Dict[str, Set[str]]):
    """Save cache to file."""
    try:
        # Convert sets to lists for JSON serialization
        cache_data = {k: list(v) for k, v in cache.items()}
        with open(CACHE_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving cache: {e}")


async def judge_single_result(
    result_entry: ResultEntry, cache: Dict[str, Set[str]]
) -> Tuple[JudgmentResult, Dict[str, Set[str]]]:
    """Judge a single result entry and return the judgment along with updated cache."""
    predicted_ward = result_entry.answer.ward
    predicted_town = result_entry.answer.town
    actual_ward = (
        result_entry.actual_address.city.name
    )  # In Tokyo, city is actually the ward
    actual_town = result_entry.actual_address.oaza.name

    # Check for exact matches first
    ward_match = exact_match_check(predicted_ward, actual_ward)
    town_match = (
        exact_match_check(predicted_town, actual_town) if predicted_town else False
    )

    # If exact match found, no need for LLM verification
    if ward_match and (town_match or not predicted_town):
        return JudgmentResult(ward=ward_match, town=town_match), cache

    # Check cache for ward and town separately
    acceptable_wards = cache.get(actual_ward, set())
    acceptable_towns = cache.get(actual_town, set()) if actual_town else set()

    # Check if prediction is in cache
    if predicted_ward in acceptable_wards:
        ward_match = True

    if predicted_town and predicted_town in acceptable_towns:
        town_match = True
    elif not predicted_town:
        town_match = False

    # If both found in cache, no need for LLM
    if ward_match and (town_match or not predicted_town):
        return JudgmentResult(ward=ward_match, town=town_match), cache

    # Use LLM verification
    print(
        f"Using LLM verification for: {predicted_ward}/{predicted_town} vs {actual_ward}/{actual_town}"
    )
    llm_result = await llm_verify_location(
        predicted_ward, predicted_town, actual_ward, actual_town
    )

    # Handle None case (LLM verification failed)
    if llm_result is None:
        print("  LLM verification failed, assuming no match")
        return JudgmentResult(ward=ward_match, town=town_match), cache

    # Update cache with LLM results
    if llm_result.ward_match:
        if actual_ward not in cache:
            cache[actual_ward] = set()
        cache[actual_ward].add(predicted_ward)
        ward_match = True
    else:
        # Add the correct answer to cache for future reference
        if llm_result.wouldve_matched_with:
            if actual_ward not in cache:
                cache[actual_ward] = set()
            cache[actual_ward].add(llm_result.wouldve_matched_with)

    if llm_result.town_match and predicted_town and actual_town:
        if actual_town not in cache:
            cache[actual_town] = set()
        cache[actual_town].add(predicted_town)
        town_match = True
    else:
        # Add the correct town answer to cache for future reference
        if llm_result.wouldve_matched_with_town and actual_town:
            if actual_town not in cache:
                cache[actual_town] = set()
            cache[actual_town].add(llm_result.wouldve_matched_with_town)

    return JudgmentResult(ward=ward_match, town=town_match), cache


async def main():
    """Main function to judge all results."""
    # Load results
    if not os.path.exists(OUTPUT_PATH):
        print(f"Results file {OUTPUT_PATH} not found!")
        return

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        results_data = Results.model_validate(json.load(f))

    # Load cache
    cache = load_cache()
    print(f"Loaded cache with {len(cache)} entries")

    # Load existing verified results if they exist
    existing_verified = {}
    if os.path.exists(VERIFIED_OUTPUT_PATH):
        try:
            with open(VERIFIED_OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing_data = VerifiedResults.model_validate(json.load(f))
                existing_verified = {
                    result.index: result for result in existing_data.results
                }
                print(f"Loaded {len(existing_verified)} existing verified results")
        except Exception as e:
            print(f"Error loading existing verified results: {e}")

    verified_results = []
    total_results = len(results_data.results)

    print(f"Processing {total_results} results...")

    for i, result_entry in enumerate(results_data.results):
        print(f"Processing result {i + 1}/{total_results} (index {result_entry.index})")

        # Check if already processed
        if result_entry.index in existing_verified:
            verified_results.append(existing_verified[result_entry.index])
            print("  Using existing result")
            continue

        # Judge the result
        judgment, cache = await judge_single_result(result_entry, cache)

        verified_result = VerifiedResultEntry(
            index=result_entry.index,
            panoid=result_entry.panoid,
            answer=result_entry.answer.model_dump(),
            actual_address=result_entry.actual_address.model_dump(),
            judgment=judgment,
        )

        verified_results.append(verified_result)

        print(f"  Ward match: {judgment.ward}, Town match: {judgment.town}")

        # Save intermediate results every 10 items
        if (i + 1) % 10 == 0:
            verified_results.sort(key=lambda x: x.index)
            output_data = VerifiedResults(results=verified_results)
            with open(VERIFIED_OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(output_data.model_dump(), f, indent=2, ensure_ascii=False)

            save_cache(cache)
            print(f"  Saved intermediate results ({i + 1}/{total_results})")

    # Sort and save final results
    verified_results.sort(key=lambda x: x.index)
    output_data = VerifiedResults(results=verified_results)

    with open(VERIFIED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data.model_dump(), f, indent=2, ensure_ascii=False)

    save_cache(cache)

    # Calculate statistics
    ward_correct = sum(1 for r in verified_results if r.judgment.ward)
    town_correct = sum(1 for r in verified_results if r.judgment.town)

    print("\nJudgment complete!")
    print(f"Total results: {len(verified_results)}")
    print(
        f"Ward accuracy: {ward_correct}/{len(verified_results)} ({ward_correct / len(verified_results) * 100:.1f}%)"
    )
    print(
        f"Town accuracy: {town_correct}/{len(verified_results)} ({town_correct / len(verified_results) * 100:.1f}%)"
    )
    print(f"Results saved to: {VERIFIED_OUTPUT_PATH}")
    print(f"Cache saved to: {CACHE_OUTPUT_PATH}")
    print(f"Cache now contains {len(cache)} entries")


if __name__ == "__main__":
    asyncio.run(main())
