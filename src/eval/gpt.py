import asyncio
import base64
import json
import os
from typing import List
from openai import AsyncAzureOpenAI

from dotenv import load_dotenv

from model.answer import Answer
from model.dataset import PanoAddress, PanoAddressDataset
from model.result import ResultEntry, Results
from prompt.template import (
    system_prompt,
    observation_start,
    observation_end,
    reasoning_start,
    reasoning_end,
    ward_start,
    ward_end,
    town_start,
    town_end,
)

load_dotenv()

oai = AsyncAzureOpenAI(api_version="2025-04-01-preview")

DATASET_PATH = os.getenv("DATASET_PATH", "")
IMAGE_PATH = os.getenv("IMAGE_PATH", "")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "out/output_gpt.json")


async def process_image(pano_data: PanoAddress, index: int) -> ResultEntry | None:
    """Process a single image and return the result entry."""
    image_file = f"{index}_{pano_data.panoId}.jpg"
    image_path = os.path.join(IMAGE_PATH, image_file)
    real_ward = pano_data.address.city.name
    real_town = pano_data.address.oaza.name

    if not os.path.exists(image_path):
        print(f"Image file {image_file} does not exist in {IMAGE_PATH}.")
        return None

    print(f"Processing image {index + 1}: {image_file}")

    with open(image_path, "rb") as img_file:
        image_data = img_file.read()

    image_data_base64 = base64.b64encode(image_data).decode("utf-8")

    try:
        response = await oai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    {system_prompt}
                    
                    Please be concise in your reasoning stage but make descriptive observations.
                    Breaking news: Please pretend you did just that and finally got the answer: 
                    {ward_start}{real_ward}{ward_end}{town_start}{real_town}{town_end}
                    (But format the answer in English with captialized first letters. Example: <ward>Setagaya</ward><town>Daita</town>)
                    """,
                    "name": "system",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data_base64}"
                            },
                        }
                    ],
                },
            ],
            max_tokens=4096,
        )

        if not response.choices:
            print(f"No choices returned for image {image_file}")
            return None

        choice = response.choices[0]
        if not choice.message or not choice.message.content:
            print(f"No content in response for image {image_file}")
            return None

        content = choice.message.content
        print(f"Response content for image {image_file}:\n{content}")

        assert response.usage is not None, "Response usage is None"

        result_entry = ResultEntry(
            index=index,
            panoid=pano_data.panoId,
            actual_address=pano_data.address,
            answer=Answer(
                raw=content,
                observation=(
                    content.split(observation_start)[1]
                    .split(observation_end)[0]
                    .strip()
                ),
                reasoning=(
                    content.split(reasoning_start)[1].split(reasoning_end)[0].strip()
                ),
                ward=(content.split(ward_start)[1].split(ward_end)[0].strip()),
                town=(
                    content.split(town_start)[1].split(town_end)[0].strip()
                    if town_start in content
                    else None
                ),
            ),
        )

        return result_entry

    except Exception as e:
        print(f"  Error processing image {image_file}: {e}")
        return None


async def process_batch(batch_items) -> list[ResultEntry]:
    """Process a batch of items concurrently."""
    tasks: List[asyncio.Task] = []
    for pano_data, index in batch_items:
        task = asyncio.create_task(process_image(pano_data, index))
        tasks.append(task)
        # await asyncio.sleep(3)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None results and exceptions, extract results and token counts
    valid_results = []

    for result in results:
        if result is not None and not isinstance(result, BaseException):
            valid_results.append(result)
        elif isinstance(result, Exception):
            print(f"  Error in batch processing: {result}")

    return valid_results


async def main():
    with open(DATASET_PATH) as f:
        dataset = PanoAddressDataset.model_validate(json.load(f)).customCoordinates

    # Load existing results if output file exists
    existing_results = []
    processed_indices = set()

    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r") as f:
                existing_data = json.load(f)
                existing_results_data = Results.model_validate(existing_data)
                existing_results = existing_results_data.results
                processed_indices = {result.index for result in existing_results}
                print(
                    f"Loaded {len(existing_results)} existing results from {OUTPUT_PATH}"
                )
        except Exception as e:
            print(f"Error loading existing results: {e}")
            print("Starting fresh...")

    print(f"Processing {len(dataset)} images...")
    print(f"Already processed: {len(processed_indices)} images")

    results = existing_results.copy()

    # Prepare items to process
    items_to_process = []
    for index, pano_data in enumerate(dataset):
        if index not in processed_indices:
            items_to_process.append((pano_data, index))

    print(f"New items to process: {len(items_to_process)}")

    batch_size = 20
    for i in range(0, len(items_to_process), batch_size):
        batch = items_to_process[i : i + batch_size]
        print(
            f"\nProcessing batch {i // batch_size + 1}/{(len(items_to_process) + batch_size - 1) // batch_size} ({len(batch)} items)"
        )

        batch_results = await process_batch(batch)
        results.extend(batch_results)

        # Save intermediate results after each batch
        results.sort(key=lambda x: x.index)
        output_data = Results(results=results)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(output_data.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"  Batch complete. Total results so far: {len(results)}")

    # Sort results by index before saving
    results.sort(key=lambda x: x.index)

    # Save results to JSON file
    output_data = Results(results=results)
    output_path = OUTPUT_PATH

    with open(output_path, "w") as f:
        json.dump(output_data.model_dump(), f, indent=2, ensure_ascii=False)

    print("\nProcessing complete!")
    print(f"Total results: {len(results)}/{len(dataset)} images")
    print(f"Newly processed: {len(results) - len(existing_results)} images")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
