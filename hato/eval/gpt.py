import asyncio
import base64
import json
import os
from openai import AsyncAzureOpenAI

from dotenv import load_dotenv

from hato.model.answer import Answer
from hato.model.dataset import PanoAddressDataset
from hato.model.result import ResultEntry, Results

load_dotenv()

oai = AsyncAzureOpenAI(api_version="2025-01-01-preview")

DATASET_PATH = os.getenv("DATASET_PATH")
IMAGE_PATH = os.getenv("IMAGE_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "out/output.json")


stats = {
    "input_tokens": 0,
    "output_tokens": 0,
}


async def process_image(pano_data, index):
    """Process a single image and return the result entry."""
    image_file = f"{index}_{pano_data.panoId}.jpg"
    image_path = os.path.join(IMAGE_PATH, image_file)

    if not os.path.exists(image_path):
        print(f"Image file {image_file} does not exist in {IMAGE_PATH}.")
        return None

    print(f"Processing image {index + 1}: {image_file}")

    with open(image_path, "rb") as img_file:
        image_data = img_file.read()

    image_data_base64 = base64.b64encode(image_data).decode("utf-8")

    try:
        response = await oai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Analyze this Tokyo street view image and identify the location.

Observe:
- Building types (residential/commercial), architectural era, density, height restrictions
- Vegetation type and abundance (native vs planted species)
- Road infrastructure: width, markings, materials, surface condition
- Municipal features: lamp styles, signage, utility poles (especially lamp colors)
- Landmarks or distinctive features, especially those unique to Tokyo
- Urban planning patterns: street layout, block size
- Topography: elevation, proximity to hills/water
- Geographical context: proximity to major roads, rivers, parks

Identify the Tokyo ward and town based on these characteristics.
Be specific and detailed in your analysis. Use strong verbs and adjectives to describe the visual elements without verbose language.

Response format:
- `observation`: Key visual details only
- `reasoning`: How these details indicate the specific location
- `ward`: Tokyo ward name
- `town`: Town/district name (if identifiable)
- `confidence`: Certainty level (0-1)""",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data_base64}",
                            },
                        }
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "reasoning_format",
                    "schema": Answer.model_json_schema(),
                },
            },
        )

        answer = Answer.model_validate_json(response.choices[0].message.content)

        print(f"  Ward: {answer.ward}, Town: {answer.town}")
        print(
            f"  Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}"
        )

        return {
            "result": ResultEntry(
                index=index,
                panoid=pano_data.panoId,
                answer=answer,
                actual_address=pano_data.address,
            ),
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    except Exception as e:
        print(f"  Error processing image {image_file}: {e}")
        return None


async def process_batch(batch_items):
    """Process a batch of items concurrently."""
    tasks = []
    for pano_data, index in batch_items:
        task = process_image(pano_data, index)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None results and exceptions, extract results and token counts
    valid_results = []
    batch_input_tokens = 0
    batch_output_tokens = 0

    for result in results:
        if result is not None and not isinstance(result, Exception):
            valid_results.append(result["result"])
            batch_input_tokens += result["input_tokens"]
            batch_output_tokens += result["output_tokens"]
        elif isinstance(result, Exception):
            print(f"  Error in batch processing: {result}")

    return valid_results, batch_input_tokens, batch_output_tokens


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

    # Process items in batches of 5
    batch_size = 10
    for i in range(0, len(items_to_process), batch_size):
        batch = items_to_process[i : i + batch_size]
        print(
            f"\nProcessing batch {i // batch_size + 1}/{(len(items_to_process) + batch_size - 1) // batch_size} ({len(batch)} items)"
        )

        batch_results, batch_input_tokens, batch_output_tokens = await process_batch(
            batch
        )
        results.extend(batch_results)

        # Update statistics synchronously after batch
        stats["input_tokens"] += batch_input_tokens
        stats["output_tokens"] += batch_output_tokens

        # Save intermediate results after each batch
        results.sort(key=lambda x: x.index)
        output_data = Results(results=results)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(output_data.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"  Batch complete. Total results so far: {len(results)}")
        print(
            f"  Batch tokens - Input: {batch_input_tokens}, Output: {batch_output_tokens}"
        )

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

    print(f"Total input tokens: {stats['input_tokens']}")
    print(f"Total output tokens: {stats['output_tokens']}")
    print(
        f"Estimated cost (2.5/M input + 10/M output): "
        f"${stats['input_tokens'] / 1000000 * 2.5 + stats['output_tokens'] / 1000000 * 10:.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
