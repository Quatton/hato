# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

from open_r1.trainer import VLMGRPOTrainer, GRPOConfig

from transformers.utils import logging
from transformers import AutoTokenizer

from prompt.template import system_prompt

from openai import OpenAI

logger = logging.get_logger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-proj-1234567890"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
)

from open_r1.qwen2_5vl_monkey_patch import (
    monkey_patch_qwen2_5vl_flash_attn,
    monkey_patch_qwen2_5vl_forward,
    monkey_patch_torch_load,
)

monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_torch_load()

tokenizer = None


def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """

    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'"
        },
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={
            "help": "Maximum number of anyres blocks for the image (for InternVL)"
        },
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={"help": "Choose reward method: 'default', 'mcp', ..."},
    )
    task_type: Optional[str] = field(
        default=None,
        metadata={"help": "Choose task type: 'default', 'gui', ..."},
    )
    is_reward_customized_from_vlm_module: bool = field(
        default=False,
        metadata={"help": "Whether to use a customized reward from vlm module"},
    )


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    # TODO: HatoVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")


def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template(
        task_type=script_args.task_type
    )

    # Get reward functions
    if script_args.is_reward_customized_from_vlm_module:
        reward_funcs = [
            vlm_module_cls.select_reward_func(func, script_args.task_type)
            for func in script_args.reward_funcs
        ]
    else:
        raise ValueError("Not supported")
    # Load the JSONL datasets
    import json
    from datasets import Dataset

    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(data_files), (
            f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"
        )

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    all_data = []
    for data_file, image_folder, accu_reward_method in zip(
        data_files, image_folders, accu_reward_methods
    ):
        with open(data_file, "r") as f:
            items = json.load(f).results
            for item in items:
                item["image_path"] = os.path.join(
                    image_folder, f"{item['id']}_{item['panoId']}.jpg"
                )
                item["accu_reward_method"] = item.get(
                    "accu_reward_method", accu_reward_method
                )  # if accu_reward_method is in the data jsonl, use the value in the data jsonl, otherwise use the defined value
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        return {
            "image_path": example["image_path"],
            "answer": example["solution"],
            "accu_reward_method": example["accu_reward_method"],
            "prompt": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [{"type": "image", "text": None}],
                },
            ],
        }

    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    # Split dataset for validation if requested
    splits = {"train": dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits["train"] = train_val_split["train"]
        splits["validation"] = train_val_split["test"]

    # Select trainer class based on vlm_trainer argument
    trainer_cls = VLMGRPOTrainer
    print("using trainer:", trainer_cls.__name__)
    initialize_tokenizer(model_args.model_name_or_path)
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits["train"],
        eval_dataset=splits.get("validation")
        if training_args.eval_strategy != "no"
        else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
