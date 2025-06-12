# %%
from unsloth import FastVisionModel
import torch

max_seq_length = 2048
lora_rank = 8  # The example uses 32 so let's say 8

# %%
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    max_seq_length=max_seq_length,
)

pmodel = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,  # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,  # False if not finetuning MLP layers
    r=lora_rank,
    lora_alpha=lora_rank * 2,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
    use_gradient_checkpointing=True,
)

# %%

if __name__ == "__main__":
    from PIL import Image
    from prompt.template import system_prompt
    from transformers.generation.streamers import TextStreamer

    FastVisionModel.for_inference(model)  # Enable for inference!

    image = Image.open("datasets/out/0_C_SxOIfPf7cF9fVPhli4fA.jpg")

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
            ],
        },
    ]

    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(
        images=[image],
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=max_seq_length,
        use_cache=True,
        temperature=0.5,
    )
    # %%
