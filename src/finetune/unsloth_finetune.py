# Need for unsloth
from unsloth import FastVisionModel
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/mistral-7b-bnb-4bit",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = False,
    finetune_language_layers = True,
    fientune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rsolar = False,
    loftq_config = None,
)

# Hugging face dataset lib
from datasets import load_dataset

# Hugging face dataset here
dataset = load_dataset("dataset")

# Place instruction here
instruction = " "
def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["caption"]} ]
        },
    ]
    return { "messages" : conversation }

converted_dataset = [convert_to_conversation(sample) for sample in dataset]

FastVisionModel.for_trining(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_caollator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    )
)