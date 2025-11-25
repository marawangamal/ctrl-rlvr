from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM
from datasets import load_dataset
import torch


# Configurable one-shot prompting
def get_gsm8k_questions(split="train", use_one_shot=False):
    """Loads and prepares the GSM8K dataset with optional one-shot prompting."""
    data = load_dataset("openai/gsm8k", "main")[split]

    def format_example(x):
        return {"prompt": x["question"], "completion": x["answer"]}

    return data.map(format_example)


model = AutoModelForCausalLM.from_pretrained(
    "allenai/tulu-2-7b", dtype=torch.bfloat16, device_map="auto"
)

run_name = "t27b-gsm8k-sft"
output_dir = f"experiments-sft/{run_name}"
trainer = SFTTrainer(
    model=model,
    train_dataset=get_gsm8k_questions(use_one_shot=True, split="train"),
    args=SFTConfig(
        # per_device_train_batch_size=1,
        bf16=True,
        per_device_train_batch_size=2,
        run_name=run_name,
        output_dir=output_dir,
        # logging
        report_to="wandb",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        push_to_hub=True,
        num_train_epochs=1,
    ),
)
trainer.train()
trainer.push_to_hub()
