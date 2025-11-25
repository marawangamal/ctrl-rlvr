# train_grpo.py: https://github.com/kossisoroyce/train_grpo.py
# CUDA_VISIBLE_DEVICES=0 MODEL_NAME=allenai/tulu-2-7b RUN_NAME=tulu2-7b-gsm8k-uncons python train_gsm8k.py
import argparse
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    """Extracts the answer from XML-formatted text."""
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        logger.warning("Failed to extract answer from XML format.")
        return ""


def extract_hash_answer(text: str) -> str | None:
    """Extracts the answer from a hash-formatted string."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# Configurable one-shot prompting
def get_gsm8k_questions(split="train", use_one_shot=False) -> Dataset:
    """Loads and prepares the GSM8K dataset with optional one-shot prompting."""
    try:
        data = load_dataset("openai/gsm8k", "main")[split]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    def format_example(x):
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}]
        if use_one_shot:
            prompt.extend(
                [
                    {
                        "role": "user",
                        "content": "What is the largest single-digit prime number?",
                    },
                    {
                        "role": "assistant",
                        "content": XML_COT_FORMAT.format(
                            reasoning="9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
                            answer="7",
                        ),
                    },
                ]
            )
        prompt.append({"role": "user", "content": x["question"]})
        return {"prompt": prompt, "answer": extract_hash_answer(x["answer"])}

    return data.map(format_example)


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Calculates reward based on correctness of the response."""
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    logger.info(
        f"Question:\n{q}\nAnswer:\n{answer[0]}\nResponse:\n{responses[0]}\nExtracted:\n{extracted_responses[0]}"
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """Calculates reward if the extracted response is a digit."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def format_reward_func(completions, strict=False, **kwargs) -> list[float]:
    """Calculates reward based on XML formatting."""
    pattern = (
        r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        if strict
        else r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    )
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Calculates reward based on XML tag counts."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def count_xml(text) -> float:
    """Counts XML tags and penalizes extra content."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def get_tokenizer(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Check if tokenizer is already a chat tokenizer
    if tokenizer.chat_template is not None:
        return tokenizer

    # Very dumb "chat" template: just concatenates user + assistant messages.
    # You can adjust this to something more realistic if you want.
    tokenizer.chat_template = """{% for message in messages %}
    {% if message['role'] == 'user' %}
    User: {{ message['content'] }}
    {% elif message['role'] == 'assistant' %}
    Assistant: {{ message['content'] }}
    {% endif %}
    {% endfor %}Assistant:"""

    # "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    return tokenizer


def fmt_str(s):
    if s is None:
        return s
    return s.replace("-", "_").replace("/", "_")


def create_run_name(dct):
    return fmt_str(f"_".join([f"{k}{v}" for k, v in dct.items()])).lower()


ARGS_NAME_MAP = {
    # Models
    "allenai/tulu-2-7b": "t27b",
    "ctrlg/hmm_gpt2-large_common-gen_4096": "g2lh4096",
    "gpt2-large": "g2l",
    # Datasets
    "trl-lib/DeepMath-103K": "dm103k",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train GRPO model with a specified model name."
    )
    parser.add_argument(
        "--model",
        type=str,
        # default="Qwen/Qwen2-0.5B-Instruct",
        default="ctrlg/gpt2-large_common-gen",  # use even smaller model: "erwanf/gpt2-mini" or "gpt2-small"
        help="Name of the model to train (e.g., 'tulu', 'google/gemma-2b', etc.)",
    )
    parser.add_argument(
        "--hmm",
        type=str,
        default=None,
        # f'ctrlg/hmm_gpt2-large_common-gen_4096' # alternatively 'ctrlg/hmm_gpt2-large_common-gen_32768' for better quality
        help="Name of the HMM model to use (e.g., 'ctrl-g/gpt2', 'gwenweng/gemma', 'gwenweng/gpt2', etc.)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Use at most this many training samples (before GRPO).",
    )
    parser.add_argument(
        "--constraint_mode",
        type=str,
        default="suffix",
        choices=["suffix", "keyphrase", "both"],
        help="Constraint mode to use (e.g., 'suffix', 'keyphrase', 'both').",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum length of the prompt.",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=6,
        help="Maximum number of new tokens to generate.",
    )
    args = parser.parse_args()

    # Model setup
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2" if device == "cuda" else None,
        # device_map="auto",
    )

    train_dataset = get_gsm8k_questions(use_one_shot=True)
    eval_dataset = get_gsm8k_questions(split="test", use_one_shot=True)

    # Print dataset statistics
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    tokenizer = get_tokenizer(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # PEFT config (optional)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    run_name = f"crl2_" + create_run_name(
        {
            "m": ARGS_NAME_MAP.get(args.model, args.model),
            "d": "gsm8k",
            "hm": ARGS_NAME_MAP.get(args.hmm, args.hmm),
            "mx": args.max_new_tokens,
            "mn": args.min_new_tokens,
            "n": args.max_samples,
            "c": args.constraint_mode,
        }
    )

    # Training config
    training_args = GRPOConfig(
        output_dir=f"./experiments/{run_name}",
        run_name=args.run_name,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        per_device_train_batch_size=8,  # Increased from 1
        gradient_accumulation_steps=1,  # Reduced from 4
        num_generations=8,  # Reduced from 16
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=1,
        max_grad_norm=0.1,
        # Save
        save_steps=100,
        save_total_limit=3,
        push_to_hub=True,
        # Logging
        logging_steps=10,
        report_to="wandb",
        log_on_each_node=False,
        # Eval
        eval_steps=3000,
        eval_strategy="steps",
    )

    # Trainer setup
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            format_reward_func,  # No need for lambda, just pass the function
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # peft_config=peft_config  # Uncomment if PEFT is working for you
    )

    resume_from_checkpoint = True if len(os.listdir(OUTPUT_DIR)) > 0 else False
    print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.push_to_hub()
