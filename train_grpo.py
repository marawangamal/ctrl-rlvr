# train_grpo.py
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
import wandb


def fmt_str(s):
    return s.lower().split("/")[-1].replace("-", "_")


if __name__ == "__main__":
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    dataset_name = "trl-lib/DeepMath-103K"

    dataset = load_dataset(dataset_name, split="train")
    run_name = f"grpo_model_{fmt_str(model_name)}_dataset_{fmt_str(dataset_name)}"
    wandb.init(project="ctrl-grpo", name=run_name)
    training_args = GRPOConfig(
        report_to="wandb",
        per_device_train_batch_size=32,
        output_dir=f"./experiments/{run_name}",
        logging_steps=10,
        run_name=run_name,
    )
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()
