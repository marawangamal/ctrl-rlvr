import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
from peft import LoraConfig, get_peft_model
from transformers import LogitsProcessorList
from datasets import load_dataset
import ctrlg
from huggingface_hub import snapshot_download

##############################################################################
#                                                                            #
#                                DATA LOADING                               #
#                                                                            #
##############################################################################


gsm8k = load_dataset("openai/gsm8k", "main")


def extract_answer_gsm8k(text):
    """Extract the numerical answer from the text."""
    try:
        # Basic extraction - parse after ####
        if "####" in text:
            answer_part = text.split("####")[1].strip()
            # Extract the first number
            for word in answer_part.split():
                word = word.strip(".,")
                if word.isdigit() or (word[0] == "-" and word[1:].isdigit()):
                    return int(word)
    except:
        pass
    return None


def load_gsm8k(n=None):

    TEMPLATE_GSM8K = """
    You are a helpful assistant that solves grade-school math word problems.
    For each problem:
    1. Think step by step and show your reasoning.
    2. On the last line, write the final answer in the form: #### <number>

    Here is an example:

    Question: A bookstore sold 18 books in the morning and 27 books in the afternoon. How many books did it sell in total that day?
    Answer:
    First, add the books sold in the morning and the afternoon.
    18 + 27 = 45.
    So the bookstore sold 45 books in total.
    #### 45

    Now solve this problem:

    Question: {question}
    Answer:
    """

    data = gsm8k["train"]
    if n is not None:
        data = data.select(range(n))

    prefix = ""

    formatted = []
    for item in data:
        problem = item["question"]
        solution = item["answer"]  # GSM8K solutions are full step-by-step
        formatted.append(
            {
                "problem": TEMPLATE_GSM8K.format(question=problem),
                "solution": solution,
                "answer": extract_answer_gsm8k(solution),
            }
        )
    return formatted


def compute_reward(response, correct_answer):
    """Compute reward based on correctness."""
    extracted = extract_answer_gsm8k(response)
    if extracted is not None and extracted == correct_answer:
        return 1.0  # Correct answer
    return 0.0  # Incorrect answer


##############################################################################
#                                                                            #
#                                DFA                                         #
#                                                                            #
##############################################################################


def get_dfa_model(
    hmm_model,
    tokenizer,
    prompt_ids,
    keyphrases=[[" "]],
    suffix_ids=None,
    min_new_tokens=5,
    max_new_tokens=32,
):

    vocab_size = len(tokenizer)

    # Prefix & suffix constraints
    prefix = ""  # generate text starting with nothing
    suffix = ".<|endoftext|>"  # generate text ending with '<|endoftext|>'; a suffix must end with the eos token
    prefix_ids = tokenizer.encode(prefix)
    if suffix_ids is None:
        suffix_ids = tokenizer.encode(suffix)

    # DFA Construction
    # ac_builder constructs a DFA representing the constraint that (at least)
    # one the patterns must appear; a pattern is a sequence of token ids
    ac_builder = ctrlg.AhoCorasickBuilder(vocab_size)

    dfa_graphs = []
    for keyphrase in keyphrases:
        patterns = [tokenizer.encode(x) for x in keyphrase]
        dfa_graphs.append(ac_builder.build(patterns))

    # taking the intersection of the DFAs, i.e., "logical and" of the constraints.
    # This function also minimizes the constructed DFA, which is mainly CPU-based operations;
    # Due to its pure python implemenation, DFA minimization can be slow for complex constraints
    dfa_graph = ctrlg.DFA_prod(dfa_graphs, mode="intersection")

    # compile the dfa_graph for efficient GPU execution
    dfa_model = ctrlg.DFAModel(dfa_graph, vocab_size).to(device)

    # Constraint logits processor
    constraint_logits_processor = ctrlg.ConstraintLogitsProcessor(
        hmm_model,
        dfa_model,
        min_new_tokens,
        max_new_tokens,
        prompt_ids,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
    )

    return constraint_logits_processor


##############################################################################
#                                                                            #
#                                TRAINING HELPERS                            #
#                                                                            #
##############################################################################


def get_logprobs(model, input_ids, attention_mask):
    """Get log probabilities for each token."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # Remove last position

        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Get the log probability of the actual next token
        target_ids = input_ids[:, 1:]  # Shift right
        gathered_logprobs = log_probs.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding
        masked_logprobs = gathered_logprobs * attention_mask[:, 1:].float()

        return masked_logprobs


def compute_advantages(rewards, group_rewards):
    """Compute advantages using the group baseline."""
    mean_reward = np.mean(group_rewards)
    std_reward = (
        np.std(group_rewards) + 1e-8
    )  # Add small epsilon to avoid division by zero

    # Normalize rewards
    advantages = (rewards - mean_reward) / std_reward
    return advantages


def grpo_loss(
    current_logprobs,
    old_logprobs,
    ref_logprobs,
    advantages,
    clip_epsilon=0.2,
    kl_coef=0.1,
):
    """Compute the GRPO loss."""
    # Compute probability ratio
    ratio = torch.exp(current_logprobs - old_logprobs)

    # Compute clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    ppo_obj = torch.min(ratio * advantages, clipped_ratio * advantages)

    # Compute KL divergence term
    kl_div = old_logprobs - ref_logprobs

    # Compute final loss
    loss = -ppo_obj.mean() + kl_coef * kl_div.mean()

    return loss


def train_grpo(
    model,
    ref_model,
    problem,
    correct_answer,
    solution,
    group_size=4,
    max_length=50,
    hmm_model=None,
    min_new_tokens=6,
    max_new_tokens=32,
    beam_size=32,
):
    """Train the model using GRPO on a single problem."""
    tokenizer.pad_token = tokenizer.eos_token

    # Generate responses for the group
    group_responses = []
    group_rewards = []

    # Create a batch of identical prompts
    prompt = f"Please solve the following math problem: {problem}"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    # Generate group responses
    for _ in range(group_size):
        with torch.no_grad():

            if hmm_model is None:
                # Regular generation
                output_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length + inputs.input_ids.shape[1],
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                )
            else:
                # Output constrained generation
                suffix_ids = tokenizer.encode(solution)
                constraint_logits_processor = get_dfa_model(
                    hmm_model=hmm_model,
                    tokenizer=tokenizer,
                    prompt_ids=inputs.input_ids[0].tolist(),
                    suffix_ids=suffix_ids,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                )
                output_ids = model.generate(
                    input_ids=inputs.input_ids,
                    do_sample=False,
                    length_penalty=0.2,
                    num_beams=beam_size,
                    num_return_sequences=beam_size,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    logits_processor=LogitsProcessorList([constraint_logits_processor]),
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Get the generated response
            response = tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            reward = compute_reward(response, correct_answer)

            group_responses.append(response)
            group_rewards.append(reward)

    # Compute advantages
    advantages = compute_advantages(group_rewards, group_rewards)

    # Optimize the model for each response
    optimizer = Adam(model.parameters(), lr=1e-5)

    for i in range(group_size):
        # Get the full sequence
        full_sequence = tokenizer(prompt + group_responses[i], return_tensors="pt").to(
            device
        )

        # Get old logprobs
        old_logprobs = get_logprobs(
            model, full_sequence.input_ids, full_sequence.attention_mask
        )

        # Get reference logprobs
        ref_logprobs = get_logprobs(
            ref_model, full_sequence.input_ids, full_sequence.attention_mask
        )

        # Forward pass
        outputs = model(
            input_ids=full_sequence.input_ids,
            attention_mask=full_sequence.attention_mask,
        )
        logits = outputs.logits[:, :-1, :]

        # Compute new logprobs
        log_probs = F.log_softmax(logits, dim=-1)
        target_ids = full_sequence.input_ids[:, 1:]
        current_logprobs = log_probs.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding
        mask = full_sequence.attention_mask[:, 1:].float()
        current_logprobs = current_logprobs * mask

        # Compute loss
        advantage = torch.tensor([advantages[i]]).to(device).expand_as(current_logprobs)
        loss = grpo_loss(current_logprobs, old_logprobs, ref_logprobs, advantage)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(group_rewards)


def evaluate_model(model, math_problems, num_samples=10):
    """Evaluate the model on a subset of math problems."""
    correct = 0
    samples = random.sample(math_problems, min(num_samples, len(math_problems)))

    for problem in samples:
        prompt = f"Please solve the following math problem: {problem['problem']}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                max_length=50 + inputs.input_ids.shape[1],
                pad_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            reward = compute_reward(response, problem["answer"])
            correct += reward

    return correct / len(samples)


def test_model(
    model,
    test_problems,
    tokenizer,
    device,
    hmm_model=None,
    max_new_tokens=32,
    min_new_tokens=6,
    beam_size=32,
):

    rewards = []
    print_indexes = np.linspace(0, len(test_problems), 5).astype(int)
    for i, problem in enumerate(test_problems):
        prompt = f"Please solve the following math problem: {problem['problem']}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            if hmm_model is None:
                output_ids = model.generate(
                    inputs.input_ids,
                    max_length=50 + inputs.input_ids.shape[1],
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                suffix_ids = tokenizer.encode(problem["solution"])
                constraint_logits_processor = get_dfa_model(
                    hmm_model=hmm_model,
                    tokenizer=tokenizer,
                    prompt_ids=inputs.input_ids[0].tolist(),
                    suffix_ids=suffix_ids,
                )
                output_ids = model.generate(
                    input_ids=inputs.input_ids,
                    do_sample=False,
                    length_penalty=0.2,
                    num_beams=beam_size,
                    num_return_sequences=beam_size,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    logits_processor=LogitsProcessorList([constraint_logits_processor]),
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=False
            )
            reward = compute_reward(response, problem["answer"])
            rewards.append(reward)

            if i in print_indexes:
                print(f"[{i}/{len(test_problems)}] Problem: {problem['problem']}")
                print(f"[{i}/{len(test_problems)}] Model's response: {repr(response)}")
                print(f"[{i}/{len(test_problems)}] Correct answer: {problem['answer']}")
                print(f"[{i}/{len(test_problems)}] Reward: {reward}")
                print("-" * 50)

    avg_reward = np.mean(rewards)
    return avg_reward


if __name__ == "__main__":
    # Load model and tokenizer

    # HPs
    max_new_tokens = 256
    min_new_tokens = 6
    beam_size = 32
    epochs = 3
    num_problems = None  # i.e. use all problems
    use_hmm_train = False
    use_hmm_test = False

    # Main
    # =========================== Main (gemma) ===========================
    # model_name = "google/gemma-2b"
    # hmm_model_path = snapshot_download(
    #     repo_id="gwenweng/gemma", local_dir="models/gemma"
    # )
    # hmm_model = ctrlg.HMM.from_pretrained(hmm_model_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    # # GEMMA LoRA config
    # config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, config)

    # =========================== Main2 (Tulu) ===========================
    model_name = "allenai/tulu-2-7b"
    hmm_model_path = "ctrlg/hmm_tulu2-7b_writing-prompts_32768"
    hmm_model = ctrlg.HMM.from_pretrained(hmm_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    # Tulu LoRA config
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # =========================== Debug ===========================
    # model_name = "gpt2-large"
    # hmm_model_path = "ctrlg/hmm_gpt2-large_common-gen_4096"

    # hmm_model = ctrlg.HMM.from_pretrained(hmm_model_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    # # GPT-2 LoRA config
    # config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules=[
    #         "c_attn",  # QKV projection
    #         "c_proj",  # output projection
    #     ],
    # )
    # model = get_peft_model(model, config)
    # =========================== Debug ===========================

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hmm_model.to(device)
    model.to(device)
    print(f"Using device: {device}")
    print(
        f"HMM model (params): {sum(p.numel() for p in hmm_model.parameters()) / 1e6:.2f}M"
    )
    print(
        f"Base model (trainable params): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M"
    )

    def ref_model(*args, **kwargs):
        # π_ref with LoRA OFF (reference = base model)
        with model.disable_adapter():
            out_ref = model(*args, **kwargs)
        return out_ref

    ##############################################################################
    #                                                                            #
    #                                TRAINING LOOP                              #
    #                                                                            #
    ##############################################################################

    math_problems = load_gsm8k(num_problems)
    split_idx = int(len(math_problems) * 0.98)
    training_problems = math_problems[:split_idx]
    eval_problems = math_problems[split_idx:]
    performance_history = []

    # Test model before training
    avg_reward = test_model(
        model=model,
        test_problems=eval_problems,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        beam_size=beam_size,
    )
    print(f"Average reward before training: {avg_reward:.4f}")

    for epoch in range(epochs):
        epoch_rewards = []

        for problem in tqdm(training_problems, desc=f"Epoch {epoch+1}/{epochs}"):
            reward = train_grpo(
                model,
                ref_model,
                problem["problem"],
                problem["answer"],
                problem["solution"],
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
                hmm_model=hmm_model if use_hmm_train else None,
            )
            epoch_rewards.append(reward)

        # Evaluate model
        accuracy = evaluate_model(model, eval_problems)
        performance_history.append(accuracy)

        print(
            f"Epoch {epoch+1}/{epochs} - Average Reward: {np.mean(epoch_rewards):.4f}, Accuracy: {accuracy:.4f}"
        )

    test_model(
        model=model,
        test_problems=eval_problems,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        beam_size=beam_size,
        hmm_model=hmm_model if use_hmm_test else None,
    )
    print(f"Average reward after training: {avg_reward:.4f}")

    # Save model checkpoint
    model.save_pretrained(f"checkpoints/grpo_{model_name}_hmm{use_hmm_train}")
