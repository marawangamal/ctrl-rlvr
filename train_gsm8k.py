# train_grpo.py: https://github.com/kossisoroyce/train_grpo.py
# CUDA_VISIBLE_DEVICES=0 MODEL_NAME=allenai/tulu-2-7b RUN_NAME=tulu2-7b-gsm8k-uncons python train_gsm8k.py
import argparse
import re
import logging
from contextlib import nullcontext
import os
from typing import Any, Dict, List, Literal, Optional

# Third-party
import ctrlg
import torch
import wandb
from datasets import load_dataset, Dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
    prepare_multimodal_messages_vllm,
)
from trl.extras.profiling import profiling_context
from trl.models import unwrap_model_for_generation
from trl.rewards import accuracy_reward
from trl.trainer.utils import nanmax, nanmin, nanstd, pad

from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from peft import LoraConfig

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

##############################################################################
#                                                                            #
#                                CUSTOM GRPO TRAINER                         #
#                                                                            #
##############################################################################


class GRPOTrainerCustom(GRPOTrainer):
    def __init__(
        self,
        *args,
        logits_processor_fns: Optional[List] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logits_processor_fns = logits_processor_fns

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [
                [example.get("image")] if example.get("image") is not None else None
                for example in inputs
            ]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        # EDITED: pass inputs to _generate for goal conditioning
        (
            prompt_ids_list,
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts, inputs)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids, padding_value=self.pad_token_id, padding_side="left"
        )
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [
            torch.tensor(ids, device=device) for ids in completion_ids_list
        ]
        completion_mask = [
            torch.ones_like(ids, dtype=torch.long) for ids in completion_ids
        ]
        completion_ids = pad(
            completion_ids, padding_value=self.pad_token_id, padding_side="right"
        )
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [
                torch.tensor(logps, device=device)
                for logps in sampling_per_token_logps_list
            ]
            sampling_per_token_logps = pad(
                sampling_per_token_logps, padding_value=0.0, padding_side="right"
            )
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor(
                [ids[-1] not in eos_and_pad for ids in completion_ids_list],
                device=device,
            )
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat(
            [prompt_ids, completion_ids], dim=1
        )  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        num_images = (
            [len(img_list) for img_list in images] if images is not None else None
        )

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt},
                    self.processing_class,
                    **self.chat_template_kwargs,
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(
                images=images, text=prompts_text, padding=True, return_tensors="pt"
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {
                k: v
                for k, v in prompt_inputs.items()
                if k not in ["input_ids", "attention_mask"]
            }
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = (
                self.args.steps_per_generation * self.num_iterations
            )  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(
                    old_per_token_logps - sampling_per_token_logps
                )
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = (
                            self._get_per_token_logps_and_entropies(
                                self.model,
                                prompt_completion_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=batch_size,
                                num_images=num_images,
                                **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                            )
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(
            prompt_ids, skip_special_tokens=True
        )
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )

        # Apply weights to each reward function's output and sum
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = (
            advantages.clone()
        )  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(
                std_func_rewards
            )
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = (
                torch.mean(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            max_delta = (
                torch.max(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output

    def _generate(self, prompts: list, inputs: list[dict[str, torch.Tensor | Any]]):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # EDITED: pass inputs to _generate_single_turn for goal conditioning
        prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(
            prompts, inputs
        )

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor(
            [len(ids) for ids in completion_ids], device=device
        )
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = (
            agg_completion_lengths.sum()
        )  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                total_prompt_tokens + total_completion_tokens
            ).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor(
            [ids[-1] not in eos_and_pad for ids in completion_ids], device=device
        )
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(
            agg_is_truncated.float().mean().item()
        )
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if (
            len(term_completion_lengths) == 0
        ):  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_lengths.float().max().item()
        )

        return (
            prompt_ids,
            completion_ids,
            total_completion_tokens,
            logprobs,
            extra_fields,
        )

    def _generate_single_turn(
        self, prompts: list, inputs: list[dict[str, torch.Tensor | Any]]
    ):
        device = self.accelerator.device

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                # wake up colocated vLLM instances if needed
                torch.cuda.empty_cache()  # required to avoid OOM in some cases
                self.llm.wake_up(tags=["weights"])

            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if is_conversational({"prompt": prompts[0]}):
                prompts = [
                    prepare_multimodal_messages_vllm(prompt) for prompt in prompts
                ]

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts = gather_object(prompts)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts[:: self.num_generations]

                    sampling_params = {
                        "n": self.num_generations,
                        "repetition_penalty": self.repetition_penalty,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": -1 if self.top_k is None else self.top_k,
                        "min_p": 0.0 if self.min_p is None else self.min_p,
                        "max_tokens": self.max_completion_length,
                        "truncate_prompt_tokens": self.max_prompt_length,
                        "guided_decoding_regex": self.guided_decoding_regex,
                        "generation_kwargs": self.args.generation_kwargs,
                    }
                    with profiling_context(self, "vLLM.generate"):
                        if self.rollout_func is not None:
                            if is_conversational({"prompt": ordered_set_of_prompts[0]}):
                                ordered_set_of_prompts = [
                                    apply_chat_template(
                                        {"prompt": p},
                                        self.processing_class,
                                        **self.chat_template_kwargs,
                                    )["prompt"]
                                    for p in ordered_set_of_prompts
                                ]
                            output = self.rollout_func(
                                ordered_set_of_prompts,
                                self.args,
                                self.processing_class,
                            )
                        else:
                            if is_conversational({"prompt": ordered_set_of_prompts[0]}):
                                output = self.vllm_client.chat(
                                    messages=ordered_set_of_prompts,
                                    **sampling_params,
                                    chat_template_kwargs=self.chat_template_kwargs,
                                )
                            else:
                                output = self.vllm_client.generate(
                                    prompts=ordered_set_of_prompts, **sampling_params
                                )
                        # Extract required fields and collect any extra fields for reward functions
                        required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                        extra_fields = {
                            k: v for k, v in output.items() if k not in required_keys
                        }
                        payload = (
                            output["prompt_ids"],
                            output["completion_ids"],
                            output["logprobs"],
                            extra_fields,
                        )
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, all_logprobs, all_extra_fields = (
                    obj_list[0]
                )

                # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
                all_prompt_ids = [
                    ids for ids in all_prompt_ids for _ in range(self.num_generations)
                ]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]
                logprobs = all_logprobs[process_slice]

                # Slice extra fields dict-of-lists per process (extra fields are per-completion, like completion_ids)
                extra_fields = {}
                for key, values in all_extra_fields.items():
                    if isinstance(values, list):
                        extra_fields[key] = values[process_slice]
                    else:
                        extra_fields[key] = values

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(
                        regex=self.guided_decoding_regex
                    )
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "truncate_prompt_tokens": self.max_prompt_length,
                    "guided_decoding": guided_decoding,
                    "logprobs": 0,  # enable returning log probabilities; 0 means for the sampled tokens only
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts)
                    gathered_prompts = [
                        None for _ in range(self.vllm_tensor_parallel_size)
                    ]
                    torch.distributed.all_gather_object(
                        gathered_prompts, prompts, group=self.tp_group
                    )
                    all_prompts = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts = prompts

                if self.args.vllm_enable_sleep_mode:
                    self.llm.wake_up(tags=["kv_cache"])

                with profiling_context(self, "vLLM.generate"):
                    if is_conversational({"prompt": prompts[0]}):
                        all_outputs = self.llm.chat(
                            all_prompts, sampling_params=sampling_params, use_tqdm=False
                        )
                    else:
                        all_outputs = self.llm.generate(
                            all_prompts, sampling_params=sampling_params, use_tqdm=False
                        )

                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [
                    output.token_ids
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(
                        group=self.tp_group
                    )
                    tp_slice = slice(
                        local_rank_in_group * orig_size,
                        (local_rank_in_group + 1) * orig_size,
                    )
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                    logprobs = all_logprobs[tp_slice]
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids
                    logprobs = all_logprobs

                extra_fields = {}  # No extra fields for colocate mode

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=2)

        elif self.use_transformers_paged:
            processor_kwargs = {
                "max_length": self.max_prompt_length,
                "truncation": True,
                "add_special_tokens": False,
            }
            if is_conversational({"prompt": prompts[0]}):
                processor_outputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    **processor_kwargs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                processor_outputs = self.processing_class(
                    text=prompts, **processor_kwargs
                )

            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
                (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                if self.args.cast_lm_head_to_fp32:
                    unwrapped_model.lm_head.to(torch.float32)
                with torch.inference_mode():
                    # Continuous batching API expects 'inputs' arg only
                    all_outputs = unwrapped_model.generate_batch(
                        processor_outputs["input_ids"],
                        generation_config=self.generation_config,
                        progress_bar=False,
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [
                output.generated_tokens for output in all_outputs.values()
            ]
            prompt_ids = processor_outputs["input_ids"]
            logprobs = None  # not used in this case
            extra_fields = {}  # No extra fields for paged mode

        else:
            # Regular generation path
            processor_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "padding_side": "left",
                "max_length": self.max_prompt_length,
                "truncation": True,
                "add_special_tokens": False,
            }
            if is_conversational({"prompt": prompts[0]}):
                generate_inputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    **processor_kwargs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                generate_inputs = self.processing_class(
                    text=prompts, **processor_kwargs
                )
            generate_inputs = super(GRPOTrainer, self)._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
                (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ),
            ):

                # *** EDITED: CRITICAL -- pass `inputs` to generate for GOAL CONDITIONING!
                # Previously: would have called unwrapped_model.generate(...) directly here.
                # NOW: Use our custom generation entrypoint for goal-aware completions.
                logits_processor_list = None
                if self.logits_processor_fns is not None:
                    logits_processor_list = LogitsProcessorList(
                        [
                            self.logits_processor_fns[i](generate_inputs, inputs)
                            for i in range(len(self.logits_processor_fns))
                        ]
                    )
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs,
                    generation_config=self.generation_config,
                    disable_compile=True,
                    logits_processor=logits_processor_list,
                )
            # Compute prompt length and extract completion ids
            prompt_ids, prompt_mask = (
                generate_inputs["input_ids"],
                generate_inputs["attention_mask"],
            )
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full(
                (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
            )
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
                is_eos.size(0), -1
            )
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [
                p[m].tolist()
                for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)
            ]
            completion_ids = [
                c[m].tolist()
                for c, m in zip(completion_ids, completion_mask.bool(), strict=True)
            ]
            logprobs = None  # not used in this case
            extra_fields = {}  # No extra fields for non-rollout_func paths

        return prompt_ids, completion_ids, logprobs, extra_fields


##############################################################################
#                                                                            #
#                                Logits Processor Functions                   #
#                                                                            #
##############################################################################


def get_dfa_model(
    hmm_model: torch.nn.Module,
    prompt_ids: List[int],  # Shape: (B, T)
    tokenizer: AutoTokenizer,
    keyphrases: List[List[str]] = [[" "]],
    suffix_ids: Optional[List[int]] = None,
    min_new_tokens: int = 5,
    max_new_tokens: int = 32,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Constructs a DFA model for the given prompt and keyphrases.

    Args:
        prompt_ids (List[int]): Prompt integer list
        keyphrases (List[List[str]], optional): List of keyphrases to be constrained. Defaults to [[' ']].
        suffix_ids (Optional[List[int]], optional): Suffix integer list. Defaults to None.
        min_new_tokens (int, optional): Minimum number of tokens to generate. Defaults to 5.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 32.
        device (torch.device, optional): Device to run the model on. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').

    Returns:
        constraint_logits_processor: Logits processor for the DFA model.
    """

    vocab_size = len(tokenizer)

    ##################################### prefix, suffix, prompt #####################################
    prefix = ""  # generate text starting with nothing
    suffix = ".<|endoftext|>"  # generate text ending with '<|endoftext|>'; a suffix must end with the eos token

    prefix_ids = tokenizer.encode(prefix)
    if suffix_ids is None:
        suffix_ids = tokenizer.encode(suffix)

    ##################################### DFA Construction #####################################
    # ac_builder constructs a DFA representing the constraint that (at least)
    # one the patterns must appear; a pattern is a sequence of token ids
    ac_builder = ctrlg.AhoCorasickBuilder(vocab_size)

    dfa_graphs = []

    # constraint 1:
    for keyphrase in keyphrases:
        patterns = [tokenizer.encode(x) for x in keyphrase]
        dfa_graphs.append(ac_builder.build(patterns))

    # taking the intersection of the DFAs, i.e., "logical and" of the constraints.
    # This function also minimizes the constructed DFA, which is mainly CPU-based operations;
    # Due to its pure python implemenation, DFA minimization can be slow for complex constraints
    dfa_graph = ctrlg.DFA_prod(dfa_graphs, mode="intersection")

    # compile the dfa_graph for efficient GPU execution
    dfa_model = ctrlg.DFAModel(dfa_graph, vocab_size).to(device)

    ##################################### token length #####################################

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


class DummyLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        generate_inputs: dict[str, torch.Tensor | Any],
        prompts: List[dict[str, Any]],
        tokenizer: AutoTokenizer,
        hmm_model: Optional[torch.nn.Module] = None,
        min_new_tokens: int = 5,
        max_new_tokens: int = 32,
        constraint_mode: Literal["suffix", "keyphrase"] = "suffix",
    ):
        self.generate_inputs = generate_inputs
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.hmm_model = hmm_model
        self.min_new_tokens = min_new_tokens
        self.max_new_tokens = max_new_tokens

        # check if all solutions are the same:
        self.dfa_logits_processor = None
        if (
            all(p["question"] == prompts[0]["question"] for p in prompts)
            and hmm_model is not None
        ):
            device = self.generate_inputs["input_ids"][0].device
            hmm_model = hmm_model.to(device)

            keyphrases = (
                [[prompts[0]["answer"]]]
                if constraint_mode == "keyphrase" or constraint_mode == "both"
                else [[" "]]
            )
            suffix_ids = (
                tokenizer.encode(" ")
                + tokenizer.encode(prompts[0]["answer"])
                + [tokenizer.eos_token_id]
                if constraint_mode == "suffix" or constraint_mode == "both"
                else None
            )
            prompt_ids = generate_inputs["input_ids"][0].tolist()

            self.dfa_logits_processor = get_dfa_model(
                hmm_model=hmm_model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                keyphrases=keyphrases,
                suffix_ids=suffix_ids,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                device=generate_inputs["input_ids"][0].device,
            )
        elif hmm_model is not None:
            # give warning that no dfa logits processor will be used
            print("Warning: no DFA logits processor will be used")

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.dfa_logits_processor is not None:
            return self.dfa_logits_processor(input_ids, scores)
        return scores


##############################################################################
#                                                                            #
#                                Utility Functions                           #
#                                                                            #
##############################################################################


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
    # logger.info(
    #     f"Question:\n{q}\nAnswer:\n{answer[0]}\nResponse:\n{responses[0]}\nExtracted:\n{extracted_responses[0]}"
    # )
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
    "ctrlg/tulu2-7b_writing-prompts": "ctrlg_t27b",
    "ctrlg/hmm_tulu2-7b_writing-prompts_32768": "ctrlg_t27bh32768",
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
        # default="Qwen/Qwen2-0.5B-Instruct", ctrlg/tulu2-7b_writing-prompts
        default="ctrlg/gpt2-large_common-gen",  # use even smaller model: "erwanf/gpt2-mini" or "gpt2-small"
        help="Name of the model to train (e.g., 'tulu', 'google/gemma-2b', etc.)",
    )
    parser.add_argument(
        "--hmm",
        type=str,
        default=None,
        # ctrlg/hmm_tulu2-7b_writing-prompts_32768
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
    hmm_model = ctrlg.HMM.from_pretrained(args.hmm) if args.hmm is not None else None

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
    OUTPUT_DIR = f"./experiments/{run_name}"
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=run_name,
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
        save_total_limit=2,
        push_to_hub=True,
        # Logging
        logging_steps=10,
        log_completions=True,
        num_completions_to_print=4,
        report_to="wandb",
        log_on_each_node=False,
        # Eval
        eval_steps=1000,
        eval_strategy="steps",
    )

    # Trainer setup
    trainer = GRPOTrainerCustom(
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
        logits_processor_fns=[
            lambda *a, **kw: DummyLogitsProcessor(
                *a,
                **kw,
                tokenizer=tokenizer,
                hmm_model=hmm_model,
                min_new_tokens=args.min_new_tokens,
                max_new_tokens=args.max_new_tokens,
                constraint_mode=args.constraint_mode,
            )
        ],
    )

    resume_from_checkpoint = True if len(os.listdir(OUTPUT_DIR)) > 0 else False
    print(f">>> Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.push_to_hub()
