# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

###############################################################################
# Copyright (C) 2020-2025 Habana Labs, Ltd. an Intel Company
###############################################################################
# lm_eval --model hf-multimodal --model_args pretrained=meta-llama/Llama-3.2-11B-Vision-Instruct,max_images=1,interleave=True,device=hpu,image_string=\<image\> --tasks mmmu_val --apply_chat_template --batch_size=16 2>&1 | tee /software/users/akarnieli/llama-vision-11b-mmmu.txt
# lm_eval --model hf-multimodal --model_args pretrained=meta-llama/Llama-3.2-11B-Vision-Instruct,max_images=1,interleave=True,device=hpu,image_string=\<image\> --tasks mmmu_val_health_and_medicine --apply_chat_template --batch_size=1 --gen_kwargs max_new_tokens=128
# python run_image_lm_eval.py --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct -o bla.txt --tasks mmmu_val_humanities_and_social_science

"""
Qwen: 

python run_image_lm_eval.py --model_name_or_path Qwen/Qwen2-VL-2B-Instruc -o bla.txt --tasks mmmu_val_humanities_and_social_science
PYTHONPATH=/home/akarnieli/qnpu/pt/src/optimum-habana:/home/akarnieli/qnpu/pt/src/neural-compressor-fork python run_image_lm_eval.py --model_name_or_path Qwen/Qwen2-VL-2B-Instruct -o /home/akarnieli/models/qwen/qwen2_vl_2b_instruct_w4a16.txt --tasks mmmu_val --local_quantized_inc_model_path /workdisk/tgafni/qwen2_vl_2B_instruct/w4a16/qwen2_vl_2b_instruct_4bits


fp8:
QUANT_CONFIG=/home/akarnieli/models/qwen/maxabs_measure.json PYTHONPATH=/home/akarnieli/qnpu/pt/src/optimum-habana:/home/akarnieli/qnpu/pt/src/neural-compressor-fork python run_image_lm_eval.py --model_name_or_path Qwen/Qwen2-VL-2B-Instruct -o /tmp/bla.txt --tasks mmmu_val --local_quantized_inc_model_path /workdisk/tgafni/qwen2_vl_2B_instruct/w4a8/qwen2_vl_2b_instruct_4bits --limit 2
QUANT_CONFIG=/home/akarnieli/models/qwen/maxabs_quant.json PYTHONPATH=/home/akarnieli/qnpu/pt/src/optimum-habana:/home/akarnieli/qnpu/pt/src/neural-compressor-fork python run_image_lm_eval.py --model_name_or_path Qwen/Qwen2-VL-2B-Instruct -o /home/akarnieli/models/qwen/qwen2_vl_2b_instruct_w4a8.txt --tasks mmmu_val --local_quantized_inc_model_path /workdisk/tgafni/qwen2_vl_2B_instruct/w4a8/qwen2_vl_2b_instruct_4bits

"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Literal, Optional

import psutil
import torch
import torch.nn.functional as F
from lm_eval import evaluator, tasks, utils
from lm_eval.models.huggingface import HFLM, TemplateLM
from lm_eval.models.hf_vlms import HFMultimodalLM
from lm_eval.models.utils import stop_sequences_criteria
from lm_eval.loggers import EvaluationTracker

# Local imports
from run_generation import setup_parser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import MllamaForConditionalGeneration, AutoProcessor

from transformers.generation import GenerationConfig
from utils import finalize_quantization, initialize_model, save_model

from optimum.habana.utils import get_hpu_memory_stats

 
from transformers import AutoModelForCausalLM, Qwen2VLConfig, Qwen2VLForConditionalGeneration
AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForConditionalGeneration)

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = utils.eval_logger

# This hack is a workaround to limitations of lm_eval which always allocates
# mp.Pool with max cpu count which explodes on multinode scenarios and for hpu
# create multiprocess with spawn context
OrigPool = mp.Pool


def LimitedSpawnPool(_):
    spawn_context = mp.get_context("spawn")
    physical_cpu_count = psutil.cpu_count(logical=False)
    pool_size = physical_cpu_count
    world_size = int(os.getenv("WORLD_SIZE", 1))
    pool_size //= max(world_size, 1)
    if (pool_size * world_size) != physical_cpu_count:
        pool_size -= 1
    return spawn_context.Pool(pool_size)


mp.Pool = LimitedSpawnPool


def setup_lm_eval_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Evaluation script for HPU"
    )
    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        help="Input length buckets to use with static_shapes",
        default=[16, 32, 64, 128, 189, 284, 384],
    )
    parser.add_argument(
        "--output_path", "-o", type=str, help="Output file with end results and runtime parameters", required=True
    )
    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        help="Comma-separated list of task names or task groupings to evaluate on",
        default="hellaswag,lambada_openai,piqa,winogrande",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        help="Limit the number of examples per task. If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument("--max_graphs", type=int, help="Maximum number of HPU graphs", default=None)
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )

    args = setup_parser(parser)
    return args

class HabanaHFMultimodalLM(HFMultimodalLM):
    def _model_multimodal_generate(self, inputs, max_length, stop, **generation_kwargs):
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer,
            stop,
            inputs["input_ids"].shape[1],
            inputs["input_ids"].shape[0],
        )
        # import ptvsd
        # ptvsd.enable_attach(address=('127.0.0.1', 5678))
        # ptvsd.wait_for_attach()
        # import debugpy
        # debugpy.listen(("localhost", 5678))
        # print("WAIT FOR DEBUGPY")
        # debugpy.wait_for_client()
        # debugpy.breakpoint()
        
        return self.model.generate(
            **inputs,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            # use_cache=True, #FIXME ?
            **generation_kwargs,
        )


def main() -> None:
    # Modified based on cli_evaluate function in https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/__main__.py/#L268
    args = setup_lm_eval_parser()

    if args.system_instruction is None and args.prompt is not None:
        logger.warning(" --system_instruction will be assigned --prompt value")
        args.system_instruction = args.prompt
    elif args.system_instruction is not None and args.prompt is None:
        logger.warning(" --prompt will be assigned --system_instruction value")
        args.prompt = args.system_instruction
    elif args.system_instruction is not None and args.prompt is not None:
        logger.warning(" --prompt overwritten by --system_instruction")
        args.prompt = args.system_instruction

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")
    if args.limit:
        logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING.REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")
    if args.fewshot_as_multiturn and args.use_chat_template is False:
        raise ValueError("When `fewshot_as_multiturn` is selected, `use_chat_template` must be set.")

    task_manager = tasks.TaskManager(args.verbosity)

    if args.tasks is None:
        logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        print(task_manager.list_all_tasks())
        sys.exit()
    elif args.tasks == "list_groups":
        print(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        print(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        print(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )

        # model, _, tokenizer, generation_config = initialize_model(args, logger)
        model, _, _, generation_config = initialize_model(args, logger)
        # model.config.use_cache = True
        # processor = AutoProcessor.from_pretrained(model_id)
        lm = HabanaHFMultimodalLM(pretrained=model)

    if args.trust_remote_code:
        # trust_remote_code fix was introduced in lm_eval 0.4.3
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    evaluation_tracker_args = {}
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    # # Regroup part of generation_config as gen_kwargs, defined as "String arguments for model generation on greedy_until tasks, e.g. `temperature=0,top_k=0,top_p=0`."
    # if generation_config.do_sample is True:
    #     gen_kwargs = f"temperature={generation_config.temperature}"
    #     if generation_config.top_k is not None:
    #         gen_kwargs += f",top_k={generation_config.top_k}"
    #     gen_kwargs += f",do_sample={generation_config.do_sample}"
    #     gen_kwargs += f",top_p={generation_config.top_p}"
    # else:
    #     gen_kwargs = None

     # needed for VL with HPU
    gen_kwargs="static_shapes=True,max_gen_toks=128,lazy_mode=True,use_cache=True,cache_implementation=static,use_flash_attention=True"


    eval_start = time.perf_counter()

    with torch.no_grad():
        results = evaluator.simple_evaluate(
            lm,
            model_args="max_images=1,interleave=True,image_string=<|image|>",
            gen_kwargs=gen_kwargs,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
            write_out=args.write_out,
            log_samples=args.log_samples,
            system_instruction=args.system_instruction,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            task_manager=task_manager,
            verbosity=args.verbosity,
            apply_chat_template=True,
            evaluation_tracker=evaluation_tracker,
            predict_only=args.predict_only,
        )

    if args.device == "hpu":
        import habana_frameworks.torch.hpu as torch_hpu

        torch_hpu.synchronize()
    eval_end = time.perf_counter()

    results["args"] = vars(args)
    results["duration"] = eval_end - eval_start

    from lm_eval.utils import make_table

    if args.local_rank == 0:
        if args.device == "hpu":
            mem = get_hpu_memory_stats()
            for k, v in mem.items():
                print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))

        json_str = json.dumps(results, indent=2, default=utils.handle_non_serializable, ensure_ascii=False)
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        if args.show_config:
            print(json_str)

    print(make_table(results))

    if args.quant_config:
        finalize_quantization(model)
    if args.save_quantized_model_with_inc:
        save_model(model, tokenizer, args.saved_model_path)

    if args.const_serialization_path and os.path.isdir(args.const_serialization_path):
        import shutil

        shutil.rmtree(args.const_serialization_path)


if __name__ == "__main__":
    main()
