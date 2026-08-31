import argparse
import json
import os
import sys
import random
import time
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm
import re
import concurrent.futures
import threading
from openai import OpenAI

from base import EchoConfig, EchoForQwen
from base_qwen3vl import EchoForQwen3

# ── 判题用外部 API 客户端（仅 --think 模式需要）────────────────────────────────
judge_client = OpenAI(
    base_url="https://yunwu.ai/v1",
    api_key=os.getenv("JUDGE_API_KEY", "EMPTY")
)


def extract_choice(text: str) -> str:
    """Extract choice letter A-E from model output."""
    patterns = [
        r"\(([A-E])\)",  # (A)
        r"\[([A-E])\]",  # [A]
        r"\b([A-E])\b",  # A as a whole word
    ]
    for p in patterns:
        matches = re.findall(p, text)
        if matches:
            return matches[-1].upper()
    return ""


def main():
    parser = argparse.ArgumentParser(description="Echo Framework Evaluation on V* Benchmark")

    # Dataset Params
    parser.add_argument("--data-path", type=str, required=True, help="Path to V* jsonl file")
    parser.add_argument("--image-root", type=str, required=True, help="Root folder for original images")
    parser.add_argument("--output-path", type=str, default="echo_eval_results.jsonl")
    parser.add_argument("--samples", type=int, default=None, help="Number of random samples to eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # API Params
    parser.add_argument("--api-url", type=str, default="http://localhost:18903/v1", help="API base URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model-name", type=str, default="qwen2.5-vl-7b", help="Model name for reasoning")
    parser.add_argument("--temperature", type=float, default=0.4)
    # Eval Mode
    parser.add_argument("--think", action="store_true",
                        help="Use GPT-4o judging for chain-of-thought outputs")
    # parser.add_argument("--model", action="store_true",)

    # Echo Configuration
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--no-visual-prompt", action="store_false", dest="visual_prompt", help="Disable visual prompt in final crop")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (save intermediate images)")
    parser.add_argument("--debug-dir", type=str, default="debug_echo_eval")
    parser.add_argument("--sample-id", type=str, default=None, help="Evaluate a specific question_id")
    parser.add_argument("--eval-failed", action="store_true", help="Only evaluate cases where the model failed in vstar_vllm_results.jsonl")
    parser.add_argument("--failed-log-path", type=str, default="../vstar_vllm_results.jsonl", help="Path to the previous results file to find failed cases")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers for evaluation")
    parser.add_argument("--sample-list", type=str, default=None, help="Path to a txt file containing question_ids to evaluate")

    args = parser.parse_args()

    # Create run-specific debug dir if debug is enabled
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_debug_root = os.path.join(args.debug_dir, f"run_{run_timestamp}")

    # Initialize Echo Framework
    config = EchoConfig()
    config.api_url = args.api_url
    config.api_key = args.api_key
    config.model_name = args.model_name
    config.max_steps = args.max_steps
    config.debug = args.debug
    config.visual_prompt = args.visual_prompt
    config.debug_dir = current_debug_root
    config.temperature = args.temperature
    config.seed = args.seed

    # ── 保存 Metadata ──────────────────────────────────────────────────────
    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "online",
        "data_path": args.data_path,
        "model_name": args.model_name,
        "api_url": args.api_url,
        "temperature": args.temperature,
        "samples": args.samples,
        "max_steps": args.max_steps,
        "think": args.think,
        "command": "python " + " ".join(sys.argv),
        "args": vars(args)
    }
    os.makedirs(current_debug_root, exist_ok=True)
    with open(os.path.join(current_debug_root, "metadata.json"), "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=4, ensure_ascii=False)

    # 提前创建 trajectories 目录
    os.makedirs(os.path.join(current_debug_root, "trajectories"), exist_ok=True)

    if "qwen3" in args.model_name.lower():
        model = EchoForQwen3(config)
    elif "internvl3" in args.model_name.lower():
        model = EchoForInternVL3(config)
    else:
        model = EchoForQwen(config)

    # Load data
    print(f"Loading data from {args.data_path}")
    dataset = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))

    # Filter or Sample
    if args.sample_id:
        print(f"Filtering for sample_id: {args.sample_id}")
        dataset = [item for item in dataset if str(item.get('question_id')) == str(args.sample_id)]
        if not dataset:
            print(f"Error: Question ID {args.sample_id} not found in {args.data_path}")
            return
    elif args.eval_failed:
        if not os.path.exists(args.failed_log_path):
            print(f"Error: Failed log file {args.failed_log_path} not found.")
            return
        failed_ids = set()
        with open(args.failed_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                res_item = json.loads(line)
                if not res_item.get('is_correct', True):
                    failed_ids.add(str(res_item.get('question_id')))
        
        print(f"Loaded {len(failed_ids)} failed IDs from {args.failed_log_path}")
        dataset = [item for item in dataset if str(item.get('question_id')) in failed_ids]
        print(f"Filtered dataset to {len(dataset)} failed samples.")
        
        if not dataset:
            print(f"No samples matching the failed IDs were found in the dataset.")
            return
            
    elif args.sample_list:
        if not os.path.exists(args.sample_list):
            print(f"Error: Sample list file {args.sample_list} not found.")
            return
        with open(args.sample_list, 'r', encoding='utf-8') as f:
            target_ids = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(target_ids)} IDs from {args.sample_list}")
        dataset = [item for item in dataset if str(item.get('question_id')) in target_ids]
        print(f"Filtered dataset to {len(dataset)} matching samples.")
        
        if not dataset:
            print("No matching samples found.")
            return

    elif args.samples and args.samples < len(dataset):
        if args.seed:
            random.seed(args.seed)
        print(f"Randomly sampling {args.samples} samples...")
        dataset = random.sample(dataset, args.samples)

    results = []
    category_metrics = {}
    metrics_lock = threading.Lock()

    def process_item(item):
        image_path = os.path.join(args.image_root, item['image'])
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found.")
            return None

        try:
            img = Image.open(image_path).convert("RGB")
            sample_id = item.get('question_id', str(random.randint(0, 1000000)))

            # 每个样本拥有独立的 debug 路径（在 run_xxx/ 目录下）
            trajectories_dir = os.path.join(current_debug_root, "trajectories")
            local_debug_dir = os.path.join(trajectories_dir, sample_id)
            if not os.path.exists(local_debug_dir):
                os.makedirs(local_debug_dir, exist_ok=True)
            
            # 1. 构造题目：如果启用 think 模式，去掉强制直接输出答案的要求
            question = item['text']
            if args.think:
                instruction_to_remove = "Answer with the option's letter from the given choices directly."
                question = question.replace(instruction_to_remove, "").strip()
            else:
                if "Answer with the option's letter" not in question:
                    question += "\n Please answer with the option's letter from the given choices directly."

            # 调用更新后的 thread-safe generate 方法
            response_text = model.generate(img, question, custom_debug_dir=local_debug_dir)
            metadata = {}
            
            if not args.think:
                choice = extract_choice(response_text)
                is_correct = (choice == item['label'])
            else:
                # GPT-4o 判题
                judge_prompt = f"""You are an objective evaluator.
Analyze the following multiple-choice question, the correct answer (ground truth), and a model's prediction (which might include reasoning).
Decide if the model's final conclusion matches the ground truth.

Question: {question}
Ground Truth Choice: {item['label']}
Model Prediction: {response_text}

Based ONLY on the final choice made by the model, is it correct?
Respond with exactly one word: 'YES' if correct, 'NO' if incorrect.
Do not provide any explanation."""

                judge_res = judge_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": judge_prompt},
                    ],
                    temperature=0.0,
                )
                ans = judge_res.choices[0].message.content.strip().upper()
                choice = "###"
                is_correct = "YES" in ans

            print(f"########## Sample {sample_id} | Choice: {choice} | GT: {item['label']} | Correct: {is_correct}")

            res_item = {
                **item,
                "echo_prediction": response_text,
                "processed_choice": choice,
                "is_correct": is_correct,
                "debug_path": os.path.abspath(local_debug_dir),
                "targets": metadata.get("targets", []),
                "found_targets": metadata.get("found_targets", []),
                "all_found": metadata.get("all_found", False)
            }

            with metrics_lock:
                cat = item.get('category', 'unknown')
                if cat not in category_metrics:
                    category_metrics[cat] = {"correct": 0, "total": 0}
                category_metrics[cat]["total"] += 1
                if is_correct:
                    category_metrics[cat]["correct"] += 1
                
                results.append(res_item)
                # 增量保存结果
                with open(os.path.join(current_debug_root,args.output_path), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(res_item, ensure_ascii=False) + "\n")
            
            return res_item
        except Exception as e:
            print(f"Error processing sample {item.get('question_id')}: {e}")
            import traceback
            traceback.print_exc()
            return None

    print(f"Starting Parallel Echo evaluation with {args.num_workers} workers...")
    # 用线程池并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_item, item) for item in dataset]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    # Summary Report
    print("\n" + "=" * 50)
    print("ECHO FRAMEWORK EVALUATION REPORT")
    print("=" * 50)
    total_correct = 0
    total_samples = 0

    for cat, stats in category_metrics.items():
        acc = stats["correct"] / stats["total"] * 100
        print(f"{cat:<25}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")
        total_correct += stats["correct"]
        total_samples += stats["total"]

    if total_samples > 0:
        overall_acc = total_correct / total_samples * 100
        print("-" * 50)
        print(f"{'OVERALL':<25}: {overall_acc:6.2f}% ({total_correct}/{total_samples})")
    print("=" * 50)


if __name__ == "__main__":
    main()
