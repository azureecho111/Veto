import argparse
import json
import os
import random
import time
import threading
import concurrent.futures
import re
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

from base import EchoConfig, EchoForQwen

def extract_choice(text: str) -> str:
    """Extract choice letter A-E from model output."""
    patterns = [
        r"\(([A-E])\)",  # (A)
        r"\[([A-E])\]",  # [A]
        r"\b([A-E])\b",  # A as a whole word
    ]
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return ""

class APIJudge:
    def __init__(self, api_url, api_key, model_name):
        self.client = OpenAI(base_url=api_url, api_key=api_key)
        self.model_name = model_name

    def judge(self, question: str, ground_truth: str, prediction: str) -> bool:
        prompt = f"""You are an objective evaluator.
Analyze the following multiple-choice question, the correct answer (ground truth), and a model's prediction (which might include reasoning).
Decide if the model's final conclusion matches the ground truth.

Question: {question}
Ground Truth Choice: {ground_truth}
Model Prediction: {prediction}

Based ONLY on the final choice made by the model, is it correct? 
Respond with exactly one word: 'YES' if correct, 'NO' if incorrect.
Do not provide any explanation."""
        
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.0
            )
            ans = res.choices[0].message.content.strip().upper()
            return "YES" in ans
        except Exception as e:
            print(f"Judge API Error: {e}")
            # Fallback to simple extraction if API fails
            pred_choice = extract_choice(prediction)
            return pred_choice == ground_truth

def main():
    parser = argparse.ArgumentParser(description="Echo Framework Evaluation with Thinking and API Judge")

    # Dataset Params
    parser.add_argument("--data-path", type=str, required=True, help="Path to V* jsonl file")
    parser.add_argument("--image-root", type=str, required=True, help="Root folder for original images")
    parser.add_argument("--output-path", type=str, default="echo_think_results.jsonl")
    parser.add_argument("--samples", type=int, default=None, help="Number of random samples to eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # API Params (Reasoning Model)
    parser.add_argument("--api-url", type=str, default="http://localhost:18903/v1", help="Reasoning API base URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="Reasoning API Key")
    parser.add_argument("--model-name", type=str, default="qwen2.5-vl-7b", help="Model name for reasonin    g")

    # Judge parameters are read from the environment by default.
    # BASE_URL = "https://yunwu.ai/v1"
    parser.add_argument("--judge-api-url", type=str, default="https://yunwu.ai/v1", help="Judge API base URL")
    parser.add_argument("--judge-api-key", type=str, default=os.getenv("JUDGE_API_KEY", "EMPTY"), help="Judge API Key")
    parser.add_argument("--judge-model-name", type=str, default="gpt-4o", help="Model name for judging")

    # Echo Configuration
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--no-visual-prompt", action="store_false", dest="visual_prompt", help="Disable visual prompt in final crop")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--debug-dir", type=str, default="debug_echo_think")
    parser.add_argument("--sample-id", type=str, default=None, help="Specific ID")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--eval-failed", action="store_true", help="Only evaluate cases where the model failed")
    parser.add_argument("--failed-log-path", type=str, default="../vstar_vllm_results.jsonl", help="Path to the previous results file")
    parser.add_argument("--sample-list", type=str, default=None, help="Path to a txt file containing question_ids")

    args = parser.parse_args()

    # Create run-specific debug dir
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

    model = EchoForQwen(config)
    judge = APIJudge(args.judge_api_url, args.judge_api_key, args.judge_model_name)

    # Load data
    dataset = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))

    # Filtering
    if args.sample_id:
        print(f"Filtering for sample_id: {args.sample_id}")
        dataset = [item for item in dataset if str(item.get('question_id')) == str(args.sample_id)]
        if not dataset:
            print(f"Error: Question ID {args.sample_id} not found.")
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
        print(f"Loaded {len(failed_ids)} failed IDs.")
        dataset = [item for item in dataset if str(item.get('question_id')) in failed_ids]
        if not dataset:
            print("No matching failed samples found.")
            return
    elif args.sample_list:
        if not os.path.exists(args.sample_list):
            print(f"Error: Sample list file {args.sample_list} not found.")
            return
        with open(args.sample_list, 'r', encoding='utf-8') as f:
            target_ids = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(target_ids)} IDs from {args.sample_list}")
        dataset = [item for item in dataset if str(item.get('question_id')) in target_ids]
        if not dataset:
            print("No matching samples found.")
            return
    elif args.samples and args.samples < len(dataset):
        random.seed(args.seed)
        dataset = random.sample(dataset, args.samples)

    results = []
    category_metrics = {}
    metrics_lock = threading.Lock()

    print(f"Starting Think-Mode Echo evaluation with {args.num_workers} workers...")

    def process_item(item):
        image_path = os.path.join(args.image_root, item['image'])
        if not os.path.exists(image_path): return None

        try:
            img = Image.open(image_path).convert("RGB")
            sample_id = item.get('question_id', str(random.randint(0, 1000)))
            
            # 1. 修改 Question：去掉强制直接输出答案的要求，使其进入 CoT/Thinking
            raw_q = item['text']
            instruction_to_remove = "Answer with the option's letter from the given choices directly."
            clean_q = raw_q.replace(instruction_to_remove, "").strip()

            local_debug_dir = os.path.join(current_debug_root, sample_id)
            if not os.path.exists(local_debug_dir):
                os.makedirs(local_debug_dir, exist_ok=True)

            # Inference
            response_text, metadata = model.generate(img, clean_q, custom_debug_dir=local_debug_dir)

            # 2. API Judge：代替正则提取
            is_correct = judge.judge(clean_q, item['label'], response_text)
            
            print(f"### ID {sample_id} | Correct: {is_correct} | GT: {item['label']}")

            res_item = {
                **item,
                "clean_question": clean_q,
                "echo_response": response_text,
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
                with open(os.path.join(current_debug_root, args.output_path), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(res_item, ensure_ascii=False) + "\n")
            return res_item
        except Exception as e:
            print(f"Error {sample_id}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_item, item) for item in dataset]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    # Summary
    print("\n" + "=" * 50)
    print("THINK-MODE EVALUATION REPORT")
    print("=" * 50)
    total_correct = 0
    total_samples = 0
    for cat, stats in category_metrics.items():
        acc = stats["correct"] / stats["total"] * 100
        print(f"{cat:<25}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")
        total_correct += stats["correct"]
        total_samples += stats["total"]
    if total_samples > 0:
        print("-" * 50)
        print(f"{'OVERALL':<25}: {total_correct/total_samples*100:6.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()
