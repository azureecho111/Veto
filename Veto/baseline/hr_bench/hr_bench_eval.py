import argparse
import json
import os
import sys
import re
import time
import random
import concurrent.futures
import threading
from typing import List, Dict, Any
import base64
from PIL import Image
from openai import OpenAI
from tqdm import tqdm
from io import BytesIO
from datasets import load_dataset

# ── 判题用外部 API 客户端（仅 --think 模式需要）────────────────────────────────
judge_client = OpenAI(
    base_url="https://yunwu.ai/v1",
    api_key=os.getenv("JUDGE_API_KEY", "EMPTY")
)

def extract_choice(text: str, think: bool) -> str:
    """Extract choice letter A-E from model output."""
    if not think:
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
    return ""

def main():
    parser = argparse.ArgumentParser(description="Baseline HR-Bench Evaluation ONLINE via API (Metadata & Think Mode Support)")
    
    # API Params
    parser.add_argument("--api-url", type=str, default="http://localhost:18903/v1", help="API base URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model-name", type=str, default="qwen2.5-vl-7b", help="Model name for reasoning")

    # Dataset Params
    parser.add_argument("--split", type=str, required=True, help="4K / 8K for HR-Bench")
    parser.add_argument("--output-path", type=str, default="baseline_hr_bench_online_results.jsonl")
    parser.add_argument("--samples", type=int, default=None, help="Number of random samples to eval")
    parser.add_argument("--sample-id", type=str, default=None, help="Evaluate a specific index")
    parser.add_argument("--sample-list", type=str, default=None, help="Path to a txt file containing index to evaluate")
    parser.add_argument("--eval-failed", action="store_true", help="Only evaluate cases where the model failed in previous log")
    parser.add_argument("--failed-log-path", type=str, default="", help="Path to the previous results file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers for evaluation")

    # Sampling Params
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=2048)

    # Eval Mode
    parser.add_argument("--think", action="store_true", help="Use GPT-4o judging for chain-of-thought outputs")

    # Debug / Output config
    parser.add_argument("--debug-dir", type=str, default="debug_baseline_hrbench_online")

    args = parser.parse_args()

    # Create run-specific debug dir
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_debug_root = os.path.join(args.debug_dir, f"run_{run_timestamp}")
    os.makedirs(current_debug_root, exist_ok=True)
    os.makedirs(os.path.join(current_debug_root, "trajectories"), exist_ok=True)

    # 保存 Metadata
    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "baseline_online",
        "api_url": args.api_url,
        "split": args.split,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "samples": args.samples,
        "think": args.think,
        "num_workers": args.num_workers,
        "command": "python " + " ".join(sys.argv),
        "args": vars(args)
    }
    with open(os.path.join(current_debug_root, "metadata.json"), "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=4, ensure_ascii=False)

    print(f"Loading data from HR-Bench (split:{args.split})")
    full_dataset = load_dataset("DreamMr/HR-Bench")[args.split]

    dataset = []
    # Filter or Sample
    if args.sample_id:
        dataset = [item for item in full_dataset if str(item.get('index')) == str(args.sample_id)]
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
        dataset = [item for item in full_dataset if str(item.get('index')) in failed_ids]
    elif args.sample_list:
        if not os.path.exists(args.sample_list):
            print(f"Error: Sample list {args.sample_list} not found.")
            return
        with open(args.sample_list, 'r', encoding='utf-8') as f:
            target_ids = {line.strip() for line in f if line.strip()}
        dataset = [item for item in full_dataset if str(item.get('index')) in target_ids]
    elif args.samples and args.samples < len(full_dataset):
        random.seed(args.seed)
        dataset = full_dataset.select(random.sample(range(len(full_dataset)), args.samples))
    else:
        dataset = full_dataset # To iterable/list form conceptually

    if len(dataset) == 0:
        print("No samples to evaluate.")
        return

    print(f"Total samples to evaluate: {len(dataset)}")

    online_client = OpenAI(
        base_url=args.api_url,
        api_key=args.api_key
    )

    results = []
    category_metrics = {}
    metrics_lock = threading.Lock()
    output_jsonl_path = os.path.join(current_debug_root, args.output_path)

    def process_item(item):
        sample_id = str(item.get('index', random.randint(0, 1000000)))
        image_base64 = item.get('image')
        if not image_base64:
            return None

        question = item['question'] + "\n" + f"A.{item['A']}\nB.{item['B']}\nC.{item['C']}\nD.{item['D']}"
        if not args.think:
            question += "\n Please answer with the option's letter from the given choices directly."

        try:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": question}
            ]
            res = online_client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed
            )
            response_text = res.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during API inference for {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            response_text = f"Error: {e}"

        # 判题
        if not args.think:
            choice = extract_choice(response_text, args.think)
            is_correct = (choice == item['answer'])
        else:
            judge_prompt = f"""You are an objective evaluator.
Analyze the following multiple-choice question, the correct answer (ground truth), and a model's prediction (which might include reasoning).
Decide if the model's final conclusion matches the ground truth.

Question: {question}
Ground Truth Choice: {item['answer']}
Model Prediction: {response_text}

Based ONLY on the final choice letter made by the model (ignore the difference in content of the choice, only judge whether the model choose the correct letter A/B/C/D), is it correct? 
Respond with exactly one word: 'YES' if correct, 'NO' if incorrect."""

            try:
                judge_res = judge_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": judge_prompt},
                    ],
                    temperature=0.0,
                )
                ans = judge_res.choices[0].message.content.strip().upper()
                is_correct = "yes" in ans.lower()
            except Exception as e:
                print(f"Error calling judge API for {sample_id}: {e}")
                is_correct = False
            choice = "###"

        print(f"########## Sample {sample_id} | Choice: {choice} | GT: {item['answer']} | Correct: {is_correct}")

        local_debug_dir = os.path.join(current_debug_root, "trajectories", sample_id)
        os.makedirs(local_debug_dir, exist_ok=True)
        
        # 存下结果 (HR-Bench格式不需要原路写入所有原始大字段，保留图像留空节省体积)
        res_item = {
            "image": "..",
            "text": question,
            "category": item.get('category', 'unknown'),
            "question_id": sample_id,
            "label": item['answer'],
            "prediction": response_text,
            "echo_prediction": response_text,
            "processed_choice": choice,
            "is_correct": is_correct,
            "debug_path": os.path.abspath(local_debug_dir)
        }

        with metrics_lock:
            cat = item.get('category', 'unknown')
            if cat not in category_metrics:
                category_metrics[cat] = {"correct": 0, "total": 0}
            category_metrics[cat]["total"] += 1
            if is_correct:
                category_metrics[cat]["correct"] += 1
            results.append(res_item)

            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(res_item, ensure_ascii=False) + "\n")

    print(f"Starting Parallel API evaluation with {args.num_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_item, item) for item in dataset]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    # Summary Report
    print("\n" + "="*50)
    print("BASELINE HR-BENCH ONLINE EVALUATION REPORT")
    print("="*50)
    total_correct = 0
    total_samples = 0
    
    for cat, stats in sorted(category_metrics.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{cat:<25}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")
        total_correct += stats["correct"]
        total_samples += stats["total"]

    if total_samples > 0:
        overall_acc = total_correct / total_samples * 100
        print("-" * 50)
        print(f"{'OVERALL':<25}: {overall_acc:6.2f}% ({total_correct}/{total_samples})")
    print("="*50)
    print(f"Results saved to: {output_jsonl_path}")

if __name__ == "__main__":
    main()
