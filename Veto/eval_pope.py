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
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset

from base import EchoConfig
from enhanced_base import EnhancedEchoForQwen

# ── 判题用外部 API 客户端（仅 --think 模式需要）────────────────────────────────
judge_client = OpenAI(
    base_url="https://yunwu.ai/v1",
    api_key=os.getenv("JUDGE_API_KEY", "EMPTY")
)

def extract_yes_no(text: str) -> str:
    """Extract 'yes' or 'no' from model output."""
    clean_text = text.lower().strip()
    if clean_text.startswith("yes"): return "yes"
    if clean_text.startswith("no"): return "no"
    
    matches = re.findall(r'\b(yes|no)\b', clean_text)
    if matches:
        return matches[0]
    return "unknown"

def main():
    parser = argparse.ArgumentParser(description="Enhanced Echo Framework Evaluation on POPE Benchmark")
    
    # API Params
    parser.add_argument("--api-url", type=str, default="http://localhost:18903/v1", help="API base URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model-name", type=str, default="qwen2.5-vl-7b", help="Model name for reasoning")

    # POPE dataset config
    parser.add_argument("--hf-repo", type=str, default="lmms-lab/POPE")
    parser.add_argument("--hf-split", type=str, default="test")

    # Output config
    parser.add_argument("--output-path", type=str, default="enhanced_echo_pope_eval_results.jsonl")
    parser.add_argument("--samples", type=int, default=None, help="Number of random samples to eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers for evaluation")

    # Sampling Params
    parser.add_argument("--temperature", type=float, default=0.0)

    # Echo Configuration
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--no-visual-prompt", action="store_false", dest="visual_prompt")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=str, default="debug_enhanced_echo_pope")

    # Eval Mode
    parser.add_argument("--think", action="store_true", help="Use GPT-4o judging for chain-of-thought outputs")

    args = parser.parse_args()

    # Create run-specific debug dir
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_debug_root = os.path.join(args.debug_dir, f"run_{run_timestamp}")
    os.makedirs(current_debug_root, exist_ok=True)
    os.makedirs(os.path.join(current_debug_root, "trajectories"), exist_ok=True)

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

    # 保存 Metadata
    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "enhanced_echo_pope",
        "api_url": args.api_url,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "samples": args.samples,
        "max_steps": args.max_steps,
        "think": args.think,
        "num_workers": args.num_workers,
        "command": "python " + " ".join(sys.argv),
        "args": vars(args)
    }
    with open(os.path.join(current_debug_root, "metadata.json"), "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=4, ensure_ascii=False)

    print(f"Loading POPE dataset from HuggingFace (`{args.hf_repo}` '{args.hf_split}' split) ...")
    hf_dataset = load_dataset(args.hf_repo, "default", split=args.hf_split)
    dataset = list(hf_dataset)

    if args.samples and args.samples < len(dataset):
        random.seed(args.seed)
        dataset = random.sample(dataset, args.samples)

    if not dataset:
        print("No samples to evaluate.")
        return

    print(f"Total samples to evaluate: {len(dataset)}")

    model = EnhancedEchoForQwen(config)

    results = []
    category_metrics = {}
    metrics_lock = threading.Lock()
    output_jsonl_path = os.path.join(current_debug_root, args.output_path)

    def process_item(item):
        sample_id = str(item.get('question_id', item.get('id', random.randint(0, 1000000))))
        question = item['question']
        gt_answer = str(item['answer']).strip().lower()
        img_pil = item['image'].convert("RGB")

        trajectories_dir = os.path.join(current_debug_root, "trajectories")
        local_debug_dir = os.path.join(trajectories_dir, sample_id)
        if not os.path.exists(local_debug_dir):
            os.makedirs(local_debug_dir, exist_ok=True)

        if args.think:
            instruction_to_remove = "Please answer yes or no directly."
            question = question.replace(instruction_to_remove, "").strip()
        else:
            if "yes or no" not in question.lower():
                question += "\n Please answer yes or no directly."

        try:
            response_text, tracker_data = model.generate_with_tracking(img_pil, question, custom_debug_dir=local_debug_dir)
            
            if not args.think:
                choice = extract_yes_no(response_text)
                is_correct = (choice == gt_answer)
            else:
                judge_prompt = f"""You are an objective evaluator.
Analyze the following yes/no question, the correct answer (ground truth), and a model's prediction.
Decide if the model's final conclusion matches the ground truth.

Question: {question}
Ground Truth Answer: {gt_answer}
Model Prediction: {response_text}

Based ONLY on the final conclusion made by the model, is it correct?
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
                    is_correct = "YES" in ans
                except Exception as e:
                    print(f"Error calling judge API for {sample_id}: {e}")
                    is_correct = False
                choice = "###"

            print(f"########## Sample {sample_id} | Choice: {choice} | GT: {gt_answer} | Correct: {is_correct} | Tokens: {tracker_data['total_tokens']} | Time: {tracker_data['total_wall_time']:.2f}s")

            res_item = {
                "id": item.get("id"),
                "question_id": item.get("question_id"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "category": item.get("category", "unknown"),
                "echo_prediction": response_text,
                "processed_choice": choice,
                "is_correct": is_correct,
                "debug_path": os.path.abspath(local_debug_dir),
                "metrics": {
                    "total_tokens": tracker_data["total_tokens"],
                    "total_api_time": tracker_data["total_api_time"],
                    "total_wall_time": tracker_data["total_wall_time"]
                }
            }

            with metrics_lock:
                cat = item.get('category', 'unknown')
                if cat not in category_metrics:
                    category_metrics[cat] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "total": 0, "correct": 0}
                
                category_metrics[cat]["total"] += 1
                if is_correct:
                    category_metrics[cat]["correct"] += 1
                    
                if choice == "yes" and gt_answer == "yes":
                    category_metrics[cat]["TP"] += 1
                elif choice == "no" and gt_answer == "no":
                    category_metrics[cat]["TN"] += 1
                elif choice == "yes" and gt_answer == "no":
                    category_metrics[cat]["FP"] += 1
                elif choice == "no" and gt_answer == "yes":
                    category_metrics[cat]["FN"] += 1

                results.append(res_item)

                with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(res_item, ensure_ascii=False) + "\n")
            
            return res_item
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    print(f"Starting Parallel Enhanced Echo evaluation on POPE with {args.num_workers} workers...")
    script_start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_item, item) for item in dataset]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time

    # Summary Report
    print("\n" + "=" * 80)
    print("ENHANCED ECHO FRAMEWORK (POPE) EVALUATION REPORT")
    print("=" * 80)
    
    overall_stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "total": 0, "correct": 0}
    total_tokens = 0
    total_time = 0.0

    print(f"{'Category':<15} | {'Acc(%)':<8} | {'Precision(%)':<12} | {'Recall(%)':<10} | {'F1(%)':<8} | {'Yes(%)':<8}")
    print("-" * 80)
    
    for cat, stats in sorted(category_metrics.items()):
        total = stats["total"]
        if total == 0: continue
        
        acc = stats["correct"] / total * 100
        tp, fp, fn, tn = stats["TP"], stats["FP"], stats["FN"], stats["TN"]
        
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        yes_ratio = (tp + fp) / total * 100
        
        print(f"{cat:<15} | {acc:<8.2f} | {precision:<12.2f} | {recall:<10.2f} | {f1:<8.2f} | {yes_ratio:<8.2f}")
        
        for k in overall_stats.keys():
            overall_stats[k] += stats[k]

    for res in results:
        if res and "metrics" in res:
            total_tokens += res["metrics"]["total_tokens"]
            total_time += res["metrics"]["total_wall_time"]

    print("-" * 80)
    if overall_stats["total"] > 0:
        total = overall_stats["total"]
        overall_acc = overall_stats["correct"] / total * 100
        tp, fp, fn = overall_stats["TP"], overall_stats["FP"], overall_stats["FN"]
        
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        yes_ratio = (tp + fp) / total * 100
        
        avg_tokens = total_tokens / total
        avg_time = total_time / total
        amortized_time = total_script_time / total
        
        print(f"{'OVERALL':<15} | {overall_acc:<8.2f} | {precision:<12.2f} | {recall:<10.2f} | {f1:<8.2f} | {yes_ratio:<8.2f}")
        print("=" * 80)
        print(f"{'AVERAGE TOKENS/SAMPLE':<25}: {avg_tokens:.2f}")
        print(f"{'AVG LATENCY UNDER LOAD':<25}: {avg_time:.2f}s (Sum of indiv times / N)")
        print(f"{'AMORTIZED TIME/SAMPLE':<25}: {amortized_time:.2f}s (Total Script Time / N)")
    print("=" * 80)
    print(f"Results saved to: {output_jsonl_path}")

if __name__ == "__main__":
    main()
