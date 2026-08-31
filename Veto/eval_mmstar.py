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
from datasets import load_dataset

from base import EchoConfig
from enhanced_base import EnhancedEchoForQwen

judge_client = OpenAI(
    base_url="https://yunwu.ai/v1",
    api_key=os.getenv("JUDGE_API_KEY", "EMPTY")
)

def extract_choice(text: str) -> str:
    patterns = [
        r"\(([A-E])\)",
        r"\[([A-E])\]",
        r"\b([A-E])\b",
    ]
    for p in patterns:
        matches = re.findall(p, text)
        if matches:
            return matches[-1].upper()
    return ""

def main():
    parser = argparse.ArgumentParser(description="Enhanced Echo Framework Evaluation on MMStar Benchmark")

    parser.add_argument("--output-path", type=str, default="enhanced_echo_mmstar_eval_results.jsonl")
    parser.add_argument("--samples", type=int, default=None, help="Number of random samples to eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--api-url", type=str, default="http://localhost:18903/v1", help="API base URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model-name", type=str, default="qwen2.5-vl-7b", help="Model name for reasoning")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--think", action="store_true", help="Use GPT-4o judging for CoT outputs")

    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--no-visual-prompt", action="store_false", dest="visual_prompt")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=str, default="debug_enhanced_echo_mmstar")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers")

    args = parser.parse_args()

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_debug_root = os.path.join(args.debug_dir, f"run_{run_timestamp}")

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

    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "enhanced_echo_mmstar",
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

    os.makedirs(os.path.join(current_debug_root, "trajectories"), exist_ok=True)

    model = EnhancedEchoForQwen(config)

    print("Loading MMStar dataset from HuggingFace (`Lin-Chen/MMStar` 'val' split) ...")
    hf_dataset = load_dataset("Lin-Chen/MMStar", split="val")
    dataset = list(hf_dataset)

    if args.samples and args.samples < len(dataset):
        random.seed(args.seed)
        print(f"Randomly sampling {args.samples} samples...")
        dataset = random.sample(dataset, args.samples)

    if not dataset:
        print("No samples to evaluate.")
        return

    print(f"Total samples to evaluate: {len(dataset)}")

    results = []
    category_metrics = {}
    metrics_lock = threading.Lock()
    output_jsonl_path = os.path.join(current_debug_root, args.output_path)

    def process_item(item):
        sample_id = str(item.get('index', random.randint(0, 1000000)))
        question = item['question']
        gt_answer = str(item['answer']).strip()
        img_pil = item['image'].convert("RGB")

        trajectories_dir = os.path.join(current_debug_root, "trajectories")
        local_debug_dir = os.path.join(trajectories_dir, sample_id)
        if not os.path.exists(local_debug_dir):
            os.makedirs(local_debug_dir, exist_ok=True)

        if args.think:
            instruction_to_remove = "Answer with the option's letter from the given choices directly."
            question = question.replace(instruction_to_remove, "").strip()
        else:
            if "Answer with the option's letter" not in question:
                question += "\n Please answer with the option's letter from the given choices directly."

        try:
            response_text, tracker_data = model.generate_with_tracking(img_pil, question, custom_debug_dir=local_debug_dir)
            metadata = {}
            
            if not args.think:
                choice = extract_choice(response_text)
                is_correct = (choice == gt_answer)
            else:
                judge_prompt = f"""You are an objective evaluator.
Analyze the following multiple-choice question, the correct answer (ground truth), and a model's prediction (which might include reasoning).
Decide if the model's final conclusion matches the ground truth.

Question: {question}
Ground Truth Choice: {gt_answer}
Model Prediction: {response_text}

Based ONLY on the final choice made by the model, is it correct?
Respond with exactly one word: 'YES' if correct, 'NO' if incorrect.
Do not provide any explanation."""

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
                    choice = "###"
                    is_correct = "YES" in ans
                except Exception as e:
                    print(f"Error calling judge API for {sample_id}: {e}")
                    is_correct = False
                    choice = "###"

            print(f"########## Sample {sample_id} | Choice: {choice} | GT: {gt_answer} | Correct: {is_correct} | Tokens: {tracker_data['total_tokens']} | Time: {tracker_data['total_wall_time']:.2f}s")

            res_item = {
                "index": item.get("index"),
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
                    category_metrics[cat] = {"correct": 0, "total": 0}
                category_metrics[cat]["total"] += 1
                if is_correct:
                    category_metrics[cat]["correct"] += 1
                
                results.append(res_item)
                with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(res_item, ensure_ascii=False) + "\n")
            
            return res_item
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    print(f"Starting Parallel Enhanced Echo evaluation on MMStar with {args.num_workers} workers...")
    script_start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_item, item) for item in dataset]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time

    # Summary Report
    print("\n" + "=" * 50)
    print("ENHANCED ECHO FRAMEWORK (MMSTAR) EVALUATION REPORT")
    print("=" * 50)
    total_correct = 0
    total_samples = 0
    total_tokens = 0
    total_time = 0.0

    for cat, stats in sorted(category_metrics.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{cat:<25}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")
        total_correct += stats["correct"]
        total_samples += stats["total"]
        
    for res in results:
        if res and "metrics" in res:
            total_tokens += res["metrics"]["total_tokens"]
            total_time += res["metrics"]["total_wall_time"]

    if total_samples > 0:
        overall_acc = total_correct / total_samples * 100
        avg_tokens = total_tokens / total_samples
        avg_time = total_time / total_samples
        amortized_time = total_script_time / total_samples
        
        print("-" * 50)
        print(f"{'OVERALL ACCURACY':<25}: {overall_acc:6.2f}% ({total_correct}/{total_samples})")
        print(f"{'AVERAGE TOKENS/SAMPLE':<25}: {avg_tokens:.2f}")
        print(f"{'AVG LATENCY UNDER LOAD':<25}: {avg_time:.2f}s (Sum of indiv times / N)")
        print(f"{'AMORTIZED TIME/SAMPLE':<25}: {amortized_time:.2f}s (Total Script Time / N)")
    print("=" * 50)
    print(f"Results saved to: {output_jsonl_path}")


if __name__ == "__main__":
    main()
