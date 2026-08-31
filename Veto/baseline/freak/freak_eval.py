import argparse
import json
import os
import sys
import re
import time
import random
import concurrent.futures
import threading
import itertools
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

def extract_choice(text: str) -> str:
    """Extract choice letter A-D from model output."""
    patterns = [
        r"\(([A-D])\)",  # (A)
        r"\[([A-D])\]",  # [A]
        r"\b([A-D])\b",  # A as a whole word
    ]
    for p in patterns:
        matches = re.findall(p, text)
        if matches:
            return matches[-1].upper()
    return ""

def encode_pil_image(img: Image.Image):
    img = img.convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description="Baseline FREAK Evaluation ONLINE via API (Enhanced Tracking with Permutation)")
    
    # API Params
    parser.add_argument("--api-url", type=str, default="http://localhost:18903/v1", help="API base URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model-name", type=str, default="qwen2.5-vl-7b", help="Model name for reasoning")

    # FREAK dataset config
    parser.add_argument("--hf-repo", type=str, default="/root/autodl-tmp/freak")
    parser.add_argument("--hf-split", type=str, default="test")

    # Output config
    parser.add_argument("--output-path", type=str, default="baseline_freak_online_results.jsonl")
    parser.add_argument("--samples", type=int, default=None, help="Number of random samples to eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers for evaluation")

    # Sampling Params
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=128)

    # Eval Mode
    parser.add_argument("--think", action="store_true", help="Use GPT-4o judging for chain-of-thought outputs")

    # Debug / Output config
    parser.add_argument("--debug-dir", type=str, default="debug_baseline_freak")

    args = parser.parse_args()

    # Create run-specific debug dir
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_debug_root = os.path.join(args.debug_dir, f"run_{run_timestamp}")
    os.makedirs(current_debug_root, exist_ok=True)
    os.makedirs(os.path.join(current_debug_root, "trajectories"), exist_ok=True)

    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "baseline_freak_online",
        "api_url": args.api_url,
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

    print(f"Loading FREAK dataset from HuggingFace (`{args.hf_repo}` '{args.hf_split}' split) ...")
    hf_dataset = load_dataset(args.hf_repo, "default", split=args.hf_split)
    full_dataset = list(hf_dataset)

    # Filter by type == 'mcq'
    dataset = [item for item in full_dataset if item.get("type") == "mcq"]
    if not dataset:
        dataset = [item for item in full_dataset if "options" in item and len(item["options"]) == 4]

    if args.samples and args.samples < len(dataset):
        random.seed(args.seed)
        dataset = random.sample(dataset, args.samples)

    if not dataset:
        print("No samples to evaluate.")
        return

    print(f"Total MCQ samples to evaluate: {len(dataset)}")

    online_client = OpenAI(
        base_url=args.api_url,
        api_key=args.api_key
    )

    results = []
    category_metrics = {"total": 0, "correct_avg": 0, "circular_correct": 0}
    metrics_lock = threading.Lock()
    output_jsonl_path = os.path.join(current_debug_root, args.output_path)

    def process_item(item):
        sample_id = str(item.get('id', random.randint(0, 1000000)))
        question_base = item['question']
        gt_text = item['ground_truth'].strip()
        options = item['options']
        img_pil = item['image']

        if len(options) != 4:
            print(f"Skipping sample {sample_id} because options len is {len(options)}, not 4.")
            return

        b64_image = encode_pil_image(img_pil)

        # Permutation Cycling for the first 3 options
        first_three = options[:3]
        last_one = options[3]
        perms = list(itertools.permutations(first_three))

        sample_tokens = 0
        sample_time = 0.0
        correct_count = 0
        
        permutation_records = []

        for p_idx, perm in enumerate(perms):
            current_options = list(perm) + [last_one]
            
            try:
                gt_index = current_options.index(gt_text)
            except ValueError:
                print(f"Warning: GT not found in options for sample {sample_id}. Will skip.")
                return
                
            gt_letter = chr(ord('A') + gt_index)

            # Build question
            question_full = question_base + "\nOptions:\n"
            for opt_idx, opt_text in enumerate(current_options):
                question_full += f"({chr(ord('A') + opt_idx)}) {opt_text}\n"

            if args.think:
                pass
            else:
                question_full += "\n Please answer with the option's letter from the given choices directly."

            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                {"type": "text", "text": question_full}
            ]

            p_start = time.time()
            try:
                res = online_client.chat.completions.create(
                    model=args.model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    seed=args.seed
                )
                response_text = res.choices[0].message.content.strip()
                p_elapsed = time.time() - p_start
                if hasattr(res, 'usage') and res.usage:
                    sample_tokens += getattr(res.usage, 'total_tokens', 0)
                sample_time += p_elapsed
            except Exception as e:
                p_elapsed = time.time() - p_start
                sample_time += p_elapsed
                print(f"Error during API inference for {sample_id} perm {p_idx}: {e}")
                response_text = f"Error: {e}"

            # Judicial parsing
            if not args.think:
                choice = extract_choice(response_text)
                is_correct = (choice == gt_letter)
            else:
                judge_prompt = f"""You are an objective evaluator.
Analyze the following multiple-choice question, the correct answer (ground truth), and a model's prediction.
Decide if the model's final conclusion matches the ground truth.

Question: {question_full}
Ground Truth Choice: {gt_letter}
Model Prediction: {response_text}

Based ONLY on the final choice made by the model, is it correct?
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
                    choice = "###"
                    is_correct = "YES" in ans
                except Exception as e:
                    print(f"Error calling judge API for {sample_id}: {e}")
                    is_correct = False
                    choice = "###"

            if is_correct:
                correct_count += 1

            permutation_records.append({
                "permutation": p_idx,
                "options": current_options,
                "gt_letter": gt_letter,
                "prediction": response_text,
                "processed_choice": choice,
                "is_correct": is_correct
            })

        print(f"########## Sample {sample_id} | Correct: {correct_count}/6 | Tokens: {sample_tokens} | Time: {sample_time:.2f}s")
        
        res_item = {
            "id": sample_id,
            "question": question_base,
            "ground_truth": gt_text,
            "original_options": options,
            "circular_correct": (correct_count == 6),
            "correct_count": correct_count,
            "permutations": permutation_records,
            "metrics": {
                "total_tokens": sample_tokens,
                "total_wall_time": sample_time
            }
        }

        with metrics_lock:
            category_metrics["total"] += 1
            category_metrics["correct_avg"] += correct_count
            if correct_count == 6:
                category_metrics["circular_correct"] += 1

            results.append(res_item)
            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(res_item, ensure_ascii=False) + "\n")

    print(f"Starting Parallel API evaluation on FREAK with {args.num_workers} workers...")
    script_start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_item, item) for item in dataset]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time

    # Summary Report
    print("\n" + "="*80)
    print("BASELINE FREAK ONLINE EVALUATION REPORT (Permutation Cycling)")
    print("="*80)

    total_samples = category_metrics["total"]
    total_tokens = 0
    total_time = 0.0

    for r in results:
        if "metrics" in r:
            total_tokens += r["metrics"]["total_tokens"]
            total_time += r["metrics"]["total_wall_time"]

    if total_samples > 0:
        avg_score = category_metrics["correct_avg"] / (total_samples * 6) * 100
        circ_acc = category_metrics["circular_correct"] / total_samples * 100
        
        avg_tokens = total_tokens / total_samples
        avg_time = total_time / total_samples
        amortized_time = total_script_time / total_samples
        
        print(f"{'OVERALL AVERAGE ACCURACY':<25}: {avg_score:6.2f}% ({category_metrics['correct_avg']}/{total_samples * 6} permutations)")
        print(f"{'CIRCULAR ACCURACY':<25}: {circ_acc:6.2f}% ({category_metrics['circular_correct']}/{total_samples} samples passed all 6)")
        print("="*80)
        print(f"{'AVERAGE TOKENS/SAMPLE':<25}: {avg_tokens:.2f} (Total for 6 permutations)")
        print(f"{'AVG LATENCY UNDER LOAD':<25}: {avg_time:.2f}s (Total for 6 permutations)")
        print(f"{'AMORTIZED TIME/SAMPLE':<25}: {amortized_time:.2f}s (Total Script Time / N)")
    print("="*80)
    print(f"Results saved to: {output_jsonl_path}")

if __name__ == "__main__":
    main()
