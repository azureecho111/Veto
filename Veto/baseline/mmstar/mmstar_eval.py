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

def encode_pil_image(img: Image.Image):
    img = img.convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description="Baseline MMStar Evaluation ONLINE via API (Enhanced Tracking)")
    
    # API Params
    parser.add_argument("--api-url", type=str, default="http://localhost:18903/v1", help="API base URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model-name", type=str, default="qwen2.5-vl-7b", help="Model name for reasoning")

    # Output config
    parser.add_argument("--output-path", type=str, default="baseline_mmstar_online_results.jsonl")
    parser.add_argument("--samples", type=int, default=None, help="Number of random samples to eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers for evaluation")

    # Sampling Params
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=2048)

    # Eval Mode
    parser.add_argument("--think", action="store_true", help="Use GPT-4o judging for chain-of-thought outputs")

    # Debug / Output config
    parser.add_argument("--debug-dir", type=str, default="debug_baseline_mmstar")

    args = parser.parse_args()

    # Create run-specific debug dir
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_debug_root = os.path.join(args.debug_dir, f"run_{run_timestamp}")
    os.makedirs(current_debug_root, exist_ok=True)
    os.makedirs(os.path.join(current_debug_root, "trajectories"), exist_ok=True)

    # 保存 Metadata
    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "baseline_mmstar_online",
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

    print("Loading MMStar dataset from HuggingFace (`Lin-Chen/MMStar` 'val' split) ...")
    hf_dataset = load_dataset("Lin-Chen/MMStar", split="val")
    dataset = list(hf_dataset)

    if args.samples and args.samples < len(dataset):
        random.seed(args.seed)
        dataset = random.sample(dataset, args.samples)

    if not dataset:
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
        question = item['question']
        gt_answer = str(item['answer']).strip()
        img_pil = item['image']

        if args.think:
            instruction_to_remove = "Answer with the option's letter from the given choices directly."
            question = question.replace(instruction_to_remove, "").strip()
        else:
            if "Answer with the option's letter" not in question:
                question += "\n Please answer with the option's letter from the given choices directly."

        start_time = time.time()
        tokens = 0
        elapsed_time = 0.0

        try:
            b64_image = encode_pil_image(img_pil)
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
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
            elapsed_time = time.time() - start_time
            if hasattr(res, 'usage') and res.usage:
                tokens = getattr(res.usage, 'total_tokens', 0)
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"Error during API inference for {sample_id}: {e}")
            response_text = f"Error: {e}"

        # 判题
        if not args.think:
            choice = extract_choice(response_text)
            is_correct = (choice == gt_answer)
        else:
            judge_prompt = f"""You are an objective evaluator.
Analyze the following multiple-choice question, the correct answer (ground truth), and a model's prediction (which might include reasoning).
Decide if the model's final conclusion matches the ground truth.

Question: {question}
Ground Truth Answer: {gt_answer}
Model Prediction: {response_text}

Based ONLY on the final conclusion made by the model, is it correct?
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
                is_correct = "YES" in ans
            except Exception as e:
                print(f"Error calling judge API for {sample_id}: {e}")
                is_correct = False
            choice = "###"

        print(f"########## Sample {sample_id} | Choice: {choice} | GT: {gt_answer} | Correct: {is_correct} | Tokens: {tokens} | Time: {elapsed_time:.2f}s")

        # Create res_item without the PIL image to make it JSON serializable
        res_item = {
            "index": item.get("index"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "category": item.get("category", "unknown"),
            "prediction": response_text,
            "processed_choice": choice,
            "is_correct": is_correct,
            "metrics": {
                "total_tokens": tokens,
                "total_wall_time": elapsed_time
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

    print(f"Starting Parallel API evaluation on MMStar with {args.num_workers} workers...")
    script_start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_item, item) for item in dataset]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time

    # Summary Report
    print("\n" + "="*50)
    print("BASELINE MMSTAR ONLINE EVALUATION REPORT")
    print("="*50)
    total_correct = 0
    total_samples = 0
    total_tokens = 0
    total_time = 0.0
    
    for cat, stats in sorted(category_metrics.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{cat:<25}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")
        total_correct += stats["correct"]
        total_samples += stats["total"]

    for r in results:
        if "metrics" in r:
            total_tokens += r["metrics"]["total_tokens"]
            total_time += r["metrics"]["total_wall_time"]

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
    print("="*50)
    print(f"Results saved to: {output_jsonl_path}")

if __name__ == "__main__":
    main()
