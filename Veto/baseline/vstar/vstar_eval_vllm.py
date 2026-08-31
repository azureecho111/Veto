import argparse
import json
import os
import sys
import re
import time
import random
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from PIL import Image
from openai import OpenAI
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Baseline V* Evaluation with vLLM (Metadata & Think Mode Support)")
    
    # Model Params
    parser.add_argument("--model-name", type=str, required=True, help="Path to vLLM compatible model")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=30000)
    parser.add_argument("--trust-remote-code", action="store_true")

    # Dataset Params
    parser.add_argument("--data-path", type=str, required=True, help="Path to V* jsonl file")
    parser.add_argument("--image-root", type=str, required=True, help="Root folder for original images")
    parser.add_argument("--crop-root", type=str, default=None, help="Root folder for manually cropped images (priority)")
    parser.add_argument("--output-path", type=str, default="baseline_vstar_results.jsonl")
    parser.add_argument("--samples", type=int, default=None, help="Number of random samples to eval")
    parser.add_argument("--sample-id", type=str, default=None, help="Evaluate a specific question_id")
    parser.add_argument("--sample-list", type=str, default=None, help="Path to a txt file containing question_ids to evaluate")
    parser.add_argument("--eval-failed", action="store_true", help="Only evaluate cases where the model failed in previous log")
    parser.add_argument("--failed-log-path", type=str, default="", help="Path to the previous results file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Sampling Params
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)

    # Eval Mode
    parser.add_argument("--think", action="store_true", help="Use GPT-4o judging for chain-of-thought outputs")

    # Debug / Output config
    parser.add_argument("--debug-dir", type=str, default="debug_baseline_vstar")

    args = parser.parse_args()

    # Create run-specific debug dir
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_debug_root = os.path.join(args.debug_dir, f"run_{run_timestamp}")
    os.makedirs(current_debug_root, exist_ok=True)
    os.makedirs(os.path.join(current_debug_root, "trajectories"), exist_ok=True)

    # 保存 Metadata
    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "baseline_vllm",
        "data_path": args.data_path,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "samples": args.samples,
        "think": args.think,
        "command": "python " + " ".join(sys.argv),
        "args": vars(args)
    }
    with open(os.path.join(current_debug_root, "metadata.json"), "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=4, ensure_ascii=False)

    print(f"Loading data from {args.data_path}")
    dataset = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))

    # Filter or Sample
    if args.sample_id:
        dataset = [item for item in dataset if str(item.get('question_id')) == str(args.sample_id)]
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
        dataset = [item for item in dataset if str(item.get('question_id')) in failed_ids]
    elif args.sample_list:
        if not os.path.exists(args.sample_list):
            print(f"Error: Sample list {args.sample_list} not found.")
            return
        with open(args.sample_list, 'r', encoding='utf-8') as f:
            target_ids = {line.strip() for line in f if line.strip()}
        dataset = [item for item in dataset if str(item.get('question_id')) in target_ids]
    elif args.samples and args.samples < len(dataset):
        random.seed(args.seed)
        dataset = random.sample(dataset, args.samples)

    if not dataset:
        print("No samples to evaluate.")
        return

    print(f"Total samples to evaluate: {len(dataset)}")

    print(f"Initializing vLLM with model {args.model_name}")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        limit_mm_per_prompt={"image": 1}
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed
    )

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)

    results = []
    category_metrics = {}
    output_jsonl_path = os.path.join(current_debug_root, args.output_path)

    print("Starting inference...")
    for item in tqdm(dataset):
        sample_id = str(item.get('question_id', item.get('index', random.randint(0, 1000000))))
        
        is_cropped = False
        image_path = None
        if args.crop_root:
            potential_crop_path = os.path.join(args.crop_root, os.path.basename(item['image']))
            if os.path.exists(potential_crop_path):
                image_path = potential_crop_path
                is_cropped = True
        
        if not image_path:
            image_path = os.path.join(args.image_root, item['image'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found.")
            continue

        # 构造 prompt (跟 eval_vstar 保持一致的调整逻辑)
        question = item['text']
        if args.think:
            instruction_to_remove = "Answer with the option's letter from the given choices directly."
            question = question.replace(instruction_to_remove, "").strip()
        else:
            if "Answer with the option's letter" not in question:
                question += "\n Please answer with the option's letter from the given choices directly."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        try:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            outputs = llm.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": Image.open(image_path).convert("RGB")}
                },
                sampling_params=sampling_params
            )
            response_text = outputs[0].outputs[0].text.strip()
        except Exception as e:
            print(f"Error during inference for {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            response_text = f"Error: {e}"

        # 判题
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

        print(f"########## Sample {sample_id} | Choice: {choice} | GT: {item['label']} | Correct: {is_correct}")

        cat = item.get('category', 'unknown')
        if cat not in category_metrics:
            category_metrics[cat] = {"correct": 0, "total": 0}
        category_metrics[cat]["total"] += 1
        if is_correct:
            category_metrics[cat]["correct"] += 1

        local_debug_dir = os.path.join(current_debug_root, "trajectories", sample_id)
        os.makedirs(local_debug_dir, exist_ok=True)
        
        # 存下结果
        res_item = {
            **item,
            "prediction": response_text,
            "echo_prediction": response_text,
            "processed_choice": choice,
            "is_correct": is_correct,
            "is_cropped": is_cropped,
            "debug_path": os.path.abspath(local_debug_dir)
        }
        results.append(res_item)

        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(res_item, ensure_ascii=False) + "\n")

    # Summary Report
    print("\n" + "="*50)
    print("BASELINE V* EVALUATION REPORT")
    print("="*50)
    total_correct = 0
    total_samples = 0
    total_crops = 0
    
    for cat, stats in sorted(category_metrics.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{cat:<25}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")
        total_correct += stats["correct"]
        total_samples += stats["total"]

    for r in results:
        if r.get('is_cropped', False):
            total_crops += 1

    if total_samples > 0:
        overall_acc = total_correct / total_samples * 100
        print("-" * 50)
        print(f"{'OVERALL':<25}: {overall_acc:6.2f}% ({total_correct}/{total_samples})")
        print(f"{'Cropped Samples USED':<25}: {total_crops}")
    print("="*50)
    print(f"Results saved to: {output_jsonl_path}")

if __name__ == "__main__":
    main()
