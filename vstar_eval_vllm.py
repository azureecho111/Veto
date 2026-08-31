import argparse
import json
import os
import re
import time
import random
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from PIL import Image
from openai import OpenAI
from tqdm import tqdm
import numpy as np

def extract_choice(text: str) -> str:
    """Extract choice letter A-E from model output."""
    # Pattern to match letters in brackets or standalone
    patterns = [
        r"\(([A-E])\)",  # (A)
        r"\[([A-E])\]",  # [A]
        r"\b([A-E])\b",   # A as a whole word
    ]
    for p in patterns:
        matches = re.findall(p, text)
        if matches:
            return matches[-1].upper() # Take the last one mentions
    return ""

class APIJudge:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name

    def judge(self, question: str, ground_truth: str, response: str) -> str:
        prompt = f"""Question: {question}
Correct Answer (Label): {ground_truth}
Model Response: {response}

You are an academic grader. Your task is to extract the model's intended choice (A, B, C, D, or E) based on its response. 
If the model correctly identifies the answer, output the corresponding letter.
If the model is incorrect, output the letter it chose.
If you cannot determine the choice, output 'Invalid'.
Output ONLY the letter or 'Invalid'."""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling judge API: {e}")
            return "啦啦啦啦"

def main():

    parser = argparse.ArgumentParser(description="V* Benchmark Evaluation with vLLM and API Judge")
    
    # Model Params
    parser.add_argument("--model", type=str, required=True, help="Path to vLLM compatible model")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=30000)
    parser.add_argument("--trust-remote-code", action="store_true")

    # Dataset Params
    parser.add_argument("--data-path", type=str, required=True, help="Path to V* jsonl file")
    parser.add_argument("--image-root", type=str, required=True, help="Root folder for original images")
    parser.add_argument("--crop-root", type=str, default=None, help="Root folder for manually cropped images (priority)")
    parser.add_argument("--output-path", type=str, default="vstar_vllm_results.jsonl")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate (randomly sampled)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    # Sampling Params
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)

    # API Judge Params
    parser.add_argument("--use-judge", action="store_true", help="Use remote API judge")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--api-base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--judge-model", type=str, default="gpt-4o")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}")
    dataset = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))

    # Sampling for reproducibility
    if args.num_samples is not None and args.num_samples < len(dataset):
        print(f"Sampling {args.num_samples} samples with seed {args.seed}...")
        random.seed(args.seed)
        dataset = random.sample(dataset, args.num_samples)

    # Initialize vLLM
    print(f"Initializing vLLM with model {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        mm_processor_kwargs={
            "max_dynamic_patch": 12 # Keep it consistent for academic comparison
        } if "internvl" in args.model.lower() else None
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|endoftext|>", "</s>"] # Adjust based on model
    )

    # Prepare judge
    judge = None
    if args.use_judge:
        judge = APIJudge(args.api_key, args.api_base, args.judge_model)

    # Prepare Processor for prompt formatting (vLLM way)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    # Inference loop
    results = []
    category_metrics = {}

    print("Starting inference...")
    for item in tqdm(dataset):
        # Image Search Logic: Priority to Crop
        is_cropped = False
        image_path = None
        
        # Check Crop folder first (assuming manual crop has same basename)
        if args.crop_root:
            potential_crop_path = os.path.join(args.crop_root, os.path.basename(item['image']))
            if os.path.exists(potential_crop_path):
                image_path = potential_crop_path
                is_cropped = True
        
        # Fallback to original
        if not image_path:
            image_path = os.path.join(args.image_root, item['image'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found.")
            continue

        # More robust prompt construction using chat template if possible, or model-specific formats
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item['text']},
                ],
            }
        ]
        
        try:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            # vLLM multimodal inference
            outputs = llm.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": Image.open(image_path).convert("RGB")}
                },
                sampling_params=sampling_params
            )
            response_text = outputs[0].outputs[0].text.strip()
        except Exception as e:
            print(f"Error during inference for {item['question_id']}: {e}")
            response_text = f"有种刚正面:{e}"

        # Judgment
        choice = extract_choice(response_text)
        if args.use_judge and (not choice or len(response_text) > 10):
            # If regex fails or response is too long, use judge
            choice = judge.judge(item['text'], item['label'], response_text)
        
        is_correct = (choice == item['label'])
        
        # Statistics
        cat = item.get('category', 'unknown')
        if cat not in category_metrics:
            category_metrics[cat] = {"correct": 0, "total": 0}
        category_metrics[cat]["total"] += 1
        if is_correct:
            category_metrics[cat]["correct"] += 1

        res_item = {
            **item,
            "prediction": response_text,
            "processed_choice": choice,
            "is_correct": is_correct,
            "is_cropped": is_cropped
        }
        results.append(res_item)
        
        # Save incrementally
        with open(args.output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(res_item, ensure_ascii=False) + "\n")

    # Summary Report
    print("\n" + "="*50)
    print("V* BENCHMARK EVALUATION REPORT")
    print("="*50)
    total_correct = 0
    total_samples = 0
    total_crops = 0
    
    for cat, stats in category_metrics.items():
        acc = stats["correct"] / stats["total"] * 100
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

if __name__ == "__main__":
    main()
