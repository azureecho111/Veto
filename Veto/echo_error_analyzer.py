import os
import json
import base64
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- 配置区 ---
# 请在此配置你的 OpenAI 兼容 API（用于 SOTA 验证，如 GPT-4o 或 Claude 3.5）
SOTA_API_KEY = os.getenv("JUDGE_API_KEY", "EMPTY")
SOTA_BASE_URL = "https://yunwu.ai/v1"
SOTA_MODEL_NAME = "gemini-3-pro-preview"
DEBUG_ROOT = "/root/autodl-tmp/mmyzh/ZoomEye/Echo/debug_echo_eval/run_20260319_095841/trajectories"
RESULTS_JSON_PATH = "/root/autodl-tmp/mmyzh/ZoomEye/Echo/debug_echo_eval/run_20260319_095841/echo_results.jsonl"  # 请确保这个路径正确
MAX_WORKERS = 8
client = OpenAI(api_key=SOTA_API_KEY, base_url=SOTA_BASE_URL)

cache_file = "./error_analysis_detailed_2.json"
cache = []
if os.path.exists(cache_file):
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load cache from {cache_file}: {e}")
        cache = []
# cache = []


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_sota_model(image_path: str, question: str) -> str:
    """使用 SOTA 模型验证最终裁切图是否包含足够信息"""
    b64_image = encode_image(image_path)
    question = question[:question.find("Please answer")]
    try:
        response = client.chat.completions.create(
            model=SOTA_MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text",
                     "text": f"Question: {question}\n Wrap your final answer (ONLY the option letter) in <answer></answer> .If the image misses information for answering the question, output 'MISS'."},
                ]
            }],
            temperature=0.0
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


def analyze_sample(sample_id: str, question: str, ground_truth: str) -> Dict[str, Any]:
    sample_dir = os.path.join(DEBUG_ROOT, sample_id)
    traj_path = os.path.join(sample_dir, "trajectory.json")
    focus_crop_path = os.path.join(sample_dir, "final_focus_crop.jpg")
    
    analysis_res = {
        "category": "ERROR: Unknown",
        "sota_output": None,
        "sota_extracted_answer": None,
        "sota_is_correct": None
    }

    if not os.path.exists(traj_path):
        analysis_res["category"] = "ERROR: Trajectory missing"
        return analysis_res

    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            trajectory = json.load(f)
    except Exception as e:
        analysis_res["category"] = f"ERROR: Failed to read trajectory: {e}"
        return analysis_res

    # 1. 尝试分析 (a) 目标提取出错
    first_step = trajectory[0] if trajectory else None
    targets = []
    if first_step and first_step.get("action") == "extract_targets":
        targets = first_step.get("result", [])

    if not targets or len(targets) == 0:
        print(sample_id)
        analysis_res["category"] = "(a) Target Extraction Error (Empty extraction)"
        return analysis_res

    # 2. 检查最终裁切图是否存在 -> 区分 (b) 和 (c)
    if os.path.exists(focus_crop_path):
        # 存在最终裁切图，说明 Echo 认为自己找全了
        sota_pred = None
        for j in cache:
            if str(sample_id) == str(j.get('question_id')):
                sota_pred = j.get('sota_output')
                if sota_pred:
                    print(f"[*] Found cached SOTA result for {sample_id}")
                    break
        
        if sota_pred is None:
            print(f"[*] Calling SOTA for Sample {sample_id}...")
            sota_pred = call_sota_model(focus_crop_path, question)
        
        analysis_res["sota_output"] = sota_pred
        
        # 提取答案
        answer_pred = "TTT"
        match = re.search(r"<answer>(.*?)</answer>", sota_pred, re.DOTALL)
        if match:
            answer_pred = match.group(1).strip()
        else:
            # 如果没找到 <answer>，尝试看看是不是直接输出了 MISS
            if "MISS" in sota_pred.upper():
                answer_pred = "MISS"
            else:
                answer_pred = sota_pred.strip()[:100] # 截断一下防止太长
        
        analysis_res["sota_extracted_answer"] = answer_pred
        
        # 判断正确性
        sota_is_correct = False
        if answer_pred != "TTT" and answer_pred != "MISS":
            gt_str = str(ground_truth).lower()
            ans_str = str(answer_pred).lower()
            # print(gt_str, ans_str)
            if gt_str in ans_str or ans_str in gt_str:
                sota_is_correct = True
        
        analysis_res["sota_is_correct"] = sota_is_correct

        if sota_is_correct:
            analysis_res["category"] = "(c) Model Capability Limitation (SOTA correct, Echo failed final gen)"
        else:
            # print(f"##################################\n {sample_id} QUESTION: {question} \n GT: {ground_truth} \n SOTA: {sota_pred} \n########")
            analysis_res["category"] = "(?) SOTA also failed/Ambiguous (Manual Review needed)"
        return analysis_res
    else:
        # 不存在最终裁切图 -> (b) 目标定位失败
        found_targets = []
        if trajectory:
            last_step = trajectory[-1]
            found_targets = last_step.get("found_targets", [])

        if len(found_targets) < len(targets):
            # print(sample_id)
            analysis_res["category"] = f"(b) Target Localization Failure (Found {len(found_targets)}/{len(targets)})"
        else:
            analysis_res["category"] = "(b) Hallucination/Logic failure (Path truncated before crop)"
        return analysis_res

def load_jsonl(path: str) -> list:
    """读取 jsonl，返回 {question_id: is_correct} 字典。"""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            results.append(item)
    return results
def main():
    if not os.path.exists(RESULTS_JSON_PATH):
        print(f"Results file NOT found at {RESULTS_JSON_PATH}")
        return
    results = load_jsonl(RESULTS_JSON_PATH)
    print(results)
    summary = {
        "(a) Target Extraction Error": 0,
        "(b) Target Localization Failure": 0,
        "(c) Model Capability Limitation": 0,
        "(?) SOTA also failed/Ambiguous": 0,
        "ERROR": 0
    }
    detailed_results = []
    # 过滤出 Echo 预测错误的样本
    wrong_samples = [s for s in results if not s.get("is_correct", False)]

    print(f"Total wrong samples to analyze: {len(wrong_samples)}")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_sample = {
            executor.submit(
                analyze_sample, 
                str(sample.get("question_id")), 
                sample.get("text"), 
                sample.get("label")
            ): sample 
            for sample in wrong_samples
        }

        for future in tqdm(as_completed(future_to_sample), total=len(wrong_samples)):
            sample = future_to_sample[future]
            sample_id = str(sample.get("question_id"))
            question = sample.get("text")
            # 优先使用 label，如果没有则使用 answer
            gt = sample.get("label") if sample.get("label") is not None else sample.get("answer")
            
            try:
                analysis_result = future.result()
                category = analysis_result["category"]
            except Exception as e:
                category = f"ERROR: {str(e)}"
                analysis_result = {
                    "category": category,
                    "sota_output": None,
                    "sota_extracted_answer": None,
                    "sota_is_correct": None
                }

            # 统计
            found_summary = False
            for key in summary.keys():
                if category.startswith(key):
                    summary[key] += 1
                    found_summary = True
                    break
            if not found_summary:
                summary["ERROR"] += 1

            detailed_results.append({
                "question_id": sample_id,
                "category": category,
                "question": question,
                "answer": gt,
                "echo_prediction": sample.get("echo_prediction"),
                "sota_output": analysis_result.get("sota_output"),
                "sota_extracted_answer": analysis_result.get("sota_extracted_answer"),
                "sota_is_correct": analysis_result.get("sota_is_correct")
            })
    # 输出统计报告
    print("\n" + "=" * 40)
    print("      ECHO ERROR ANALYSIS REPORT")
    print("=" * 40)
    for k, v in summary.items():
        percentage = (v / len(wrong_samples) * 100) if wrong_samples else 0
        print(f"{k:<35}: {v:>3} ({percentage:>5.1f}%)")
    print("=" * 40)
    # 保存详细报告
    with open("error_analysis_detailed_2.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)
    print("\nDetailed analysis saved to: error_analysis_detailed.json")


if __name__ == "__main__":
    main()
