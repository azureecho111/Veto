"""
analyze_repeats.py
------------------
读取 eval_hr_bench_enhanced.py 多次重复跑出的 echo_results.jsonl 文件，
分析各题目在多次评估中的稳定性（全对 / 有时对有时错 / 全错）。

默认自动寻找最新的 debug_echo_eval/run_TIMESTAMP/ 目录。
结果路径结构：
    debug_echo_eval/
    └── run_TIMESTAMP/
        ├── run_1/
        │   └── echo_results.jsonl
        ├── run_2/
        │   └── echo_results.jsonl
        └── run_3/
            └── echo_results.jsonl

使用方式：
    python analyze_repeats.py
    python analyze_repeats.py --result-dir debug_echo_eval/run_20260312_080000
    python analyze_repeats.py --verbose
"""

import argparse
import json
import os
from collections import defaultdict


def find_latest_run_dir(base_dir: str = "debug_echo_eval") -> str:
    """自动找到最新的 run_TIMESTAMP 目录"""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory '{base_dir}' not found. "
                                f"Please run eval_hr_bench_enhanced.py first, or specify --result-dir.")
    entries = [e for e in os.listdir(base_dir) if e.startswith("run_") and
               os.path.isdir(os.path.join(base_dir, e))]
    if not entries:
        raise FileNotFoundError(f"No 'run_*' directories found inside '{base_dir}'.")
    entries.sort(reverse=True)  # 字典序倒排，最新的排最前
    latest = os.path.join(base_dir, entries[0])
    print(f"[Info] Auto-selected result directory: {latest}")
    return latest


def collect_jsonl_files(root_dir: str, output_name: str) -> list[str]:
    """收集所有 run_*/echo_results.jsonl 路径并按 run 序号排序"""
    files = []
    for entry in sorted(os.listdir(root_dir)):
        if entry.startswith("run_") and os.path.isdir(os.path.join(root_dir, entry)):
            path = os.path.join(root_dir, entry, output_name)
            if os.path.exists(path):
                files.append((entry, path))
    return files


def load_results(filepath: str) -> list[dict]:
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-run eval stability.")
    parser.add_argument("--result-dir", type=str, default=None,
                        help="Root run_TIMESTAMP directory. Defaults to the latest in debug_echo_eval/.")
    parser.add_argument("--base-dir", type=str, default="debug_echo_eval",
                        help="Base directory containing run_TIMESTAMP folders (default: debug_echo_eval)")
    parser.add_argument("--output-name", type=str, default="echo_results.jsonl",
                        help="Filename of per-run result files (default: echo_results.jsonl)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print individual question_ids for each stability group")
    parser.add_argument("--unstable-output", type=str, default="unstable_samples.txt",
                        help="Path to output txt file for unstable question_ids (default: unstable_samples.txt)")
    args = parser.parse_args()

    # 1. 确定根目录
    root_dir = args.result_dir if args.result_dir else find_latest_run_dir(args.base_dir)

    # 2. 收集各 run 的 jsonl 文件
    run_files = collect_jsonl_files(root_dir, args.output_name)
    if not run_files:
        print(f"[Error] No '{args.output_name}' files found under '{root_dir}'. "
              f"Check the path or --output-name argument.")
        return

    print(f"[Info] Found {len(run_files)} run(s): {[r for r, _ in run_files]}")

    # 3. 读取所有数据
    # 结构：{ question_id -> { "category": str, "runs": { run_label: bool } } }
    # 结构：{ question_id -> { "category": str, "runs": { run_label: bool } } }
    questions: dict[str, dict] = {}
    run_labels = []
    per_run_cat_metrics: dict[str, dict[str, dict]] = {}  # run_label -> cat -> {correct, total}

    for run_label, filepath in run_files:
        run_labels.append(run_label)
        per_run_cat_metrics[run_label] = defaultdict(lambda: {"correct": 0, "total": 0})
        records = load_results(filepath)
        print(f"  [{run_label}] Loaded {len(records)} records from {filepath}")

        for rec in records:
            qid = str(rec.get("question_id", ""))
            is_correct = bool(rec.get("is_correct", False))
            cat = rec.get("category", "unknown")

            if qid not in questions:
                questions[qid] = {"category": cat, "runs": {}}
            questions[qid]["runs"][run_label] = is_correct

            per_run_cat_metrics[run_label][cat]["total"] += 1
            if is_correct:
                per_run_cat_metrics[run_label][cat]["correct"] += 1

    total_runs = len(run_labels)

    # 4. 打印每次 Run 的独立准确率
    print("\n" + "=" * 60)
    print("PER-RUN ACCURACY REPORT")
    print("=" * 60)
    all_cats = sorted({cat for m in per_run_cat_metrics.values() for cat in m})
    
    # 表头
    header = f"{'Category':<30}" + "".join(f"{rl:>18}" for rl in run_labels)
    print(header)
    print("-" * (30 + 18 * total_runs))

    for cat in all_cats:
        row = f"{cat:<30}"
        for rl in run_labels:
            stats = per_run_cat_metrics[rl][cat]
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"] * 100
                row += f"{acc:12.2f}% ({stats['correct']}/{stats['total']})"
            else:
                row += f"{'N/A':>18}"
        print(row)

    print("-" * (30 + 18 * total_runs))
    # 整体
    row = f"{'OVERALL':<30}"
    for rl in run_labels:
        tc = sum(m["correct"] for m in per_run_cat_metrics[rl].values())
        tt = sum(m["total"] for m in per_run_cat_metrics[rl].values())
        acc = tc / tt * 100 if tt > 0 else 0
        row += f"{acc:12.2f}% ({tc}/{tt})"
    print(row)
    print("=" * 60)

    # 5. 稳定性分析：按 question_id 分类
    always_correct = []
    always_wrong = []
    unstable = []

    # 只统计至少在所有 run 中都出现过的题目
    for qid, info in questions.items():
        results_per_run = [info["runs"].get(rl) for rl in run_labels]
        # 如果有 None（某次 run 缺失该题），跳过或视为 wrong
        results_per_run = [r for r in results_per_run if r is not None]
        if not results_per_run:
            continue

        n_correct = sum(results_per_run)
        n_total = len(results_per_run)

        if n_correct == n_total:
            always_correct.append(qid)
        elif n_correct == 0:
            always_wrong.append(qid)
        else:
            unstable.append(qid)

    total_qs = len(questions)

    print("\n" + "=" * 60)
    print("STABILITY ANALYSIS ACROSS ALL RUNS")
    print("=" * 60)
    print(f"Total unique questions evaluated : {total_qs}")
    print(f"  ✅ Always Correct  (全部做对) : {len(always_correct):4d}  ({len(always_correct)/total_qs*100:.1f}%)")
    print(f"  ⚠️  Unstable        (时对时错) : {len(unstable):4d}  ({len(unstable)/total_qs*100:.1f}%)")
    print(f"  ❌ Always Wrong    (全部做错) : {len(always_wrong):4d}  ({len(always_wrong)/total_qs*100:.1f}%)")
    print("=" * 60)

    # 6. 按类别拆分稳定性
    print("\nPER-CATEGORY STABILITY BREAKDOWN")
    print("-" * 60)
    cat_stability: dict[str, dict] = defaultdict(lambda: {"always_correct": 0, "unstable": 0, "always_wrong": 0, "total": 0})
    for qid in always_correct:
        cat = questions[qid]["category"]
        cat_stability[cat]["always_correct"] += 1
        cat_stability[cat]["total"] += 1
    for qid in unstable:
        cat = questions[qid]["category"]
        cat_stability[cat]["unstable"] += 1
        cat_stability[cat]["total"] += 1
    for qid in always_wrong:
        cat = questions[qid]["category"]
        cat_stability[cat]["always_wrong"] += 1
        cat_stability[cat]["total"] += 1

    hdr = f"{'Category':<30} {'Total':>6} {'✅Always OK':>12} {'⚠️Unstable':>12} {'❌Always Fail':>14}"
    print(hdr)
    print("-" * 78)
    for cat in sorted(cat_stability):
        s = cat_stability[cat]
        t = s["total"]
        print(f"{cat:<30} {t:>6} "
              f"{s['always_correct']:>6} ({s['always_correct']/t*100:4.1f}%)  "
              f"{s['unstable']:>6} ({s['unstable']/t*100:4.1f}%)  "
              f"{s['always_wrong']:>6} ({s['always_wrong']/t*100:4.1f}%)")
    print("-" * 78)
    # 总行
    gt = total_qs
    gc = len(always_correct)
    gu = len(unstable)
    gw = len(always_wrong)
    print(f"{'TOTAL':<30} {gt:>6} "
          f"{gc:>6} ({gc/gt*100:4.1f}%)  "
          f"{gu:>6} ({gu/gt*100:4.1f}%)  "
          f"{gw:>6} ({gw/gt*100:4.1f}%)")

    # 7. 导出 unstable 样本列表到 txt 文件
    unstable_output_path = args.unstable_output
    with open(unstable_output_path, 'w', encoding='utf-8') as f:
        for qid in sorted(unstable, key=lambda x: int(x) if x.isdigit() else x):
            f.write(qid + "\n")
    print(f"\n[Info] Unstable sample list ({len(unstable)} IDs) saved to: {os.path.abspath(unstable_output_path)}")

    # 8. verbose 模式：打印各组 question_id
    if args.verbose:
        print("\n--- ALWAYS CORRECT question_ids ---")
        print(", ".join(always_correct))
        print("\n--- UNSTABLE question_ids ---")
        print(", ".join(unstable))
        print("\n--- ALWAYS WRONG question_ids ---")
        print(", ".join(always_wrong))

    print("\n[Done]")


if __name__ == "__main__":
    main()
