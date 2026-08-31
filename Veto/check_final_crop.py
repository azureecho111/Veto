"""
check_final_crop.py
-------------------
统计指定目录下每个子文件夹中是否存在 final_focus_crop.jpg 文件。

使用方式：
  python check_final_crop.py <directory>
  python check_final_crop.py          # 不传参数时会提示输入
"""
import json
import os
import sys

TARGET_FILE = "final_focus_crop.jpg"

def check_directory(root_dir: str):
    if not os.path.isdir(root_dir):
        print(f"[Error] Not a valid directory: {root_dir}")
        sys.exit(1)

    subdirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    subdirs.sort()

    total = len(subdirs)
    has_crop = []
    no_crop = []
    contain_final_crop_correct = 0
    no_contain_final_crop_correct = 0

    dataset = []
    with open(os.path.join(root_dir, "echo_results.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))

    for subdir in subdirs:
        crop_path = os.path.join(root_dir, subdir, TARGET_FILE)
        if os.path.isfile(crop_path):
            has_crop.append(subdir)
            for j in dataset:
                if j['question_id'] == subdir:
                    if j['is_correct']:
                        contain_final_crop_correct += 1

        else:
            no_crop.append(subdir)
            for j in dataset:
                if j['question_id'] == subdir:
                    if j['is_correct']:
                        no_contain_final_crop_correct += 1

    if total > 0:
        pct = len(has_crop) / total * 100
        print(f"Coverage                  : {pct:.1f}%")

    if no_crop:
        print("\n[Subdirs WITHOUT final_focus_crop.jpg]")
        for d in no_crop:
            print(f"  - {d}")
    print(f"\nDirectory : {os.path.abspath(root_dir)}")
    print(f"Target    : {TARGET_FILE}")
    print("=" * 50)
    print(f"Total subdirectories      : {total}")
    print(f"With    {TARGET_FILE:25s}: {len(has_crop)}")
    print(f"Without {TARGET_FILE:25s}: {len(no_crop)}")
    print(contain_final_crop_correct, "||||", no_contain_final_crop_correct)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = input("请输入目录路径: ").strip().strip('"')

    check_directory(root)
