import json
from collections import defaultdict


def load_jsonl(path: str) -> list:
    """读取 jsonl"""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            results.append(item)
    return results

data_list = load_jsonl("/root/autodl-tmp/mmyzh/ZoomEye/Echo/debug_enhanced_echo_eval/run_20260320_193145/echo_results.jsonl")
correct_count = defaultdict(int)
categories = defaultdict(int)

token_count = 0
time_cost = 0

for i in data_list:
    if 'metrics' in i.keys():
        token_count += i['metrics']['total_tokens']
        time_cost += i['metrics']['total_wall_time']
    # categories[i["category"]] += 1
    # categories["sum"] += 1
    # print(i['is_correct'])
    # if i['is_correct']:
    #     correct_count[i["category"]] += 1
    #     correct_count["sum"] += 1
#
# for k in categories.keys():
#     print(f"{k}: {correct_count[k]} / {categories[k]} -- {correct_count[k] / categories[k]}")
# print(f"sum: {correct_count['sum']} / {categories['sum']} -- {correct_count['sum'] / categories['sum']}")
#
print(f"TOKEN: {token_count / len(data_list)} TIME: {time_cost / len(data_list)}")