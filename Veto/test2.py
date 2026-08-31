import json
def load_jsonl(path: str) -> dict:
    """读取 jsonl，返回 {question_id: is_correct} 字典。"""
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = str(item.get("question_id", item.get("index", "")))
            results[qid] = bool(item.get("is_correct", False))
    return results

data = load_jsonl("/root/autodl-tmp/mmyzh/ZoomEye/Echo/debug_echo_offline/run_20260314_064331/offline_results.jsonl")
print(data.keys())
for i in range(800):
    # print(i)
    if str(i) in data.keys():
        continue
    print(i)