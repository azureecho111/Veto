import json
import argparse
import os

def analyze_diff(vanilla_path, echo_path):
    def load_results(path):
        results = {}
        if not os.path.exists(path):
            print(f"Error: File {path} not found.")
            return results
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                q_id = str(item.get('question_id'))
                results[q_id] = item
        return results

    vanilla_results = load_results(vanilla_path)
    echo_results = load_results(echo_path)

    print(f"Vanilla samples: {len(vanilla_results)}")
    print(f"Echo samples: {len(echo_results)}")

    common_ids = set(vanilla_results.keys()) & set(echo_results.keys())
    print(f"Common samples: {len(common_ids)}")

    regressions = [] # Correct in Vanilla, Incorrect in Echo
    improvements = [] # Incorrect in Vanilla, Correct in Echo
    both_correct = []
    both_incorrect = []

    for q_id in common_ids:
        v_item = vanilla_results[q_id]
        e_item = echo_results[q_id]

        v_correct = v_item.get('is_correct', False)
        e_correct = e_item.get('is_correct', False)

        if v_correct and not e_correct:
            regressions.append(q_id)
        elif not v_correct and e_correct:
            improvements.append(q_id)
        elif v_correct and e_correct:
            both_correct.append(q_id)
        else:
            both_incorrect.append(q_id)

    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"Improvements (Vanilla X -> Echo O): {len(improvements)}")
    print(f"Regressions  (Vanilla O -> Echo X): {len(regressions)}")
    print(f"Stayed Correct (Both O)          : {len(both_correct)}")
    print(f"Stayed Incorrect (Both X)        : {len(both_incorrect)}")
    
    vanilla_acc = (len(both_correct) + len(regressions)) / len(common_ids) * 100 if common_ids else 0
    echo_acc = (len(both_correct) + len(improvements)) / len(common_ids) * 100 if common_ids else 0
    print(f"\nVanilla Acc on common: {vanilla_acc:.2f}%")
    print(f"Echo Acc on common: {echo_acc:.2f}%")
    print(f"Net Gain: {len(improvements) - len(regressions)} samples")
    print("="*50)

    if regressions:
        print(f"\nSaving {len(regressions)} regressed IDs to 'regressions.txt'...")
        with open('regressions.txt', 'w', encoding='utf-8') as f:
            for q_id in regressions:
                f.write(f"{q_id}\n")

        print("\nDEGRADED SAMPLES (Regressions):")
        for q_id in regressions:
            v_item = vanilla_results[q_id]
            e_item = echo_results[q_id]
            print(f"- ID: {q_id}")
            print(f"  Category: {v_item.get('category')}")
            print(f"  Question: {v_item.get('text', v_item.get('question'))[:100]}...")
            print(f"  Vanilla Ans: {v_item.get('processed_choice', v_item.get('prediction'))} (Correct)")
            print(f"  Echo Ans: {e_item.get('processed_choice', e_item.get('echo_prediction'))} (Wrong)")
            if 'debug_path' in e_item:
                print(f"  Echo Debug: {e_item['debug_path']}")
            print("-" * 30)

if __name__ == "__main__":
    v_path = "vstar_vanilla_whole.jsonl"
    # 使用实际找到的文件名
    e_path = "Echo/0310_echo_whole2_FIX_ONLY_DEBUG_2.jsonl"
    
    analyze_diff(v_path, e_path)
