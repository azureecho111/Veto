#python eval_vstar_ablation.py --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/ --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 20 --model-name qwen2.5-vl-7b --select-strategy size --debug-dir ./ablation_results
#python eval_vstar_ablation.py --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/ --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 20 --model-name qwen2.5-vl-7b --select-strategy random --debug-dir ./ablation_results
python eval_hr_bench_ablation.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 16 --select-strategy random --debug-dir ./ablation_results
python eval_hr_bench_ablation.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 16 --select-strategy size --debug-dir ./ablation_results
#4K
python eval_hr_bench_ablation.py --split hrbench_8k --samples 800 --output-path echo_results.jsonl --num-workers 16 --select-strategy random --debug-dir ./ablation_results
python eval_hr_bench_ablation.py --split hrbench_8k --samples 800 --output-path echo_results.jsonl --num-workers 16 --select-strategy size --debug-dir ./ablation_results
