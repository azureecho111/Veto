python eval_hr_bench_ablation.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 16 --max-reasoning-step 0 --temperature 0.4
python eval_hr_bench_ablation.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 16 --max-reasoning-step 1 --temperature 0.4
python eval_hr_bench_ablation.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 16 --max-reasoning-step 2 --temperature 0.4
python eval_hr_bench_ablation.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 16 --max-reasoning-step 3 --temperature 0.4
python eval_hr_bench_ablation.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 16 --max-reasoning-step 4 --temperature 0.4

