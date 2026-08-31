python eval_vstar_ablation.py --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/ --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 32 --max-reasoning-step 0
python eval_vstar_ablation.py --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/ --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 32 --max-reasoning-step 1
python eval_vstar_ablation.py --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/ --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 32 --max-reasoning-step 2
python eval_vstar_ablation.py --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/ --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 32 --max-reasoning-step 3
python eval_vstar_ablation.py --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/ --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 32 --max-reasoning-step 4
python eval_hr_bench_ablation.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 8 --max-reasoning-step 0 --temperature 0.4
python eval_hr_bench_ablation.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 8 --max-reasoning-step 1 --temperature 0.4
python eval_hr_bench_ablation.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 8 --max-reasoning-step 2 --temperature 0.4
python eval_hr_bench_ablation.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 8 --max-reasoning-step 3 --temperature 0.4
python eval_hr_bench_ablation.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 8 --max-reasoning-step 4 --temperature 0.4

