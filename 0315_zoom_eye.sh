python eval_hr_bench.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 20 --model-name qwen3-vl-8b
python eval_hr_bench.py --split hrbench_8k --samples 800 --output-path echo_results.jsonl --num-workers 20 --model-name qwen3-vl-8b
python eval_vstar.py --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/ --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 20 --model-name qwen3-vl-8b
# Normal
python baseline/vstar/vstar_eval.py --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/ --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl --samples 191 --num-workers 20 --model-name qwen3-vl-8b
python baseline/hr_bench/hr_bench_eval.py --model-name qwen3-vl-8b --split hrbench_4k --samples 800
python baseline/hr_bench/hr_bench_eval.py --model-name qwen3-vl-8b --split hrbench_8k --samples 800