#python eval_vstar.py --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/ --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 20 --model-name qwen3-vl-8b
# Normal
#python baseline/vstar/vstar_eval.py --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/ --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl --samples 191 --num-workers 8 --model-name qwen3-vl-8b
#python baseline/hr_bench/hr_bench_eval.py --model-name qwen3-vl-8b --split hrbench_4k --samples 800 --num-workers 4 --api-url http://localhost:18904/v1/
#python baseline/hr_bench/hr_bench_eval.py --model-name qwen3-vl-8b --split hrbench_8k --samples 800 --num-workers 4 --api-url http://localhost:18904/v1/
#python eval_vstar.py --image-root ../hf_vstar/ --data-path ../hf_vstar/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 8 --model-name qwen3-vl-8b --api-url http://localhost:18904/v1/ --debug --no-visual-prompt
python eval_hr_bench.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 6 --model-name qwen3-vl-8b  --no-visual-prompt
python eval_hr_bench.py --split hrbench_8k --samples 800 --output-path echo_results.jsonl --num-workers 6 --model-name qwen3-vl-8b  --no-visual-prompt
