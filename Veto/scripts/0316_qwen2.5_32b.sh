# Normal
#python eval_vstar.py --image-root ../hf_vstar/ --data-path ../hf_vstar/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 8 --model-name qwen2.5-vl-32b --temperature 0.4
python eval_hr_bench.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 12 --model-name qwen2.5-vl-32b --temperature 0.4 --no-visual-prompt
python eval_hr_bench.py --split hrbench_8k --samples 800 --output-path echo_results.jsonl --num-workers 12 --model-name qwen2.5-vl-32b --temperature 0.4 --no-visual-prompt