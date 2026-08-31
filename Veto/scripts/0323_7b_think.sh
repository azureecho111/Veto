python eval_vstar.py --image-root ../hf_vstar/ --data-path ../hf_vstar/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 12 --no-visual-prompt --temperature 0.4 --think
python eval_vstar.py --image-root ../hf_vstar/ --data-path ../hf_vstar/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 12 --no-visual-prompt --temperature 0.4 --no-visual-prompt --think
python eval_hr_bench.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 12 --temperature 0.4 --no-visual-prompt --think
python eval_hr_bench.py --split hrbench_8k --samples 800 --output-path echo_results.jsonl --num-workers 12  --temperature 0.4 --no-visual-prompt --think
