python eval_vstar.py --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/ --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl --samples 191 --output-path echo_results.jsonl --num-workers 12 --no-visual-prompt --temperature 0.4
python eval_hr_bench.py --split hrbench_4k --samples 800 --output-path echo_results.jsonl --num-workers 12 --temperature 0.4 --no-visual-prompt
python eval_hr_bench.py --split hrbench_8k --samples 800 --output-path echo_results.jsonl --num-workers 12  --temperature 0.4 --no-visual-prompt
