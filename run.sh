export HF_ENDPOINT=https://hf-mirror.com


python /root/autodl-tmp/2026mm_lry/MM/Echo/baseline/vstar/vstar_eval_vllm.py \
    --model-name "/root/autodl-tmp/InternVL3-8B-Instruct" \
    --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl \
    --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3 \
    --output-path /root/autodl-tmp/2026mm_lry/MM/results/baseline_internvl3_vstar_results.jsonl

vllm serve /root/autodl-tmp/InternVL3-8B-Instruct \
  --port 18904 \
  --tensor-parallel-size 1 \
  --max-model-len 20000 \
  --gpu-memory-utilization 0.8 \
  --quantization fp8 \
  --trust-remote-code

HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
  OpenGVLab/InternVL3-8B-Instruct \
  --local-dir /root/autodl-tmp/InternVL3-8B-Instruct

nvidia-smi

cd /Users/azurehans/PycharmProjects/MM/MM/Echo

python /root/autodl-tmp/2026mm_lry/MM/Echo/eval_vstar.py \
    --api-url "http://localhost:18904/v1" \
    --model-name "/root/autodl-tmp/InternVL3-8B-Instruct" \
    --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl \
    --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3 \
    --output-path /root/autodl-tmp/2026mm_lry/MM/results/0324_echo_debug_internvl3_vstar_results.jsonl \
    --num-workers 8 \
    --debug


HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
  XiaomiMiMo/MiMo-VL-7B-SFT \
  --local-dir /root/autodl-tmp/MiMo-VL-7B-SFT


vllm serve /root/autodl-tmp/MiMo-VL-7B-SFT \
  --port 18905 \
  --tensor-parallel-size 1 \
  --max-model-len 20000 \
  --gpu-memory-utilization 0.8 \
  --quantization fp8 \
  --trust-remote-code

python /root/autodl-tmp/2026mm_lry/MM/Echo/baseline/vstar/vstar_eval.py \
    --api-url "http://localhost:18905/v1" \
    --model-name "/root/autodl-tmp/MiMo-VL-7B-SFT" \
    --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl \
    --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3 \
    --output-path /root/autodl-tmp/2026mm_lry/MM/results/mimo_baseline_vstar_results.jsonl \
    --num-workers 8

python eval_vstar.py \
    --data-path /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3/test_questions.jsonl \
    --image-root /root/autodl-tmp/2026mm_lry/Echo/hf_vstar_3 \
    --api-url http://localhost:18905/v1 \
    --model-name /root/autodl-tmp/MiMo-VL-7B-SFT \
    --output-path /root/autodl-tmp/2026mm_lry/MM/results/echo_mimo_baseline_vstar_results.jsonl \
    --num-workers 8

