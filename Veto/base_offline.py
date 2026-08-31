from typing import List, Optional, Tuple, Dict, Any
from PIL import Image
import io


# 从原始 base.py 导入所有共享的类和工具函数
from base import (
    EchoConfig,
    EchoForQwen,
    RequestState,
)


# ─── 扩展 EchoConfig，新增离线专用字段 ────────────────────────────────────────

_OFFLINE_DEFAULTS = {
    "max_model_len": 20000,
    "gpu_memory_utilization": 0.90,
    "max_tokens": 2048,
}

for _k, _v in _OFFLINE_DEFAULTS.items():
    if not hasattr(EchoConfig, _k):
        setattr(EchoConfig, _k, _v)


def _patch_echo_config(cfg: EchoConfig) -> EchoConfig:
    """确保离线字段存在于 config 实例上。"""
    for k, v in _OFFLINE_DEFAULTS.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ─── 离线推理类 ────────────────────────────────────────────────────────────────

class EchoForQwenOffline(EchoForQwen):
    """
    EchoForQwen 的离线版本。
    继承全部推理逻辑（find_targets / ground_target_instances / reasoning_step / generate）,
    只覆盖 __init__ 和 _generate_from_vllm，用 vllm.LLM 替代 OpenAI HTTP 客户端。
    """

    def __init__(self, config: EchoConfig = None):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        self.config = _patch_echo_config(config if config else EchoConfig())

        print(f"[EchoOffline] Loading model: {self.config.model_name}")
        print(f"[EchoOffline] max_model_len={self.config.max_model_len}, "
              f"gpu_memory_utilization={self.config.gpu_memory_utilization}")

        self.llm = LLM(
            model=self.config.model_name,
            dtype="bfloat16",
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            seed=self.config.seed,
            max_tokens=self.config.max_tokens,
        )
        print("[EchoOffline] Model loaded successfully.")

    # ------------------------------------------------------------------
    # 核心覆盖：将 PIL Image 转换成 vllm 多模态输入并调用离线推理
    # ------------------------------------------------------------------

    def _generate_from_vllm(self, prompt: str, image: Optional[Image.Image] = None) -> str:
        """
        离线推理接口。
        - 如果有图片，将其以 base64 编码嵌入 chat messages（Qwen2.5-VL 的标准格式）。
        - 使用 apply_chat_template 生成 token 序列的文本表示。
        - 通过 llm.generate() 执行推理。
        """
        # 1. 构建 messages（与在线版本保持相同的 content 格式，供 apply_chat_template 使用）
        content = []
        mm_data = {}

        if image is not None:
            # vllm 接受 PIL Image 直接作为 multi_modal_data
            content.append({
                "type": "image",
                "image": image,  # apply_chat_template 识别此格式为占位符
            })
            mm_data["image"] = image

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # 2. 将 messages 转换为模型的输入文本（含特殊 token）
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 3. 构建 vllm 的 inputs dict
        inputs = {"prompt": text}
        if mm_data:
            inputs["multi_modal_data"] = mm_data

        # 4. 推理
        outputs = self.llm.generate(inputs, self.sampling_params)
        return outputs[0].outputs[0].text

    # ------------------------------------------------------------------
    # 覆盖 find_targets：父类的 ICL 分支直接调用 self.client，需替换为离线推理
    # ------------------------------------------------------------------

    def find_targets(self, state):
        """
        与父类逻辑完全相同，但 ICL 分支使用 self._generate_from_vllm（无图片）
        替代父类的 self.client.chat.completions.create。
        """
        import os, json
        from base import PROMPTS, extract_targets
        from utils import split_targets_sentence

        if not os.path.exists(self.config.ic_path):
            print(f"[EchoOffline] IC Examples not found at {self.config.ic_path}, falling back to simple extraction.")
            p = PROMPTS["extract_targets"].format(question=state.question)
            res = self._generate_from_vllm(p)  # 无图片，纯文本
            state.targets = [t.strip().lower() for t in res.split(",")]
        else:
            with open(self.config.ic_path, 'r', encoding='utf-8') as f:
                ic_examples = json.load(f)

            ic_question_list = ic_examples["question_list"]
            ic_response_list = ic_examples["response_list"]

            prompt_text = f"""Suppose you have an image and will use it to answer the corresponding question. Please analyze what target objects in the image you need to closely examine in order to answer this question.
Rules:
    1. The object must be unambiguous; do not expand or narrow the search scope. For example, for "a woman with a hat," do not simply output "woman," as that would broaden the scope.
    2. If the object includes a quantifier, output the object directly.
    3. First conduct an analysis, then summarize all the object at the end.
Example:

{"\n\n".join([f"[Question]: {question[:state.question.find('Please answer') if 'Please answer' in state.question else None]}\n[Response]: {response}" for question, response in zip(ic_question_list, ic_response_list)])}
...
Now, please find targets in the following question:

[Question]: {state.question}
[Response]: """

            # ★ 关键改动：用 _generate_from_vllm 替代 self.client
            response_text = self._generate_from_vllm(prompt_text)
            targets_sentence = extract_targets(response_text)
            print(f"### [EchoOffline] IC Response:{response_text} || Extracted Item: {targets_sentence}")

            if targets_sentence:
                state.targets = [t.strip().lower() for t in split_targets_sentence(targets_sentence)]
            else:
                state.targets = []

            # 安全检查：如果提取出的 target 包含数学公式或特殊符号（如 + / =），说明提取解析失败，清空以退化回 baseline
            for t in state.targets:
                if any(char in t for char in ['+', '/', '=', '*', '(', ')']):
                    print(f"[Echo Safety] Detected formula in targets: '{t}'. Falling back to simple reasoning.")
                    state.targets = []
                    break

            state.add_history("extract_targets", state.targets, f"IC Learning from {self.config.ic_path}",
                               model_output=response_text)

        print(f"[EchoOffline] Targets detected: {state.targets} (Question: {state.question})")

    # reasoning_step / ground_target_instances / remove_unnecessary 全部通过继承复用，
    # 它们内部只调用 self._generate_from_vllm，已由上面的覆盖路由到离线引擎。

    # ------------------------------------------------------------------
    # Final answer 特殊处理：覆盖 generate 最后一步，也改为离线推理
    # （父类 generate 内部的 self.client.chat.completions.create 要绕过）
    # ------------------------------------------------------------------

    def generate(self, image_pil: Image.Image, question: str, custom_debug_dir: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        覆盖父类 generate，将最终 final answer 阶段的 client.chat.completions
        也替换为 llm.generate()。
        """
        import os
        import json
        from PIL import ImageDraw, ImageFont
        from utils import encode_image

        out_dir = custom_debug_dir if custom_debug_dir else self.config.debug_dir
        state = RequestState(image_pil, question, output_dir=out_dir, debug=self.config.debug)
        self.find_targets(state)

        # 默认元数据
        metadata = {
            "targets": [],
            "found_targets": [],
            "all_found": False
        }

        # 安全检查：如果没提取出 target，直接用原图回答
        if not state.targets:
            res = self._generate_from_vllm(question, image_pil)
            return res, metadata

        try:
            while not self.reasoning_step(state):
                pass
        except Exception as e:
            print(f"[EchoOffline] Exception during reasoning (e.g., crop error): {e}. Falling back to baseline.")
            res = self._generate_from_vllm(question, image_pil)
            return res, metadata
        
        metadata["targets"] = state.targets
        metadata["found_targets"] = list(state.found_targets.keys())
        metadata["all_found"] = (len(state.found_targets) == len(state.targets)) and len(state.targets) > 0

        # 如果没有找全所有 target，直接用原图回答
        if len(state.found_targets) != len(state.targets) and len(state.targets) > 0:
            res = self._generate_from_vllm(question, image_pil)
            return res, metadata

        if not state.found_targets:
            res = self._generate_from_vllm(question, image_pil)
            return res, metadata

        # Focus Crop 逻辑（与父类完全相同，复制以避免调用 self.client）
        orig_w, orig_h = state.original_image.size
        min_gl, min_gt = orig_w, orig_h
        max_gr, max_gb = 0, 0

        all_boxes = []
        for target, coords_list in state.found_targets.items():
            for box in coords_list:
                min_gl = min(min_gl, box[0])
                min_gt = min(min_gt, box[1])
                max_gr = max(max_gr, box[2])
                max_gb = max(max_gb, box[3])
                all_boxes.append((target, box))

        margin = 150
        fl = max(0, min_gl - margin)
        ft = max(0, min_gt - margin)
        fr = min(orig_w, max_gr + margin)
        fb = min(orig_h, max_gb + margin)

        try:
            fl, ft, fr, fb = int(fl), int(ft), int(fr), int(fb)
            if fr <= fl or fb <= ft: raise ValueError("Invalid focus box")
            focus_img = state.original_image.crop((fl, ft, fr, fb)).copy()
        except Exception as e:
            print(f"[EchoOffline] Exception during focus crop ({e}). Falling back to baseline.")
            res = self._generate_from_vllm(question, image_pil)
            return res, metadata
        draw = ImageDraw.Draw(focus_img)

        if self.config.visual_prompt:
            colors = ["red", "blue", "green", "magenta", "orange", "cyan"]
            target_color_map = {t: colors[i % len(colors)] for i, t in enumerate(state.found_targets.keys())}
    
            focus_w, focus_h = focus_img.size
            font_size = max(24, int(max(focus_w, focus_h) * 0.02))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
                font_size = 15
    
            line_width = max(3, int(max(focus_w, focus_h) * 0.005))
    
            for target, box in all_boxes:
                color = target_color_map[target]
                obj_w = box[2] - box[0]
                obj_h = box[3] - box[1]
                dilate_x = min(int(focus_w * 0.05), max(5, int(obj_w * 0.1)))
                dilate_y = min(int(focus_h * 0.05), max(5, int(obj_h * 0.1)))
                rel_box = [box[0] - fl - dilate_x, box[1] - ft - dilate_y,
                           box[2] - fl + dilate_x, box[3] - ft + dilate_y]
                draw.rectangle(rel_box, outline=color, width=line_width)
                text_x, text_y = rel_box[0], rel_box[1] - font_size - 5
                try:
                    rect_bbox = draw.textbbox((text_x, text_y), target, font=font)
                    draw.rectangle((rect_bbox[0]-2, rect_bbox[1]-2, rect_bbox[2]+2, rect_bbox[3]+2), fill="white")
                except Exception:
                    pass
                draw.text((text_x, text_y), target, fill=color, font=font)

        if state.debug:
            focus_img.save(os.path.join(state.output_dir, "final_focus_crop.jpg"))

        final_prompt = (
            f"Task: {question}\n\n"
            f"\nPlease analyze the focused visual information and provide the final answer to the question: {question}"
        )

        response = self._generate_from_vllm(final_prompt, focus_img)
        print("⭐ Final answer:", response)
        return response, metadata
