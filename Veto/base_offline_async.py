from typing import List, Optional, Tuple, Dict, Any
from PIL import Image, ImageDraw
import os
import json
import asyncio
import uuid

# 导入基础组件
from base import (
    EchoConfig,
    RequestState,
    PROMPTS,
    extract_labeled_bboxes,
    map_coords_to_orig,
    calculate_min_distance_point_to_bbox,
    extract_targets,
    split_targets_sentence,
    split_targets_sentence,
    smart_resize,
)
from utils import encode_image


class EchoForQwenAsync:
    """
    Echo 框架的异步增强版。
    使用 vllm.AsyncLLMEngine 实现高吞吐量推理。
    所有推理步骤均为 async，支持跨样本并行。
    """

    def __init__(self, config: EchoConfig = None):
        from vllm import AsyncLLMEngine, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from transformers import AutoProcessor

        self.config = config if config else EchoConfig()

        # 确保离线特有配置存在
        if not hasattr(self.config, "max_model_len"): self.config.max_model_len = 20000
        if not hasattr(self.config, "gpu_memory_utilization"): self.config.gpu_memory_utilization = 0.90
        if not hasattr(self.config, "max_tokens"): self.config.max_tokens = 2048

        print(f"[EchoAsync] Loading model with AsyncEngine: {self.config.model_name}")

        engine_args = AsyncEngineArgs(
            model=self.config.model_name,
            dtype="bfloat16",
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=True,
            # 异步引擎通常运行在单独的 loop 或线程中
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            seed=self.config.seed,
            max_tokens=self.config.max_tokens,
        )
        print("[EchoAsync] Async Engine loaded successfully.")

    async def _generate_from_vllm(self, prompt: str, image: Optional[Image.Image] = None) -> str:
        """异步产生模型响应"""
        content = []
        mm_data = {}

        if image is not None:
            content.append({"type": "image", "image": image})
            mm_data["image"] = image

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = {"prompt": text}
        if mm_data:
            inputs["multi_modal_data"] = mm_data

        request_id = str(uuid.uuid4())
        results_generator = self.engine.generate(inputs, self.sampling_params, request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        return final_output.outputs[0].text

    async def find_targets(self, state: RequestState):
        """异步版目标提取"""
        if not os.path.exists(self.config.ic_path):
            p = PROMPTS["extract_targets"].format(question=state.question)
            res = await self._generate_from_vllm(p)
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
    3. First conduct an analysis, then summarize the answer at the end.
Example:

{"\n".join([f"[Question]: {question}\n[Response]: {response}" for question, response in zip(ic_question_list, ic_response_list)])}
...
Now, please answer:
[Question]: {state.question}
[Response]: """

            response_text = await self._generate_from_vllm(prompt_text)
            targets_sentence = extract_targets(response_text)

            if targets_sentence:
                # 注意：这里直接使用了 base.py 里的 split_targets_sentence，确保一致
                state.targets = [t.strip().lower() for t in split_targets_sentence(targets_sentence)]
            else:
                state.targets = []

            state.add_history("extract_targets", state.targets, f"IC Learning from {self.config.ic_path}",
                              model_output=response_text)

        if state.debug:
            print(f"[EchoAsync] Targets detected: {state.targets}")

    def is_end_loop(self, state: RequestState) -> bool:
        """同步判断"""
        w, h = state.current_image.size
        if w < self.config.min_image_size or h < self.config.min_image_size: return True
        if max(w / h, h / w) > self.config.max_aspect_ratio: return True
        if state.step_count >= self.config.max_steps: return True
        if len(state.found_targets) == len(state.targets) and len(state.targets) > 0: return True
        return False

    async def remove_unnecessary(self, state: RequestState, mode: str = "efficient"):
        """异步版 Squeeze 逻辑"""
        all_targets = ", ".join([i for i in state.targets if i not in state.found_targets.keys()])
        cw, ch = state.current_image.size

        if mode == "efficient":
            nav_points = []
            all_targets_list = [i for i in state.targets if i not in state.found_targets.keys()]
            if hasattr(state, 'current_nav_points'):
                for t in all_targets_list:
                    if t in state.current_nav_points:
                        nav_points.append(state.current_nav_points[t])

            p_squeeze = PROMPTS["squeeze_efficient"].format(
                question=state.question[:state.question.find("Please answer")],
                targets=all_targets
            )
            res_sq = await self._generate_from_vllm(p_squeeze, state.current_image)
            bbox_dict = extract_labeled_bboxes(res_sq)

            best_bbox, max_score, best_label = None, -1, None
            candidate_status = {}

            for label, coords in bbox_dict.items():
                if len(coords) != 4: continue
                l_orig, t_orig, r_orig, b_orig = map_coords_to_orig(coords, cw, ch)
                box_pixel = [l_orig, t_orig, r_orig, b_orig]

                if label == state.last_pruned_edge: continue

                new_w, new_h = cw, ch
                if label == "top":
                    new_h = ch - b_orig
                elif label == "bottom":
                    new_h = t_orig
                elif label == "left":
                    new_w = cw - r_orig
                elif label == "right":
                    new_w = l_orig

                if new_w <= 0 or new_h <= 0 or max(new_w / new_h, new_h / new_w) > self.config.max_aspect_ratio:
                    continue

                min_dist = float('inf')
                if nav_points:
                    for np in nav_points:
                        d = calculate_min_distance_point_to_bbox(np, box_pixel)
                        min_dist = min(min_dist, d)
                else:
                    min_dist = 0

                if nav_points and min_dist <= 0: continue

                if min_dist > max_score:
                    max_score, best_bbox, best_label = min_dist, coords, label
                candidate_status[label] = ("valid", box_pixel)

            if best_bbox:
                state.add_history("squeeze_analyze", bbox_dict, f"Pick: {best_label}",
                                  model_output=f"Squeeze: {res_sq}")
                return self._apply_pruning(state, best_bbox, target_edge=best_label)
            return False
        return False

    def _apply_pruning(self, state: RequestState, bbox_raw: List[float], target_edge: str):
        """同步方法逻辑 (不涉及模型)"""
        cw, ch = state.current_image.size
        l, t, r, b = map_coords_to_orig(bbox_raw, cw, ch)
        if target_edge:
            nl, nt, nr, nb = 0, 0, cw, ch
            if target_edge == "top":
                nt = b
            elif target_edge == "bottom":
                nb = t
            elif target_edge == "left":
                nl = r
            elif target_edge == "right":
                nr = l

            if (nr - nl) > (cw * self.config.min_crop_ratio) or (nb - nt) > (ch * self.config.min_crop_ratio):
                state.last_pruned_edge = target_edge
                nl, nt, nr, nb = max(0, int(nl)), max(0, int(nt)), min(cw, int(nr)), min(ch, int(nb))
                if nr <= nl or nb <= nt: raise ValueError(f"Invalid crop box in prune: {nl}, {nt}, {nr}, {nb}")
                state.current_image = state.current_image.crop((nl, nt, nr, nb))
                state.add_history("prune_crop", [nl, nt, nr, nb], f"Edge {target_edge} removed.",
                                  local_bbox=[nl, nt, nr, nb])
                ox, oy = state.current_offset
                state.current_offset = (ox + nl, oy + nt)
                return True
        return False

    async def ground_target_instances(self, state: RequestState, target: str):
        """异步版 Grounding 逻辑"""
        cw, ch = state.current_image.size
        orig_w, orig_h = state.original_image.size
        ox, oy = state.current_offset
        verified_instances = []

        if not hasattr(state, 'current_nav_points'): state.current_nav_points = {}

        # 1. Focus
        p = PROMPTS["focus"].format(target=target, question=state.question)
        res = await self._generate_from_vllm(p, state.current_image)
        bbox_dict = extract_labeled_bboxes(res)
        if bbox_dict:
            for label, coords in bbox_dict.items():
                if len(coords) != 4: continue
                l, t, r, b = map_coords_to_orig(coords, cw, ch)
                gl, gt, gr, gb = l + ox, t + oy, r + ox, b + oy

                if self.config.verification_action_2:
                    d = self.config.dilate_action_2
                    vl, vt, vr, vb = max(0, gl - d), max(0, gt - d), min(orig_w, gr + d), min(orig_h, gb + d)
                    vl, vt, vr, vb = int(vl), int(vt), int(vr), int(vb)
                    if vr <= vl or vb <= vt: raise ValueError("Invalid focus verify crop")
                    crop_v = state.original_image.crop((vl, vt, vr, vb))
                    v_res = await self._generate_from_vllm(PROMPTS["verify"].format(target=target), crop_v)

                    if "yes" in v_res.lower():
                        verified_instances.append([gl, gt, gr, gb])
                        if target not in state.found_patches: state.found_patches[target] = []
                        state.found_patches[target].append(crop_v)
                else:
                    verified_instances.append([gl, gt, gr, gb])

            if verified_instances:
                state.found_targets[target] = verified_instances
                state.add_history("ground_success", verified_instances, model_output=f"Det: {res}")
                return True

        # 2. Nav
        p_nav = PROMPTS["navigation"].format(targets=target)
        res_nav = await self._generate_from_vllm(p_nav, state.current_image)
        nav_dict = extract_labeled_bboxes(res_nav)

        rh, rw = smart_resize(ch, cw)
        nav_point_local = None
        for _, coords in nav_dict.items():
            if len(coords) >= 4:
                nx, ny = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2
            elif len(coords) >= 2:
                nx, ny = coords[0], coords[1]
            else:
                continue
            nav_point_local = [nx * (cw / rw), ny * (ch / rh)]
            break

        if nav_point_local:
            state.current_nav_points[target] = nav_point_local
            gl, gt = nav_point_local[0] + ox, nav_point_local[1] + oy
            d = 128
            vl, vt, vr, vb = max(0, gl - d), max(0, gt - d), min(orig_w, gl + d), min(orig_h, gt + d)
            vl, vt, vr, vb = int(vl), int(vt), int(vr), int(vb)
            if vr <= vl or vb <= vt: raise ValueError("Invalid nav verify crop")
            crop_v_nav = state.original_image.crop((vl, vt, vr, vb))
            v_res_nav = await self._generate_from_vllm(PROMPTS["verify"].format(target=target), crop_v_nav)

            if "yes" in v_res_nav.lower():
                verified_instances.append([vl, vt, vr, vb])
                state.found_targets[target] = verified_instances
                state.add_history("ground_success", verified_instances, f"Verified via NavPoint.",
                                  model_output=f"Nav: {res_nav}")
                return True
        return False

    async def reasoning_step(self, state: RequestState) -> bool:
        """异步版单步推理"""
        if self.is_end_loop(state): return True
        if not hasattr(state, 'current_nav_points'): state.current_nav_points = {}
        state.current_nav_points.clear()

        rem = [t for t in state.targets if t not in state.found_targets.keys()]
        for t in rem:
            await self.ground_target_instances(state, t)

        if len(state.found_targets) == len(state.targets) and len(state.targets) > 0:
            return True

        if not await self.remove_unnecessary(state, mode="efficient"):
            return True
        return self.is_end_loop(state)

    async def generate(self, image_pil: Image.Image, question: str, custom_debug_dir: str = None) -> str:
        """异步生成接口"""
        out_dir = custom_debug_dir if custom_debug_dir else self.config.debug_dir
        state = RequestState(image_pil, question, output_dir=out_dir, debug=self.config.debug)
        await self.find_targets(state)

        if not state.targets:
            return await self._generate_from_vllm(question, image_pil)

        try:
            while not await self.reasoning_step(state):
                pass
        except Exception as e:
            print(f"[EchoAsync] Exception during reasoning (e.g., crop error): {e}. Falling back to baseline.")
            return await self._generate_from_vllm(question, image_pil)

        if not state.found_targets:
            return await self._generate_from_vllm(question, image_pil)

        # Focus Crop
        orig_w, orig_h = state.original_image.size
        min_gl, min_gt, max_gr, max_gb = orig_w, orig_h, 0, 0
        all_boxes = []
        for target, coords_list in state.found_targets.items():
            for box in coords_list:
                min_gl, min_gt = min(min_gl, box[0]), min(min_gt, box[1])
                max_gr, max_gb = max(max_gr, box[2]), max(max_gb, box[3])
                all_boxes.append((target, box))

        margin = 150
        fl, ft = max(0, min_gl - margin), max(0, min_gt - margin)
        fr, fb = min(orig_w, max_gr + margin), min(orig_h, max_gb + margin)

        try:
            fl, ft, fr, fb = int(fl), int(ft), int(fr), int(fb)
            if fr <= fl or fb <= ft: raise ValueError("Invalid focus box")
            focus_img = state.original_image.crop((fl, ft, fr, fb)).copy()
        except Exception as e:
            print(f"[EchoAsync] Exception during focus crop ({e}). Falling back to baseline.")
            return await self._generate_from_vllm(question, image_pil)

        draw = ImageDraw.Draw(focus_img)
        if self.config.visual_prompt:
            colors = ["red", "blue", "green", "magenta", "orange", "cyan"]
            target_color_map = {t: colors[i % len(colors)] for i, t in enumerate(state.found_targets.keys())}
    
            from PIL import ImageFont
            focus_w, focus_h = focus_img.size
            font_size = max(24, int(max(focus_w, focus_h) * 0.02))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
    
            line_width = max(3, int(max(focus_w, focus_h) * 0.005))
    
            for target, box in all_boxes:
                color = target_color_map[target]
                rel_box = [box[0] - fl - 10, box[1] - ft - 10, box[2] - fl + 10, box[3] - ft + 10]
                draw.rectangle(rel_box, outline=color, width=line_width)
                draw.text((rel_box[0], rel_box[1] - 25), target, fill=color, font=font)

        if state.debug:
            focus_img.save(os.path.join(state.output_dir, "final_focus_crop.jpg"))

        final_prompt = (
            f"Task: {question}\n\n"
            "I have localized the relevant objects for the task. Below is a high-resolution focused image showing the targets marked with bounding boxes. "
            f"\nPlease analyze the focused visual information and provide the final answer to the question: {question}"
        )

        res = await self._generate_from_vllm(final_prompt, focus_img)
        return res