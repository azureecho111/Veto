from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import os
import json
import re
from base import EchoConfig, EchoForQwen, RequestState, PROMPTS, calculate_min_distance_point_to_bbox, extract_targets
from utils import encode_image, extract_labeled_bboxes, map_coords_to_orig_qwen3, split_targets_sentence, smart_resize
import concurrent.futures

# --- InternVL3 Specific PROMPTS ---
INTERNVL3_PROMPTS = {
    "extract_targets": PROMPTS["extract_targets"],
    "navigation": "For each of the target objects: <ref>{targets}</ref>, where is its most likely location in the image? Please provide the normalized bounding box coordinates (0-1000). Ensure you provide a separate point for each target mentioned. Output in the format: 'label: [x, y, x, y]'. If the target is not present, infer a plausible location.",
    "squeeze_efficient": """
Which edge area is the least likely to contain {targets}? For each edge (top, bottom, left, and right), identify the largest contiguous rectangular region along that edge.
Output each edge's region precisely using normalized coordinates (0-1000) as:
<ref>top</ref>: [x1, y1, x2, y2];
<ref>bottom</ref>: [x1, y1, x2, y2];
<ref>left</ref>: [x1, y1, x2, y2];
<ref>right</ref>: [x1, y1, x2, y2];
""",
    "focus": "Please provide the bounding box coordinate of the region this sentence describes: <ref>{target}</ref>. Return the normalized coordinates (0-1000) for each instance as: [x1, y1, x2, y2]. If there are multiple instances, output all of them.",
    "verify": "In this image, are '{target}' clearly identifiable? If there is a '{target}', output 'EXIST', otherwise output 'no'.",
    "final_answer": PROMPTS["final_answer"]
}

class EchoForInternVL3(EchoForQwen):
    """
    Echo Framework specifically adapted for InternVL3.
    Inherits from EchoForQwen but overrides coordinate handling to use normalized (0-1000).
    """

    def __init__(self, config: EchoConfig = None):
        super().__init__(config)

    def remove_unnecessary(self, state: RequestState, mode: str = "efficient"):
        """Action 1: Squeeze redundancy."""
        all_targets = ", ".join([i for i in state.targets if i not in state.found_targets.keys()])
        cw, ch = state.current_image.size

        if mode == "efficient":
            nav_points = []
            all_targets_list = [i for i in state.targets if i not in state.found_targets.keys()]
            if hasattr(state, 'current_nav_points'):
                for t in all_targets_list:
                    if t in state.current_nav_points:
                        nav_points.append(state.current_nav_points[t])
                
            p_squeeze = INTERNVL3_PROMPTS["squeeze_efficient"].format(question=state.question, targets=all_targets)
            res_sq = self._generate_from_vllm(p_squeeze, state.current_image)
            bbox_dict = extract_labeled_bboxes(res_sq)
            print(f"############ {res_sq} || {bbox_dict}")
            best_bbox, max_score, best_label = None, -1, None
            candidate_status = {}

            for label in ["top", "bottom", "left", "right"]:
                if label not in bbox_dict: continue
                coords = bbox_dict[label]
                if len(coords) != 4: 
                    candidate_status[label] = ("invalid", coords)
                    continue
                
                l_orig, t_orig, r_orig, b_orig = map_coords_to_orig_qwen3(coords, cw, ch)
                box_pixel = [l_orig, t_orig, r_orig, b_orig]

                if label == state.last_pruned_edge:
                    candidate_status[label] = ("skipped_pingpong", box_pixel)
                    continue

                new_w, new_h = cw, ch
                if label == "top": new_h = ch - b_orig
                elif label == "bottom": new_h = t_orig
                elif label == "left": new_w = cw - r_orig
                elif label == "right": new_w = l_orig

                if new_w <= 0 or new_h <= 0 or max(new_w / new_h, new_h / new_w) > self.config.max_aspect_ratio:
                    candidate_status[label] = ("skipped_dimension", box_pixel)
                    continue

                min_dist = float('inf')
                if nav_points:
                    for np in nav_points:
                        d = calculate_min_distance_point_to_bbox(np, box_pixel)
                        min_dist = min(min_dist, d)
                else:
                    min_dist = 0
                
                if nav_points and min_dist <= 0:
                    candidate_status[label] = ("skipped_contains_nav", box_pixel)
                    continue

                if min_dist > max_score:
                    max_score, best_bbox, best_label = min_dist, coords, label
                
                candidate_status[label] = ("valid", box_pixel)

            if state.debug:
                viz_img = state.current_image.copy()
                draw = ImageDraw.Draw(viz_img)
                for label, (status, box) in candidate_status.items():
                    if status == "invalid" or box[0] >= box[2] or box[1] >= box[3]: continue
                    if label == best_label: color, width, txt = "red", 10, f"PICKED: {label}"
                    elif status == "valid": color, width, txt = "blue", 5, label
                    else: color, width, txt = "gray", 3, f"{label} ({status})"
                    draw.rectangle(box, outline=color, width=width)
                    draw.text((box[0] + 5, box[1] + 5), txt, fill=color)
                for i, np in enumerate(nav_points):
                    r = 12
                    draw.ellipse([np[0]-r, np[1]-r, np[0]+r, np[1]+r], fill="lime", outline="white", width=3)
                path = os.path.join(state.output_dir, f"step_{state.step_count}_squeeze_analyze_visualize.jpg")
                viz_img.save(path)

            if best_bbox:
                state.add_history("squeeze_analyze", bbox_dict, f"Pick: {best_label}", model_output=f"Squeeze: {res_sq}")
                return self._apply_pruning(state, best_bbox, target_edge=best_label)
            return False
        return False

    def _apply_pruning(self, state: RequestState, bbox_raw: List[float], target_edge: str = None):
        """Action 1 Pruning logic."""
        cw, ch = state.current_image.size
        l, t, r, b = map_coords_to_orig_qwen3(bbox_raw, cw, ch)

        if target_edge:
            nl, nt, nr, nb = 0, 0, cw, ch
            if target_edge == "top": nt = b
            elif target_edge == "bottom": nb = t
            elif target_edge == "left": nl = r
            elif target_edge == "right": nr = l

            if (nr - nl) > (cw * self.config.min_crop_ratio) or (nb - nt) > (ch * self.config.min_crop_ratio):
                state.last_pruned_edge = target_edge
                nl, nt, nr, nb = max(0, int(nl)), max(0, int(nt)), min(cw, int(nr)), min(ch, int(nb))
                if nr <= nl or nb <= nt: return False
                state.current_image = state.current_image.crop((nl, nt, nr, nb))
                state.add_history("prune_crop", [nl, nt, nr, nb], f"Edge {target_edge} removed.", local_bbox=[nl, nt, nr, nb])
                ox, oy = state.current_offset
                state.current_offset = (ox + nl, oy + nt)
                return True
        return False

    def ground_target_instances(self, state: RequestState, target: str):
        """Action 2 Localization."""
        cw, ch = state.current_image.size
        orig_w, orig_h = state.original_image.size
        ox, oy = state.current_offset
        verified_instances = []

        if not hasattr(state, 'current_nav_points'): state.current_nav_points = {}

        # 1. Focus Phase
        p = INTERNVL3_PROMPTS["focus"].format(target=target, question=state.question)
        res = self._generate_from_vllm(p, state.current_image)
        bbox_dict = extract_labeled_bboxes(res)
        
        if bbox_dict:
            for label, coords in bbox_dict.items():
                if len(coords) != 4: continue
                l, t, r, b = map_coords_to_orig_qwen3(coords, cw, ch)
                gl, gt, gr, gb = l + ox, t + oy, r + ox, b + oy
                
                if self.config.verification_action_2:
                    d = self.config.dilate_action_2
                    vl, vt, vr, vb = max(0, gl-d), max(0, gt-d), min(orig_w, gr+d), min(orig_h, gb+d)
                    crop_v = state.original_image.crop((int(vl), int(vt), int(vr), int(vb)))
                    v_res = self._generate_from_vllm(INTERNVL3_PROMPTS["verify"].format(target=target), crop_v)
                    print(f"@@@@@@@ VERIFIED: {v_res}")
                    if state.debug:
                        draw = ImageDraw.Draw(crop_v)
                        draw.rectangle([gl-vl-15, gt-vt-15, gr-vl+15, gb-vt+15], outline="red", width=3)
                        with state.lock:
                            crop_v.save(os.path.join(state.output_dir, f"step_{state.step_count}_verification_focus_{target}.jpg"))
                    
                    if "yes" in v_res.lower():
                        with state.lock:
                            verified_instances.append([gl, gt, gr, gb])
                            if target not in state.found_patches: state.found_patches[target] = []
                            state.found_patches[target].append(crop_v)
                else:
                    verified_instances.append([gl, gt, gr, gb])

            if verified_instances:
                with state.lock:
                    state.found_targets[target] = verified_instances
                state.add_history("ground_success", verified_instances, model_output=f"Det: {res}")
                return True

        # 2. Navigation Phase
        p_nav = INTERNVL3_PROMPTS["navigation"].format(targets=target)
        res_nav = self._generate_from_vllm(p_nav, state.current_image)
        print(f"####### NAV: {res_nav}")
        nav_dict = extract_labeled_bboxes(res_nav)
        
        if nav_dict:
            for _, coords in nav_dict.items():
                # Correct Order: [x, y]
                x, y = (coords[0], coords[1]) if len(coords) == 2 else ((coords[0]+coords[2])/2, (coords[1]+coords[3])/2)
                l, t, r, b = map_coords_to_orig_qwen3([x, y, x, y], cw, ch)
                nav_point_local = [l, t]
                with state.lock:
                    state.current_nav_points[target] = nav_point_local
                
                gl, gt = l + ox, t + oy
                d = 128
                vl, vt, vr, vb = max(0, gl-d), max(0, gt-d), min(orig_w, gl+d), min(orig_h, gt+d)
                crop_v_nav = state.original_image.crop((int(vl), int(vt), int(vr), int(vb)))
                v_res_nav = self._generate_from_vllm(INTERNVL3_PROMPTS["verify"].format(target=target), crop_v_nav)
                
                if state.debug:
                    with state.lock:
                        crop_v_nav.save(os.path.join(state.output_dir, f"step_{state.step_count}_verification_nav_{target}.jpg"))

                if "yes" in v_res_nav.lower():
                    with state.lock:
                        verified_instances.append([vl, vt, vr, vb])
                        state.found_targets[target] = verified_instances
                    state.add_history("ground_success", verified_instances, f"Via NavPoint", model_output=f"Nav: {res_nav}")
                    return True
                break

        state.add_history("ground_no_result", None, f"No verified {target}", model_output=f"Focus: {res}")
        return False

    def reasoning_step(self, state: RequestState) -> bool:
        """Standard reasoning step."""
        if self.is_end_loop(state): return True
        if not hasattr(state, 'current_nav_points'): state.current_nav_points = {}
        state.current_nav_points.clear()

        rem = [t for t in state.targets if t not in state.found_targets.keys()]
        if rem:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(rem)) as executor:
                executor.map(lambda t: self.ground_target_instances(state, t), rem)

        if len(state.found_targets) == len(state.targets) and len(state.targets) > 0:
            return True

        if not self.remove_unnecessary(state, mode="efficient"):
            return True
        return self.is_end_loop(state)

    def generate(self, image_pil: Image.Image, question: str, custom_debug_dir: str = None) -> str:
        """Final generation step."""
        out_dir = custom_debug_dir if custom_debug_dir else self.config.debug_dir
        state = RequestState(image_pil, question, output_dir=out_dir, debug=self.config.debug)
        self.find_targets(state)

        # 安全检查：如果没提取出 target，直接用原图回答
        if not state.targets:
            return self._generate_from_vllm(question, image_pil)
            
        try:
            while not self.reasoning_step(state): pass
        except Exception as e:
            return self._generate_from_vllm(question, image_pil)

        if not state.found_targets or any(t not in state.found_targets.keys() for t in state.targets):
            return self._generate_from_vllm(question, image_pil)

        if not state.found_targets: return self._generate_from_vllm(question, image_pil)

        # Build Focus Crop
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
        focus_img = state.original_image.crop((int(fl), int(ft), int(fr), int(fb))).copy()
        
        if self.config.visual_prompt:
            draw = ImageDraw.Draw(focus_img)
            colors = ["red", "blue", "green", "magenta", "orange", "cyan"]
            target_color_map = {t: colors[i % len(colors)] for i, t in enumerate(state.found_targets.keys())}
            focus_w, focus_h = focus_img.size
            font_size = max(24, int(max(focus_w, focus_h) * 0.02))
            try: font = ImageFont.truetype("arial.ttf", font_size)
            except: font = ImageFont.load_default(); font_size = 15
            line_width = max(3, int(max(focus_w, focus_h) * 0.005))

            for target, box in all_boxes:
                color = target_color_map[target]
                obj_w, obj_h = box[2] - box[0], box[3] - box[1]
                dx = min(int(focus_w*0.05), max(5, int(obj_w*0.1)))
                dy = min(int(focus_h*0.05), max(5, int(obj_h*0.1)))
                rel_box = [box[0]-fl-dx, box[1]-ft-dy, box[2]-fl+dx, box[3]-ft+dy]
                draw.rectangle(rel_box, outline=color, width=line_width)
                try:
                    txt_bbox = draw.textbbox((rel_box[0], rel_box[1]-font_size-5), target, font=font)
                    draw.rectangle([txt_bbox[0]-2, txt_bbox[1]-2, txt_bbox[2]+2, txt_bbox[3]+2], fill="white")
                except: pass
                draw.text((rel_box[0], rel_box[1]-font_size-5), target, fill=color, font=font)

        if state.debug: focus_img.save(os.path.join(state.output_dir, "final_focus_crop.jpg"))
        
        final_prompt = INTERNVL3_PROMPTS["final_answer"].format(question=question)
        final_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(focus_img)}"}},
            {"type": "text", "text": f"Task: {question}\n\nI have localized the relevant objects. Here is a high-resolution focused image showing the targets marked with bounding boxes. \n{final_prompt}"}
        ]
        res = self.client.chat.completions.create(model=self.config.model_name, messages=[{"role": "user", "content": final_content}], temperature=self.config.temperature)
        return res.choices[0].message.content
