from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw
import os
import json
import re
from openai import OpenAI
from utils import encode_image, extract_bbox_from_text, extract_labeled_bboxes, map_coords_to_orig, \
    split_targets_sentence, smart_resize

import math
import threading
import concurrent.futures


def calculate_min_distance_point_to_bbox(point: List[float], bbox: List[float]) -> float:
    """计算点 (px, py) 到矩形 [l, t, r, b] 的最短欧氏距离。
    如果点在矩形内，距离为 0。"""
    px, py = point
    l, t, r, b = bbox

    # 最近点的坐标
    closest_x = max(l, min(px, r))
    closest_y = max(t, min(py, b))

    # 距离
    dx = px - closest_x
    dy = py - closest_y
    return math.sqrt(dx * dx + dy * dy)

def extract_targets(sentence: str, pattern=r": (.+)"):
    match = re.search(pattern, sentence)
    if match:
        return match.group(1)
    return None


class EchoConfig:
    """Echo推理框架的超参数配置类"""

    def __init__(self):
        # API 配置
        self.api_url = "http://localhost:18903/v1"
        self.api_key = "EMPTY"
        self.model_name = "qwen2.5-vl-7b"

        # 推理逻辑超参数
        self.max_steps = 6  # 最大推理步数，防止死循环
        self.min_image_size = 128  # 图片裁切的最细颗粒度（像素）
        self.edge_threshold = 0.02  # 判定Bbox是否在边缘的比例阈值
        self.mask_influence_strength = 0.5
        # TODO
        self.min_crop_ratio = 0.3  # 只有当新裁切出来的图片宽度/高度变化超过该比例时，才执行物理裁切
        self.max_aspect_ratio = 20.0  # 防止图片裁成极细长条导致模型失效

        self.verification_action_2 = True
        self.dilate_action_2 = 128
        self.visual_prompt = True

        self.debug = True  # 是否保存中间过程图片和日志
        self.debug_dir = "debug_echo"  # 保存的基础目录

        # In-Context Learning 配置
        self.ic_path = os.path.join(os.getcwd(), "ic_examples", "vstar.json")

        self.temperature = 0.4
        self.seed = 42


# --- PROMPT MANAGEMENT ---
PROMPTS = {
    "extract_targets": "What are the specific objects mentioned in the question: '{question}' that are needed to determine the answer? List them as singular nouns, separated by commas.",
    # 注意下面的label可能要改一下啊
    "navigation": "For each of the target objects: {targets}, where is its most likely location in the image? Provide the absolute pixel coordinates for each as: 'label: [x, y]'. Ensure you provide a separate point for each target mentioned. If the target object is not present in the picture, please infer a plausible location based on the image content.",

    "squeeze_efficient": """
If objects like {targets} are present near the four edges of the image, which edge area is the least likely to contain such objects? For each edge (top, bottom, left, and right), identify the largest contiguous rectangular region along that edge that meets **all** of the following criteria:

1.  **Target Exclusion:** The area can be confidently judged as **not containing** the specified target(s): {targets}. From common-sense, the target will never logically appear in this area (For example, a horse-drawn carriage will never be in the sky).
2.  **Question Irrelevance:** The content of the area is irrelevant to answering the question: {question}. You should check the area carefully and infer whether this area will influence the final answer.

**For each edge, provide a detailed justification** explaining why the proposed rectangle meets these criteria. If a suitable, sufficiently large rectangular region cannot be confidently defined for an edge, skip that edge in the final coordinate list. 

Please structure your final output precisely as follows:
'Analysis: [Provide a concise summary, then a detailed justification for **each** edge you processed, one by one.]
top: [x1,y1,x2,y2];
bottom: [x1,y1,x2,y2];
left: [x1,y1,x2,y2];
right: [x1,y1,x2,y2];'

(Replace `[x1,y1,x2,y2]` with the actual coordinates, keep the bracket.)
    """,
    "squeeze_parallel": "Check the {edge} of the image. Is there a redundant area here that doesn't contain {targets}? If yes, return coordinates [x1, y1, x2, y2]. If no, say 'None'.",

    "focus": "Locate ALL physical instances of '{target}' mentioned in or relevant to the question: '{question}' in this image. Return the absolute pixel coordinates for each instance using this format: {target}_1: [x,y,x,y]; {target}_2: [x,y,x,y]. If only one instance exists, return {target}: [x,y,x,y].",
    "verify": "In this image, are '{target}' clearly identifiable? Answer 'Yes' or 'No'.",
    # TODO:注意final answer的使用
    "final_answer": "Task: Answer the question: '{question}' based on the provided images.",

}


class RequestState:
    """每一个样本的管理类，管理从输入样本，到返回给用户最终输出的过程中，本Framework每一步的决策，模型的输出以及（图片、文本）的中间结果。
    除此以外，还应该包括：有哪些target objects，哪些还没有被找到，找到的object对应的各自的image region，以及*该region*在整张图片上的位置。
    TODO：多个物体的推理逻辑没想好，以及对于未知数量的情况也没有分析。
    TODO: 未知数量物体，一个想法是：让模型直接grounding所有他能看见的target.
    """

    def __init__(self, image_pil: Image.Image, question: str, output_dir: str = "debug_echo", debug: bool = True):
        self.original_image = image_pil
        self.current_image = image_pil.copy()
        self.current_offset = (0, 0)  # (left, top) relative to original image
        self.question = question
        self.targets: List[str] = []
        self.found_targets: Dict[str, List[List[int]]] = {}
        self.found_patches: Dict[str, List[Image.Image]] = {}  # 存储验证成功的局部高分辨率贴图
        self.history: List[Dict[str, Any]] = []
        self.step_count = 0
        self.debug = debug
        self.output_dir = output_dir
        self.last_pruned_edge: Optional[str] = None  # 记录上一次裁切的是哪条边
        self.lock = threading.Lock()
        if self.debug and not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

    def add_history(self, action: str, result: Any, reason: str = "", local_bbox: Optional[List[int]] = None,
                    move_step=True, model_output: str = ""):
        global_bbox = None
        if local_bbox:
            ox, oy = self.current_offset
            global_bbox = [local_bbox[0] + ox, local_bbox[1] + oy, local_bbox[2] + ox, local_bbox[3] + oy]

        with self.lock:
            self.history.append({
                "step": self.step_count,
                "action": action,
                "result": result,
                "reason": reason,
                "model_output": model_output,
                "local_bbox": local_bbox,
                "global_bbox": global_bbox,
                "current_offset": self.current_offset,
                "current_targets": self.targets,
                "found_targets": list(self.found_targets.keys())
            })
            if self.debug:
                # 保存图片
                path = os.path.join(self.output_dir, f"step_{self.step_count}_{action}.jpg")
                self.current_image.save(path)

                # 保存JSON轨迹以便分析
                log_path = os.path.join(self.output_dir, "trajectory.json")
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(self.history, f, indent=4, ensure_ascii=False)
            if move_step:
                self.step_count += 1


class EchoForQwen:
    """为Qwen2.5-VL设计的框架，即这个类封装了Qwen2.5-VL模型，并且可以实现我们idea的推理逻辑。
    目前初步我们实现的版本，内部并没有真正的Qwen2.5-VL模型的成员变量，而是我们假定已经通过vLLM serve部署好了模型，可以直接request访问。"""

    def __init__(self, config: EchoConfig = None):
        self.config = config if config else EchoConfig()
        self.client = OpenAI(api_key=self.config.api_key, base_url=self.config.api_url)

    def _generate_from_vllm(self, prompt: str, image: Optional[Image.Image] = None) -> str:
        """每一次request to vLLM部署的模型就调用这个接口"""
        content = []
        if image:
            b64 = encode_image(image)
            content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        res = self.client.chat.completions.create(model=self.config.model_name,
                                                  messages=[{"role": "user", "content": content}],
                                                  temperature=0.0, seed=42)
        return res.choices[0].message.content

    def find_targets(self, state: RequestState):
        """利用大模型获取一个问题中的目标物体（使用 In-Context Learning 提高准确度）"""
        if not os.path.exists(self.config.ic_path):
            print(f"[Echo] IC Examples not found at {self.config.ic_path}, falling back to simple extraction.")
            p = PROMPTS["extract_targets"].format(question=state.question)
            res = self._generate_from_vllm(p)
            state.targets = [t.strip().lower() for t in res.split(",")]
        else:
            with open(self.config.ic_path, 'r', encoding='utf-8') as f:
                ic_examples = json.load(f)

            ic_question_template = ic_examples["question_template"]
            ic_question_list = ic_examples["question_list"]
            ic_response_list = ic_examples["response_list"]

            messages = []
            content = [
                {
                    "type": "text",
                    "text": f"""
Suppose you have an image and will use it to answer the corresponding question. Please analyze what target objects in the image you need to closely examine in order to answer this question.
Rules:
    1. The object must be unambiguous; do not expand or narrow the search scope. For example, for "a woman with a hat," do not simply output "woman," as that would broaden the scope.
    2. If the object includes a quantifier, output the object directly.
    3. First conduct an analysis, then summarize the answer at the end.
Example:

{"\n".join([f"[Question]: {question}\n[Response]: {response}" for question, response in zip(ic_question_list, ic_response_list)])}
...
Now, please answer:
[Question]: {state.question[:state.question.find("Please answer")]}
[Response]: """
                }
            ]
            messages = [
                {"role": "user", "content": content}
            ]
            res = self.client.chat.completions.create(model=self.config.model_name, messages=messages, temperature=self.config.temperature, seed=self.config.seed)
            response_text = res.choices[0].message.content

            # Remove '*' in response text. This is NECESSARY for Qwen2.5-VL-32B, as it's used to print *.
            response_text = response_text.replace('*', '')

            targets_sentence = extract_targets(response_text)
            print(f"### [Echo] IC Response:{response_text} || Extracted Item: {targets_sentence}")

            if targets_sentence:
                state.targets = [t.strip().lower() for t in split_targets_sentence(targets_sentence)]
            else:
                state.targets = []
            # state.targets = []
            state.add_history("extract_targets", state.targets, f"IC Learning from {self.config.ic_path}",
                              model_output=response_text)

        print(f"[Echo] Targets detected: {state.targets} (Question: {state.question})")

    def is_end_loop(self, state: RequestState) -> bool:
        """判断当前状态是否应该结束推理链，即满足以下条件之一：
            1. 图片的至少有一条边长度过于小了
            2. 所有目标的位置都已明确，模型已经能回答问题，或者模型已经不再需要进一步裁剪图片
            3. ...暂时没想好"""
        w, h = state.current_image.size
        # 1. 满足最小尺寸阈值
        if w < self.config.min_image_size or h < self.config.min_image_size:
            return True
        # 2. 满足纵横比保护
        if max(w / h, h / w) > self.config.max_aspect_ratio:
            return True
        # 3. 满足最大步数限制
        if state.step_count >= self.config.max_steps:
            return True
        # 4. 满足目标找全逻辑
        if len(state.found_targets) == len(state.targets) and len(state.targets) > 0:
            return True
        return False

    def remove_unnecessary(self, state: RequestState, mode: str = "efficient"):
        """Action 1: 在每一个迭代步中，辨别并删除当前图片中的冗余部分，返回删除冗余的图片。
        目前先实现最简单的策略：要求模型从图片四条边出发扩展四个区域，这个区域不能包含任何的target objects，要求模型输出坐标。
        实现两种模式：efficient：一次request询问模型对四条边的分析；parallel：parallel：四次request分别问指定边引出的区域是否存在target objects。"""
        # 重要：使用全量的 state.targets 而不仅是剩余目标。
        # 因为我们希望 Squeeze 动作在排除背景时，不要把已经找到的目标也剪掉（保证上下文完整性）
        all_targets = ", ".join([i for i in state.targets if i not in state.found_targets.keys()])
        cw, ch = state.current_image.size

        if mode == "efficient":
            # 1. 语义导航：直接从 state 中获取 Action 2 算好的 Navigation Points
            nav_points = []
            all_targets_list = [i for i in state.targets if i not in state.found_targets.keys()]
            if hasattr(state, 'current_nav_points'):
                for t in all_targets_list:
                    if t in state.current_nav_points:
                        nav_points.append(state.current_nav_points[t])
                
            if state.debug:
                print(f"[Echo] {len(nav_points)} Navigation Points retrieved: {nav_points}")

            # 2. 候选区生成：获取四个边缘候选矩形
            p_squeeze = PROMPTS["squeeze_efficient"].format(question=state.question[:state.question.find("Please answer")], targets=all_targets)
            res_sq = self._generate_from_vllm(p_squeeze, state.current_image)
            bbox_dict = extract_labeled_bboxes(res_sq)
            
            print(bbox_dict)
            # 3. 启发式筛选：寻找离 Navigation Point 最远的方向
            best_bbox, max_score, best_label = None, -1, None
            candidate_status = {} # 用于绘图区分：picked, valid, skipped

            for label, coords in bbox_dict.items():
                if len(coords) != 4: 
                    candidate_status[label] = ("invalid", coords)
                    continue
                
                l_orig, t_orig, r_orig, b_orig = map_coords_to_orig(coords, cw, ch)
                box_pixel = [l_orig, t_orig, r_orig, b_orig]

                # 检查是否由于防抖被跳过
                if label == state.last_pruned_edge:
                    candidate_status[label] = ("skipped_pingpong", box_pixel)
                    continue

                # 纵横比保护
                new_w, new_h = cw, ch
                if label == "top": new_h = ch - b_orig
                elif label == "bottom": new_h = t_orig
                elif label == "left": new_w = cw - r_orig
                elif label == "right": new_w = l_orig

                if new_w <= 0 or new_h <= 0:
                    candidate_status[label] = ("skipped_dimension", box_pixel)
                    continue
                
                if max(new_w / new_h, new_h / new_w) > self.config.max_aspect_ratio:
                    candidate_status[label] = ("skipped_aspect_ratio", box_pixel)
                    continue

                # 距离计算：计算该矩形区域到所有 nav_points 的最小距离
                min_dist_to_any_target = float('inf')
                if nav_points:
                    for np in nav_points:
                        d = calculate_min_distance_point_to_bbox(np, box_pixel)
                        min_dist_to_any_target = min(min_dist_to_any_target, d)
                else:
                    # 如果没有预测出点，默认使用一个极大的数（让它退化为面积竞争，或者在后面加别的保底）
                    min_dist_to_any_target = 0
                
                # 核心逻辑：如果预测中心之一落在了这个区域内，绝对不能删
                if nav_points and min_dist_to_any_target <= 0:
                    candidate_status[label] = ("skipped_contains_nav", box_pixel)
                    continue

                score = min_dist_to_any_target

                if score > max_score:
                    max_score = score
                    best_bbox = coords
                    best_label = label
                
                candidate_status[label] = ("valid", box_pixel)

            # --- DEBUG 可视化 ---
            if state.debug:
                viz_img = state.current_image.copy()
                draw = ImageDraw.Draw(viz_img)
                
                # 1. 所有框分类着色绘图
                for label, (status, box) in candidate_status.items():
                    if box[0] >= box[2] or box[1] >= box[3]:
                        continue
                    if status == "invalid": continue # 无法映射的点不画或者是长度不对
                    
                    if label == best_label:
                        color, width = "red", 10
                        label_text = f"PICKED: {label}"
                    elif status == "valid":
                        color, width = "blue", 5
                        label_text = label
                    else:
                        color, width = "gray", 3
                        label_text = f"{label} ({status})"
                    
                    draw.rectangle(box, outline=color, width=width)
                    draw.text((box[0] + 5, box[1] + 5), label_text, fill=color)

                # 2. 画出所有导航点 (盖在框上面)
                for i, np in enumerate(nav_points):
                    r = 12
                    draw.ellipse([np[0]-r, np[1]-r, np[0]+r, np[1]+r], fill="lime", outline="white", width=3)
                    draw.text((np[0] + 15, np[1]), f"NAV_{i}", fill="lime")

                path = os.path.join(state.output_dir, f"step_{state.step_count}_squeeze_analyze_visualize.jpg")
                viz_img.save(path)
                print(f"[Echo] Debug Viz saved: {path} (Lime: NavPoint, Red: Picked, Blue: Valid, Gray: Skipped)")

            if best_bbox:
                state.add_history("squeeze_analyze", bbox_dict, 
                                  f"Distance-based pick: {best_label}. Multi-NavPoints count: {len(nav_points)}", 
                                  model_output=f"Squeeze: {res_sq}")
                return self._apply_pruning(state, best_bbox, target_edge=best_label, reason=f"Farthest from all predicted targets centers.")
            else:
                return False
        else:  # parallel mode
            for edge in ["top", "bottom", "left", "right"]:
                p = PROMPTS["squeeze_parallel"].format(edge=edge, targets=all_targets)
                res = self._generate_from_vllm(p, state.current_image)
                bbox_raw = extract_bbox_from_text(res)
                if bbox_raw:
                    # Parallel mode aspect ratio check
                    l, t, r, b = map_coords_to_orig(bbox_raw, cw, ch)
                    new_w, new_h = cw, ch
                    if edge == "top":
                        new_h = ch - b
                    elif edge == "bottom":
                        new_h = t
                    elif edge == "left":
                        new_w = cw - r
                    elif edge == "right":
                        new_w = l

                    if new_w > 0 and new_h > 0 and max(new_w / new_h, new_h / new_w) <= self.config.max_aspect_ratio:
                        return self._apply_pruning(state, bbox_raw, target_edge=edge,
                                            reason=f"Parallel analysis on {edge}: {res}")



    def _apply_pruning(self, state: RequestState, bbox_raw: List[float], target_edge: str = None, reason: str = None):
        cw, ch = state.current_image.size
        l, t, r, b = map_coords_to_orig(bbox_raw, cw, ch)

        # 如果没有指定边，通过坐标启发式判定
        if not target_edge:
            eth_v = ch * self.config.edge_threshold
            eth_h = cw * self.config.edge_threshold
            is_at_top, is_at_bottom = t <= eth_v, b >= ch - eth_v
            is_at_left, is_at_right = l <= eth_h, r >= cw - eth_h

            if is_at_top:
                target_edge = "top"
            elif is_at_bottom:
                target_edge = "bottom"
            elif is_at_left:
                target_edge = "left"
            elif is_at_right:
                target_edge = "right"

        if target_edge:
            nl, nt, nr, nb = 0, 0, cw, ch
            edge_label = target_edge
            if target_edge == "top":
                nt = b
            elif target_edge == "bottom":
                nb = t
            elif target_edge == "left":
                nl = r
            elif target_edge == "right":
                nr = l

            # 使用Config定义的物理裁切门槛
            if (nr - nl) > (cw * self.config.min_crop_ratio) or (nb - nt) > (ch * self.config.min_crop_ratio):
                # 更新状态
                state.last_pruned_edge = edge_label

                # --- 1. 执行物理裁切 ---
                nl, nt, nr, nb = max(0, int(nl)), max(0, int(nt)), min(cw, int(nr)), min(ch, int(nb))
                if nr <= nl or nb <= nt: raise ValueError(f"Invalid crop box in prune: {nl}, {nt}, {nr}, {nb}")
                state.current_image = state.current_image.crop((nl, nt, nr, nb))

                # --- 2. 只有裁切完成后再记录历史，这样保存的就是裁切后的结果图了 ---
                state.add_history("prune_crop", [nl, nt, nr, nb], f"Edge {edge_label} redundant removed. {reason}",
                                  local_bbox=[nl, nt, nr, nb], )

                # --- 3. 更新全局偏移量 (必须在 add_history 之后，因为 add_history 内部依赖旧的 offset 换算全图坐标) ---
                ox, oy = state.current_offset
                state.current_offset = (ox + nl, oy + nt)
                return True

        else:
            # Middle redundancy -> Mask
            state.last_pruned_edge = "mask"
            draw = ImageDraw.Draw(state.current_image)
            draw.rectangle([l, t, r, b], fill="black")
            state.add_history("prune_mask", [l, t, r, b], f"Center redundancy masked. {reason}",
                              local_bbox=[l, t, r, b], )
            return True
        return False
    def ground_target_instances(self, state: RequestState, target: str):
        """Action 2 (Observer): 在当前视野中尝试定位目标物体。
        该函数仅负责“观察”并记录坐标，不会改变 current_image。
        它会尝试识别当前图中该 target 的所有实例，并换算为全局坐标保存。
        """
        cw, ch = state.current_image.size
        orig_w, orig_h = state.original_image.size
        ox, oy = state.current_offset
        verified_instances = []

        if not hasattr(state, 'current_nav_points'):
            state.current_nav_points = {}

        # 1. 优先使用原来的 Focus 逻辑进行全局搜索
        p = PROMPTS["focus"].format(target=target, question=state.question[:state.question.find("Please answer")])
        res = self._generate_from_vllm(p, state.current_image)
        # print(f"######## {res} 让爱再继续")
        bbox_dict = extract_labeled_bboxes(res)
        if bbox_dict:

            for label, coords in bbox_dict.items():
                if len(coords) != 4: continue
                # print(f"#### Before : {coords}")
                # coords = (coords[1], coords[0], coords[3], coords[2])
                # print(f"#### After : {coords}")

                l, t, r, b = map_coords_to_orig(coords, cw, ch)
                # print(f"#### After : {l}, {t}, {r}, {b}")
                if l > r:
                    return False
                # 计算全局绝对坐标
                gl, gt, gr, gb = l + ox, t + oy, r + ox, b + oy
                if self.config.verification_action_2:
                    # 按照用户要求：膨胀 -> 裁切原图 -> 验证
                    d = self.config.dilate_action_2
                    vl, vt, vr, vb = max(0, gl - d), max(0, gt - d), min(orig_w, gr + d), min(orig_h, gb + d)
                    vl, vt, vr, vb = int(vl), int(vt), int(vr), int(vb)
                    if vr <= vl or vb <= vt: raise ValueError("Invalid focus verify crop")
                    # # 从原图中裁切出局部区域进行验证
                    # print('####',vl, vt,vr,vb,'####')
                    crop_v = state.original_image.crop((vl, vt, vr, vb))

                    v_res = self._generate_from_vllm(PROMPTS["verify"].format(target=target), crop_v)
                    # print(f"@@@@@@@@@@@@ {v_res}")

                    draw = ImageDraw.Draw(crop_v)
                    # 转换全局 BBox 到局部 crop 图像的相对坐标，并膨胀 2 像素以防遮挡物体边缘
                    rel_bbox = [gl - vl - 15, gt - vt - 15, gr - vl + 15, gb - vt + 15]
                    draw.rectangle(rel_bbox, outline="red", width=3)

                    if self.config.debug:
                        with state.lock:
                            _path = os.path.join(state.output_dir, f"step_{state.step_count}_verification_focus_{target}.jpg")
                            crop_v.save(_path)
                            print(f"[Echo] Verification Step: {v_res}, Corresponding image: {_path}")

                    if "yes" in v_res.lower():
                        with state.lock:
                            verified_instances.append([gl, gt, gr, gb])
                            # 保存成功验证的局部贴图
                            if target not in state.found_patches:
                                state.found_patches[target] = []
                            state.found_patches[target].append(crop_v)
                    else:
                        print(f"[Echo] Instance of {target} failed verification at global {gl, gt, gr, gb}")
                else:
                    verified_instances.append([gl, gt, gr, gb])

            if verified_instances:
                with state.lock:
                    state.found_targets[target] = verified_instances
                state.add_history("ground_success", verified_instances,
                                  f"Verified {len(verified_instances)} instances of {target}",
                                  model_output=f"Detection: {res}" + (
                                      f"\nVerification: {v_res}" if self.config.verification_action_2 else ""))
                return True

        # 2. 如果 Focus 逻辑找不到或验证失败，尝试使用 Nav Point 作为补充
        print(f"[Echo] Target '{target}' not found by focus, trying Nav Point prediction...")
        p_nav = PROMPTS["navigation"].format(targets=target)
        res_nav = self._generate_from_vllm(p_nav, state.current_image)
        nav_points_raw_dict = extract_labeled_bboxes(res_nav)
        
        rh, rw = smart_resize(ch, cw)
        nav_point_local = None
        for l_nav, coords in nav_points_raw_dict.items():
            if len(coords) >= 4:
                nx, ny = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2
            elif len(coords) >= 2:
                nx, ny = coords[0], coords[1]
            else: continue
            nav_point_local = [nx * (cw / rw), ny * (ch / rh)]
            break # 用第一个有效点

        # 无论预测验证是否成功，获取到的预测点都要存下来供 Action 1 Squeeze 使用
        if nav_point_local:
            with state.lock:
                state.current_nav_points[target] = nav_point_local
            
            # 膨胀 128 像素裁剪并验证
            gl = nav_point_local[0] + ox 
            gt = nav_point_local[1] + oy
            d = 128
            vl, vt, vr, vb = max(0, gl - d), max(0, gt - d), min(orig_w, gl + d), min(orig_h, gt + d)
            vl, vt, vr, vb = int(vl), int(vt), int(vr), int(vb)
            if vr <= vl or vb <= vt: raise ValueError("Invalid nav verify crop")
            crop_v_nav = state.original_image.crop((vl, vt, vr, vb))
            v_res_nav = self._generate_from_vllm(PROMPTS["verify"].format(target=target), crop_v_nav)
            
            if state.debug:
                 with state.lock:
                     _path = os.path.join(state.output_dir, f"step_{state.step_count}_verification_nav_{target}.jpg")
                     crop_v_nav.save(_path)

            if "yes" in v_res_nav.lower():
                 # 验证通过，把它当做找到了
                 with state.lock:
                    verified_instances.append([vl, vt, vr, vb])
                    if target not in state.found_patches:
                        state.found_patches[target] = []
                    state.found_patches[target].append(crop_v_nav)
                    state.found_targets[target] = verified_instances
                 print(f"[Echo] Target '{target}' verified successfully at nav point crop.")
                 state.add_history("ground_success", verified_instances,
                                   f"Verified instance of {target} via NavPoint.",
                                   model_output=f"Nav: {res_nav}\nVerification: {v_res_nav}")
                 return True

        state.add_history("ground_no_result", None, f"No verified {target} detected in current view", model_output=f"Focus: {res}")
        return False

    def reasoning_step(self, state: RequestState) -> bool:
        """推理一个样本时每一步的接口。一次样本将会多次调用这个接口，这个接口除了返回每一次的结果，应该还需要返回一个bool代表推理是否完全结束。
        核心逻辑：(Grounding Phase: 观察所有目标) -> (Squeeze Phase: 缩小搜索范围)
        注意：Action 1和 Action 2都是有几率失败的，你需要设置一个保底的逻辑。"""
        if self.is_end_loop(state): return True

        if not hasattr(state, 'current_nav_points'):
            state.current_nav_points = {}
        # 清空本轮 nav point 缓存，确保新的一轮获取最新的相对坐标
        state.current_nav_points.clear()

        # 1. 观察阶段：对所有还没找到或者需要确认的目标进行 Grounding
        rem = [t for t in state.targets if t not in state.found_targets.keys()]
        if rem:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(rem)) as executor:
                executor.map(lambda t: self.ground_target_instances(state, t), rem)

        # 2. 状态检查：如果全部找到了，可以提前结束
        if len(state.found_targets) == len(state.targets) and len(state.targets) > 0:
            return True

        # 3. 变换阶段：如果还有目标没找到，或者图片还太大，执行 Squeeze 减枝
        if not self.remove_unnecessary(state, mode="efficient"):
            return True

        return self.is_end_loop(state)

    def generate(self, image_pil: Image.Image, question: str, custom_debug_dir: str = None) -> str:
        """供外部调用，每一次传入一个样本的数据，然后返回这个样本输出的最终结果。"""
        out_dir = custom_debug_dir if custom_debug_dir else self.config.debug_dir
        state = RequestState(image_pil, question, output_dir=out_dir, debug=self.config.debug)
        self.find_targets(state)

        # 安全检查：如果没提取出 target，直接用原图回答
        if not state.targets:
            return self._generate_from_vllm(question, image_pil)

        try:
            while not self.reasoning_step(state): pass
        except Exception as e:
            print(f"[Echo] Exception during reasoning (e.g., crop error): {e}. Falling back to baseline.")
            return self._generate_from_vllm(question, state.original_image)

        # 0317: 修复了回滚策略，我记得以前就有的...另外发现的问题：模型效果变差应该是：response的提取格式不对，导致Echo inference提前结束，而这里没有用原图兜底，导致很多图只看了局部！
        if not state.found_targets or any(t not in state.found_targets.keys() for t in state.targets):
            return self._generate_from_vllm(question, image_pil)

        # 核心改进：Focus Crop 逻辑
        # 1. 计算所有已发现目标的最小包围盒 (MBB)
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

        # 2. 膨胀 MBB 范围以增加上下文 (增加 缓冲区 像素)
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
            print(f"[Echo] Exception during focus crop ({e}). Falling back to baseline.")
            return self._generate_from_vllm(question, state.original_image)

        draw = ImageDraw.Draw(focus_img)
        
        # 3. 在 Focus 图上画出各物体的标记 (颜色区分 + 膨胀提示)
        # 为不同物体分配不同颜色以免混淆
        if self.config.visual_prompt:
            colors = ["red", "blue", "green", "magenta", "orange", "cyan"]
            target_color_map = {t: colors[i % len(colors)] for i, t in enumerate(state.found_targets.keys())}
    
            from PIL import ImageFont
            focus_w, focus_h = focus_img.size
            # 根据最终输入给模型的图片的尺寸动态决定字体和线框的粗细
            font_size = max(24, int(max(focus_w, focus_h) * 0.02))
            try:
                # 引入常见字体以支持调节字体大小
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
                font_size = 15
                
            line_width = max(3, int(max(focus_w, focus_h) * 0.005))
    
            for target, box in all_boxes:
                color = target_color_map[target]
    
                # 物体大小
                obj_w = box[2] - box[0]
                obj_h = box[3] - box[1]
    
                # 动态膨胀：根据物体大小动态膨胀10%，并限高防止框超出图片范围太多
                dilate_x = min(int(focus_w * 0.05), max(5, int(obj_w * 0.1)))
                dilate_y = min(int(focus_h * 0.05), max(5, int(obj_h * 0.1)))
    
                # 转换全局到局部坐标
                rel_box = [box[0] - fl - dilate_x, box[1] - ft - dilate_y, box[2] - fl + dilate_x, box[3] - ft + dilate_y]
                draw.rectangle(rel_box, outline=color, width=line_width)
    
                text_x, text_y = rel_box[0], rel_box[1] - font_size - 5
    
                try: # 增加一个白底以便于看清文字
                    rect_bbox = draw.textbbox((text_x, text_y), target, font=font)
                    draw.rectangle((rect_bbox[0]-2, rect_bbox[1]-2, rect_bbox[2]+2, rect_bbox[3]+2), fill="white")
                except Exception:
                    pass
    
                draw.text((text_x, text_y), target, fill=color, font=font)

        # 4. 构造极简的 Final Prompt
        if self.config.visual_prompt:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(focus_img)}"}},
                {"type": "text", "text": f"Task: {question}\n\nI have localized the relevant objects for the task. Below is a high-resolution focused image showing the targets marked with bounding boxes. \nPlease analyze the focused visual information and provide the final answer to the question: {question}"},
                # {"type": "text", "text": f"Task: {question}\n"},
                # {"type": "text", "text": f""}
            ]
        else:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(focus_img)}"}},
                # {"type": "text", "text": f"Task: {question}\n\nI have localized the relevant objects for the task. Below is a high-resolution focused image showing the targets marked with bounding boxes. \nPlease analyze the focused visual information and provide the final answer to the question: {question}"},
                {"type": "text", "text": f"Task: {question}\n"},
                # {"type": "text", "text": f""}
            ]

        if state.debug:
            focus_img.save(os.path.join(state.output_dir, "final_focus_crop.jpg"))

        res = self.client.chat.completions.create(model=self.config.model_name,
                                                  messages=[{"role": "user", "content": content}],
                                                  temperature=self.config.temperature, seed=self.config.seed)
        print("⭐ Final answer:", res.choices[0].message.content)
        return res.choices[0].message.content