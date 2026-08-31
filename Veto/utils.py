import base64
import requests
import os
import uuid
import mimetypes
import re
from io import BytesIO
from PIL import Image
from qwen_vl_utils import smart_resize
from typing import Dict, List
def encode_image(image_input):
    """
    Encode an image to base64. Handles PIL Image objects, URLs, and local file paths.
    """
    if hasattr(image_input, 'save'):
        buffered = BytesIO()
        img_format = image_input.format if image_input.format else 'PNG'
        image_input.save(buffered, format=img_format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    image_path = image_input
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        response = requests.get(image_path, headers={"User-Agent": user_agent}, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type) or ".jpg"
        if not os.path.exists("downloads"): os.makedirs("downloads")
        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))
        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512): fh.write(chunk)
        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_bbox_from_text(text):
    """
    Extract the first [c1, c2, ...] from Qwen output. Works for [x, y] or [x1, y1, x2, y2].
    """
    match = re.search(r'\[\s*([\d\.\s,]+)\s*\]', text)
    if match:
        content = match.group(1)
        coords = [float(x.strip()) for x in content.split(',') if x.strip()]
        return coords
    return None

def extract_labeled_bboxes(text: str) -> Dict[str, List[float]]:
    """
    Extract multiple labeled bounding boxes or points. 
    Matches patterns like 'top: [x,y,x,y]', 'center: [x,y]', etc.
    Returns: {label: [coords], ...}
    """
    results = {}
    # Find label followed by [ ... ]
    pattern = r'(?P<label>\[?[\w"]+\]?)?\s*:?\s*\[\s*(?P<coords>[\d\.\s,]+)\s*\]'
    for match in re.finditer(pattern, text):
        label = match.group('label')
        if label:
            label = label.strip('"').lower()
        else:
            label = f"region_{len(results)}"
        
        coords_str = match.group('coords')
        coords = [float(x.strip()) for x in coords_str.split(',') if x.strip()]
        results[label] = coords
    return results

def extract_labeled_bboxes_qwen3(text: str) -> Dict[str, List[float]]:
    """
    Extract multiple labeled bounding boxes for Qwen3-VL style JSON output.
    Format: {"bbox_2d": [x1, y1, x2, y2], "label": "car"}
    """
    results = {}
    pattern = r'\{\s*"bbox_2d":\s*\[([\d\.\s,]+)\],\s*"label":\s*"([^"]+)"\s*\}'
    for match in re.finditer(pattern, text):
        coords_str = match.group(1)
        label = match.group(2).lower()
        coords = [float(x.strip()) for x in coords_str.split(',') if x.strip()]
        key = label
        idx = 1
        while key in results:
            key = f"{label}_{idx}"
            idx += 1
        results[key] = coords
    return results

def map_coords_to_orig(bbox_raw, orig_w, orig_h):
    """
    Maps absolute coordinates from Qwen's internal smart_resize back to original resolution.
    """
    rh, rw = smart_resize(orig_h, orig_w)
    sw, sh = orig_w / rw, orig_h / rh
    c1, c2, c3, c4 = bbox_raw
    is_x_first = True
    if is_x_first:
        x_raw, y_raw = [c1, c3], [c2, c4]
    else:
        x_raw, y_raw = [c2, c4], [c1, c3]
    left = max(0, min(x_raw)) * sw
    right = min(rw, max(x_raw)) * sw
    top = max(0, min(y_raw)) * sh
    bottom = min(rh, max(y_raw)) * sh
    return [int(left), int(top), int(right), int(bottom)]

def map_coords_to_orig_qwen3(bbox_raw, orig_w, orig_h):
    """
    Maps Qwen3 normalized coordinates (0-1000) to original resolution.
    Fixed: Qwen3 uses [x1, y1, x2, y2]
    """
    if len(bbox_raw) == 4:
        x1, y1, x2, y2 = bbox_raw
        left = x1 * orig_w / 1000.0
        top = y1 * orig_h / 1000.0
        right = x2 * orig_w / 1000.0
        bottom = y2 * orig_h / 1000.0
    elif len(bbox_raw) == 2:
        x, y = bbox_raw
        left = x * orig_w / 1000.0
        top = y * orig_h / 1000.0
        right, bottom = left, top
    else:
        return [0, 0, 0, 0]
    return [int(left), int(top), int(right), int(bottom)]

def crop_image(image_pil, bbox):
    return image_pil.crop(bbox)

def extract_targets(sentence: str, pattern = r"So I need the information about the following objects: (.+)"):
    match = re.search(pattern, sentence)
    if match:
        return match.group(1)
    return None

def split_targets_sentence(targets_sentence:str, split_tag = r' and |, '):
    if targets_sentence.endswith('.'):
        targets_sentence = targets_sentence[:-1]
    targets = re.split(split_tag, targets_sentence)
    return targets
