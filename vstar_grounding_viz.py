import json
import base64
import requests
import os
import uuid
import mimetypes
import re
from io import BytesIO
from PIL import Image, ImageDraw
from openai import OpenAI
from qwen_vl_utils import smart_resize


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

        if not os.path.exists("downloads"):
            os.makedirs("downloads")

        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)
        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_bbox(text):
    """
    Extract bbox from response. Handles formats like [x1, y1, x2, y2].
    """
    # Find the first sequence of 4 numbers in brackets
    match = re.search(r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]', text)
    if match:
        return [float(x) for x in match.groups()]

    try:
        data = json.loads(re.search(r'\{.*\}', text, re.DOTALL).group())
        if "bbox_2d" in data:
            return data["bbox_2d"]
    except:
        pass
    return None


def main():
    # --- CONFIGURATION ---
    image_source = "../ZoomEye/hf_vstar/direct_attributes/sa_29509.jpg"  # 修改为你的图片路径
    # image_source = "test.jpg"
    api_key = "EMPTY"
    base_url = "http://localhost:18903/v1"
    model_name = "qwen3-vl-8b"

    if not os.path.exists(image_source):
        print(f"Error: Image '{image_source}' not found.")
        return

    # 1. Encode image
    print(f"Encoding image: {image_source}...")
    base64_image = encode_image(image_source)

    # 2. Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": """
                    Please find a location along the four sides of the image where the target object "trash bin" is least likely to appear. Return its location in the form of coordinates. 
                    The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "sky"}.
                    Important: Provide ABSOLUTE pixel coordinates.
                    """
                }
            ]
        }
    ]

    # 3. Call API
    client = OpenAI(api_key=api_key, base_url=base_url)
    print("Calling Qwen API...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        response_text = response.choices[0].message.content
        print(f"Model Response: {response_text}")
    except Exception as e:
        print(f"API Error: {e}")
        return

    # 4. Extract BBox
    bbox_raw = extract_bbox(response_text)
    if not bbox_raw:
        print("Could not extract bounding box.")
        return

    # 5. Coordinate Mapping (INTELLIGENT SWAP DETECTION)
    print("Mapping coordinates to original resolution...")
    orig_img = Image.open(image_source).convert("RGB")
    orig_w, orig_h = orig_img.size
    
    # Calculate the resolution Qwen actually processed
    resized_h, resized_w = smart_resize(orig_h, orig_w)
    
    # Scaling factors
    scale_w = orig_w / resized_w
    scale_h = orig_h / resized_h
    
    c1, c2, c3, c4 = bbox_raw
    print(f"Model raw outputs: {bbox_raw}")

    # --- NATIVE SWAP DETECTION ---
    # We need to decide if the model followed the Prompt [x,y,x,y] or used default [y,x,y,x]
    # Indicator: If c1 > resized_h and resized_w > resized_h, then c1 MUST be x.
    is_x_first = False
    if max(c1, c3) > resized_h and max(c1, c3) <= resized_w:
        is_x_first = True
    elif max(c2, c4) > resized_w and max(c2, c4) <= resized_h:
        # Opposite case: first must be y because second exceeds width
        is_x_first = False
    else:
        # Fallback: Assume the Prompt was followed [x1, y1, x2, y2]
        is_x_first = True 
    
    if is_x_first:
        print("Interpreting coordinates as [x, y, x, y] (Prompt Format)")
        x_raw = [c1, c3]
        y_raw = [c2, c4]
    else:
        print("Interpreting coordinates as [y, x, y, x] (Native Qwen Format)")
        x_raw = [c2, c4]
        y_raw = [c1, c3]

    # Map back to original resolution with clamping
    left = max(0, min(x_raw)) * scale_w
    right = min(resized_w, max(x_raw)) * scale_w
    top = max(0, min(y_raw)) * scale_h
    bottom = min(resized_h, max(y_raw)) * scale_h
    
    # Final clamping to original resolution
    left, right = max(0, left), min(orig_w, right)
    top, bottom = max(0, top), min(orig_h, bottom)
    
    print(f"Final Scaled BBox: [L={left:.0f}, T={top:.0f}, R={right:.0f}, B={bottom:.0f}]")

    # 6. Visualization (test.jpg)
    viz_img = orig_img.copy()
    draw_viz = ImageDraw.Draw(viz_img)
    draw_viz.rectangle([left, top, right, bottom], outline="red", width=12)
    viz_img.save("test.jpg")
    print("Saved visualization to test.jpg")

    # 7. Process Image for Pruning (modified.jpg)
    print("Applying Negative Pruning logic...")
    processed_img = orig_img.copy()
    width, height = orig_w, orig_h
    # Edge threshold: 2% of the side length or 50 pixels (whichever is larger for high res)
    edge_thresh_w = max(50, orig_w * 0.02)
    edge_thresh_h = max(50, orig_h * 0.02)

    is_at_top = top <= edge_thresh_h
    is_at_bottom = bottom >= orig_h - edge_thresh_h
    is_at_left = left <= edge_thresh_w
    is_at_right = right >= orig_w - edge_thresh_w

    print(f"Edge Check: Top={is_at_top}, Bottom={is_at_bottom}, Left={is_at_left}, Right={is_at_right}")

    if is_at_top or is_at_bottom or is_at_left or is_at_right:
        print("BBox detected at EDGE. Executing CROP...")
        # Determine new boundaries
        new_left, new_top, new_right, new_bottom = 0, 0, orig_w, orig_h

        # Priority-based cropping to maintain a single rectangle
        if is_at_top:
            new_top = int(bottom)
        elif is_at_bottom:
            new_bottom = int(top)
        elif is_at_left:
            new_left = int(right)
        elif is_at_right:
            new_right = int(left)

        # Final safety check on crop size
        if new_right > new_left + 10 and new_bottom > new_top + 10:
            processed_img = processed_img.crop((new_left, new_top, new_right, new_bottom))
            print(f"SUCCESS: Image cropped to {processed_img.size[0]}x{processed_img.size[1]}")
        else:
            print("ERROR: Crop result would be too small or invalid. Falling back to Mask.")
            draw_mask = ImageDraw.Draw(processed_img)
            draw_mask.rectangle([left, top, right, bottom], fill="black")
    else:
        print("BBox detected in MIDDLE. Executing MASK...")
        draw_mask = ImageDraw.Draw(processed_img)
        draw_mask.rectangle([left, top, right, bottom], fill="black")

    # 8. Save result
    processed_img.save("modified.jpg")
    print("Process complete. Check 'test.jpg' and 'modified.jpg'")


if __name__ == "__main__":
    main()
