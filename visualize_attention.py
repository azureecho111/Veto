import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import cv2
import json
import os
import argparse
import random
from tqdm import tqdm

def get_attention_map(model, processor, image, text, layer_idx=24):
    """
    Perform a single forward pass and extract attention maps for the last input token
    using hooks to save VRAM by clearing unwanted attention matrices immediately.
    """
    # Prepare inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # Process images and text
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    # Move to model's primary device
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    captured_attn = [None]
    hooks = []

    # Aggressive search for the Transformer layers
    layers_backbone = None
    
    def find_layers(mod):
        # Look for 'layers' attribute that is a ModuleList or has many children
        if hasattr(mod, "layers") and isinstance(mod.layers, torch.nn.ModuleList):
            return mod.layers
        # Check child modules
        for name, child in mod.named_children():
            # Common names for the LLM backbone
            if name in ["model", "language_model", "llm"]:
                res = find_layers(child)
                if res is not None: return res
        return None

    layers_backbone = find_layers(model)
    
    # Final fallback: just find the first ModuleList that looks like layers
    if layers_backbone is None:
        for m in model.modules():
            if isinstance(m, torch.nn.ModuleList) and len(m) > 10: # Most LLMs have > 10 layers
                layers_backbone = m
                break
    
    if layers_backbone is None:
        print("Error: Could not find Transformer layers in the model. Structure:")
        for name, _ in model.named_children():
            print(f" - {name}")
        return None, None, None

    def hook_fn(module, input, output):
        # In Transformers, layer output is usually (hidden_states, attentions, past_key_values)
        if isinstance(output, tuple) and len(output) >= 2:
            attn_weights = output[1]
            if attn_weights is not None:
                # [batch, num_heads, q_len, k_len] -> [num_heads, k_len] for the last token
                if captured_attn[0] is None: 
                     captured_attn[0] = attn_weights[0, :, -1, :].detach().cpu()
            
            # CRITICAL: Clear the attention matrix from the output tuple to free VRAM
            new_output = list(output)
            new_output[1] = None
            return tuple(new_output)
        return output

    # Register hooks on all layers
    for i, layer in enumerate(layers_backbone):
        if i == layer_idx:
            hooks.append(layer.register_forward_hook(hook_fn))
        else:
            # For other layers, just clear the attention output
            def clear_hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    new_output = list(output)
                    new_output[1] = None
                    return tuple(new_output)
                return output
            hooks.append(layer.register_forward_hook(clear_hook))

    # Forward pass
    with torch.no_grad():
        # output_attentions=True is required for the layers to compute the matrices
        model(**inputs, output_attentions=True, use_cache=False)
    
    # Remove hooks
    for h in hooks:
        h.remove()

    if captured_attn[0] is None:
        print(f"Error: Failed to capture attention at layer {layer_idx}")
        return None, None, None

    # Average across heads
    avg_attn = captured_attn[0].mean(dim=0).numpy() # [k_len]

    # Find image token indices
    input_ids = inputs['input_ids'][0].cpu().numpy()
    
    # Get image grid information
    thw = inputs['image_grid_thw'][0].cpu().numpy()
    t, h, w = thw[0], thw[1], thw[2]
    
    # Locate image tokens
    start_token = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_token = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
    idx_start = np.where(input_ids == start_token)[0]
    idx_end = np.where(input_ids == end_token)[0]
    
    if len(idx_start) > 0 and len(idx_end) > 0:
        # The tokens between start and end are the image tokens
        # Note: Depending on the specific processor, there might be a factor in indexing
        image_attn = avg_attn[idx_start[0]+1 : idx_end[0]]
        
        if len(image_attn) == h * w:
            image_attn_grid = image_attn.reshape(h, w)
            return image_attn_grid, (h, w), (inputs['pixel_values'].shape)
        else:
            print(f"Warning: Expected {h*w} tokens, got {len(image_attn)}")
    
    return None, None, None

def visualize_attention(image_pil, attn_grid, output_path, question, prediction, is_correct):
    """
    Overlay attention map on image and add text info.
    """
    img = np.array(image_pil)
    img_h, img_w = img.shape[:2]
    
    # Normalizing attention map
    attn_min, attn_max = attn_grid.min(), attn_grid.max()
    attn_norm = (attn_grid - attn_min) / (attn_max - attn_min + 1e-8)
    
    # Resize attention map to image size
    heatmap = cv2.resize(attn_norm, (img_w, img_h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend images
    overlay = cv2.addWeighted(img, 0.6, heatmap_img, 0.4, 0)
    
    # Add text banner at bottom
    banner_h = 150
    final_img = np.ones((img_h + banner_h, img_w, 3), dtype=np.uint8) * 255
    final_img[:img_h, :img_w] = overlay
    
    # Put text
    font = cv2.FONT_HERSHEY_SIMPLEX
    status_text = "CORRECT" if is_correct else "INCORRECT"
    status_color = (0, 150, 0) if is_correct else (0, 0, 255) # Green vs Red in BGR

    cv2.putText(final_img, f"Q: {question[:80]}...", (10, img_h + 40), font, 0.7, (0,0,0), 2)
    cv2.putText(final_img, f"Pred: {prediction} | {status_text}", (10, img_h + 80), font, 0.7, status_color, 2)
    
    cv2.imwrite(output_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--results", type=str, required=True, help="Result JSONL from previous eval")
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="vstar_attention_maps")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--layer-idx", type=int, default=24, help="Layer index to extract attention from. Default 24.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Loading model from {args.model}...")
    # Use naive transformers implementation to get attentions
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager" # Required to get attention scores
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    print(f"Loading results from {args.results}...")
    dataset = []
    with open(args.results, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))

    # Sample cases randomly from the entire dataset
    samples = random.sample(dataset, min(len(dataset), args.num_samples))

    print("Generating attention maps...")
    for i, item in enumerate(tqdm(samples)):
        img_path = os.path.join(args.image_root, item['image'])
        if not os.path.exists(img_path):
            continue
            
        image_pil = Image.open(img_path).convert("RGB")
        question = item['text']
        is_correct = item.get('is_correct', False)
        
        attn_grid, grid_size, _ = get_attention_map(model, processor, image_pil, question, layer_idx=args.layer_idx)
        
        if attn_grid is not None:
            status_tag = "correct" if is_correct else "incorrect"
            out_name = f"attn_{i}_{status_tag}_{os.path.basename(item['image'])}"
            out_path = os.path.join(args.output_dir, out_name)
            visualize_attention(image_pil, attn_grid, out_path, question, item.get('prediction', 'N/A'), is_correct)
        else:
            print(f"Failed to extract attention for item {i}")

    print(f"Done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
