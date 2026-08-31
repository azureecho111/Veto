import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import argparse

class VStarCropperApp:
    def __init__(self, root, results_path, image_root, output_dir):
        self.root = root
        self.root.title("V* Benchmark Error Case Cropper")
        self.root.geometry("1400x900")

        self.results_path = results_path
        self.image_root = image_root
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.error_cases = []
        self.current_idx = 0
        self.load_error_cases()

        # Image state
        self.image_pil = None
        self.display_photo = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Selection state
        self.rect_start_x = None
        self.rect_start_y = None
        self.current_rect = None

        self.setup_ui()
        self.load_case(0)

    def load_error_cases(self):
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # V* evaluation: is_correct is calculated in our vllm script
                    if not item.get('is_correct', False):
                        self.error_cases.append(item)
            if not self.error_cases:
                messagebox.showinfo("Info", "No error cases found in the results file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load results: {e}")

    def setup_ui(self):
        # Top info panel
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.pack(side=tk.TOP, fill=tk.X)

        self.info_label = ttk.Label(info_frame, text="Case info...", font=("Arial", 12))
        self.info_label.pack(side=tk.LEFT)

        self.nav_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.nav_label.pack(side=tk.RIGHT)

        # Right control panel
        ctrl_frame = ttk.Frame(self.root, padding="10", width=300)
        ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(ctrl_frame, text="Question:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.q_text = tk.Text(ctrl_frame, height=5, width=40, wrap=tk.WORD, font=("Arial", 10))
        self.q_text.pack(pady=5)

        ttk.Label(ctrl_frame, text="Model Prediction:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.pred_label = ttk.Label(ctrl_frame, text="", foreground="red", wraplength=280)
        self.pred_label.pack(pady=5, anchor=tk.W)

        ttk.Label(ctrl_frame, text="Ground Truth:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.gt_label = ttk.Label(ctrl_frame, text="", foreground="green")
        self.gt_label.pack(pady=5, anchor=tk.W)

        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(pady=20, fill=tk.X)

        ttk.Button(btn_frame, text="Previous", command=self.prev_case).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Next", command=self.next_case).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(ctrl_frame, text="CROP & SAVE", style='Accent.TButton', command=self.save_crop).pack(pady=10, fill=tk.X)
        ttk.Label(ctrl_frame, text="Controls:\n- Roll Mouse: Zoom\n- Right Click Drag: Pan\n- Left Click Drag: Select Crop", foreground="gray").pack(pady=20)

        # Main canvas
        self.canvas = tk.Canvas(self.root, bg="black", cursor="cross")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_drag)
        
        self.canvas.bind("<MouseWheel>", self.on_zoom) # Windows
        self.canvas.bind("<Button-4>", self.on_zoom) # Linux
        self.canvas.bind("<Button-5>", self.on_zoom) # Linux

    def load_case(self, idx):
        if not self.error_cases or idx < 0 or idx >= len(self.error_cases):
            return
        
        self.current_idx = idx
        case = self.error_cases[idx]
        img_path = os.path.join(self.image_root, case['image'])
        
        try:
            self.image_pil = Image.open(img_path).convert("RGB")
            # Reset view
            self.scale = min(self.canvas.winfo_width() / self.image_pil.width, 
                             self.canvas.winfo_height() / self.image_pil.height) if self.canvas.winfo_width() > 1 else 0.5
            self.offset_x = 0
            self.offset_y = 0
            self.show_image()
            
            # Update UI text
            self.info_label.config(text=f"ID: {case['question_id']} | Category: {case['category']}")
            self.nav_label.config(text=f"Case {idx + 1} / {len(self.error_cases)}")
            self.q_text.delete(1.0, tk.END)
            self.q_text.insert(tk.END, case['text'])
            self.pred_label.config(text=f"{case['processed_choice']} -> {case['prediction']}")
            self.gt_label.config(text=f"{case['label']}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image {img_path}: {e}")

    def show_image(self):
        if self.image_pil is None: return
        
        width = int(self.image_pil.width * self.scale)
        height = int(self.image_pil.height * self.scale)
        
        # Inplace resizing using high quality filter
        display_img = self.image_pil.resize((width, height), Image.Resampling.LANCZOS)
        self.display_photo = ImageTk.PhotoImage(display_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.display_photo)

    def on_zoom(self, event):
        if self.image_pil is None: return
        # Handle zoom
        if event.num == 4 or event.delta > 0: # Up
            self.scale *= 1.1
        else: # Down
            self.scale /= 1.1
        self.show_image()

    def on_pan_start(self, event):
        self.pan_last_x = event.x
        self.pan_last_y = event.y

    def on_pan_drag(self, event):
        dx = event.x - self.pan_last_x
        dy = event.y - self.pan_last_y
        self.offset_x += dx
        self.offset_y += dy
        self.pan_last_x = event.x
        self.pan_last_y = event.y
        self.show_image()

    def on_press(self, event):
        self.rect_start_x = event.x
        self.rect_start_y = event.y
        if self.current_rect:
            self.canvas.delete(self.current_rect)

    def on_drag(self, event):
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        self.current_rect = self.canvas.create_rectangle(
            self.rect_start_x, self.rect_start_y, event.x, event.y, outline="red", width=2
        )

    def on_release(self, event):
        self.rect_end_x = event.x
        self.rect_end_y = event.y

    def save_crop(self):
        if not self.rect_start_x or not self.image_pil:
            messagebox.showwarning("Warning", "Please select a region first!")
            return

        # Calculate coordinates in original image space
        x1 = min(self.rect_start_x, self.rect_end_x) - self.offset_x
        y1 = min(self.rect_start_y, self.rect_end_y) - self.offset_y
        x2 = max(self.rect_start_x, self.rect_end_x) - self.offset_x
        y2 = max(self.rect_start_y, self.rect_end_y) - self.offset_y

        orig_x1 = int(x1 / self.scale)
        orig_y1 = int(y1 / self.scale)
        orig_x2 = int(x2 / self.scale)
        orig_y2 = int(y2 / self.scale)

        # Clip coordinates
        orig_x1 = max(0, min(orig_x1, self.image_pil.width))
        orig_y1 = max(0, min(orig_y1, self.image_pil.height))
        orig_x2 = max(0, min(orig_x2, self.image_pil.width))
        orig_y2 = max(0, min(orig_y2, self.image_pil.height))

        if orig_x2 - orig_x1 < 10 or orig_y2 - orig_y1 < 10:
            messagebox.showwarning("Warning", "Selected area too small!")
            return

        cropped = self.image_pil.crop((orig_x1, orig_y1, orig_x2, orig_y2))
        
        # Save path maintaining folder structure but into output_dir
        orig_name = self.error_cases[self.current_idx]['image']
        # Flatten directory structure if needed or keep it
        save_name = os.path.basename(orig_name)
        save_path = os.path.join(self.output_dir, save_name)
        
        cropped.save(save_path)
        print(f"Saved crop to: {save_path}")
        # Visual feedback on GT label briefly
        self.gt_label.config(text=f"{self.error_cases[self.current_idx]['label']} - SAVED!")
        self.root.after(1000, lambda: self.gt_label.config(text=f"{self.error_cases[self.current_idx]['label']}"))

    def next_case(self):
        if self.current_idx < len(self.error_cases) - 1:
            self.load_case(self.current_idx + 1)

    def prev_case(self):
        if self.current_idx > 0:
            self.load_case(self.current_idx - 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to evaluation JSONL results")
    parser.add_argument("--image-root", type=str, required=True, help="Path to original V* images root")
    parser.add_argument("--output", type=str, default="vstar_hand_crops", help="Directory to save cropped images")
    args = parser.parse_args()

    root = tk.Tk()
    app = VStarCropperApp(root, args.results, args.image_root, args.output)
    root.mainloop()
