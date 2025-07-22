import torch
import os
import uuid
from PIL import Image
import numpy as np

class SaveTrainingDataPair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_path": ("STRING", {"default": "datasets/my_training_data"}),
                "style_name": ("STRING", {"default": "style_a"}),
                "image": ("IMAGE",),
                "conditioning": ("CONDITIONING",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path_info",)
    FUNCTION = "save_pair"
    CATEGORY = "spawner/dataset"
    OUTPUT_NODE = True

    def save_pair(self, root_path, style_name, image, conditioning):
        style_dir = os.path.join(root_path, style_name)
        os.makedirs(style_dir, exist_ok=True)
        filename_base = str(uuid.uuid4())

        image_filepath = os.path.join(style_dir, f"{filename_base}.png")
        embed_filepath = os.path.join(style_dir, f"{filename_base}.pt")

        img_tensor = image[0].cpu().numpy()
        img_np = (img_tensor * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        pil_img.save(image_filepath, 'PNG')

        if conditioning:
            cond_cpu = conditioning[0][0].cpu()
            details_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in conditioning[0][1].items()}
            data_to_save = [[cond_cpu, details_cpu]]
            torch.save(data_to_save, embed_filepath)
        
        message = f"数据对 '{filename_base}' 已保存到:\n{style_dir}"
        print(message)
        return {"ui": {"text": [message]}}
