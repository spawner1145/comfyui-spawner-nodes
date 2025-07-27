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

        image_filepath = os.path.join(style_dir, f"{filename_base}_img.png")
        img_tensor = image[0].cpu().numpy()
        img_np = (img_tensor * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        pil_img.save(image_filepath, 'PNG')

        if conditioning and len(conditioning) > 0:
            cond_tensor = conditioning[0][0].cpu()
            details_dict = conditioning[0][1]
            
            embed_filepath = os.path.join(style_dir, f"{filename_base}_embed.pt")
            details_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in details_dict.items()}
            torch.save([[cond_tensor, details_cpu]], embed_filepath)
            
            if "pooled_output" in details_dict:
                pooled_filepath = os.path.join(style_dir, f"{filename_base}_pooled.pt")
                pooled_data = details_dict["pooled_output"].cpu() if isinstance(details_dict["pooled_output"], torch.Tensor) else details_dict["pooled_output"]
                torch.save(pooled_data, pooled_filepath)
            
            if "attention_mask" in details_dict:
                mask_filepath = os.path.join(style_dir, f"{filename_base}_mask.pt")
                mask_data = details_dict["attention_mask"].cpu() if isinstance(details_dict["attention_mask"], torch.Tensor) else details_dict["attention_mask"]
                torch.save(mask_data, mask_filepath)
        
        message = f"数据对 '{filename_base}' 已保存到:\n{style_dir}"
        print(message)
        return {"ui": {"text": [message]}}
