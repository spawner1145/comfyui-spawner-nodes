import base64
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import folder_paths

class ImageFromBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_str": (
                    "STRING", 
                    {
                        "label": "Base64字符串",
                        "multiline": True,
                        "placeholder": "粘贴Base64编码（支持带前缀，如data:image/png;base64,）...",
                        "rows": 5
                    }
                )
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process_base64"
    CATEGORY = "spawner/utils"

    def process_base64(self, base64_str):
        if not base64_str.strip():
            raise ValueError("Base64字符串不能为空，请粘贴有效的图像编码")
        
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]

        try:
            img_data = base64.b64decode(base64_str, validate=True)
            with Image.open(BytesIO(img_data)) as img:
                img_rgba = img.convert("RGBA")
        except base64.binascii.Error:
            raise ValueError("Base64编码无效，请检查字符串是否完整")
        except Exception as e:
            raise ValueError(f"图像解码失败：{str(e)}（可能不是有效的图像Base64）")

        rgb_img = img_rgba.convert("RGB")
        img_array = np.array(rgb_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        alpha_channel = img_rgba.split()[-1]
        mask_array = np.array(alpha_channel).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(-1)

        return (img_tensor, mask_tensor)


class ImageMaskToBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"label": "上游图像"})
            },
            "optional": {
                "mask": ("MASK", {"label": "上游蒙版"}),
                "add_data_prefix": (
                    "BOOLEAN", 
                    {"label": "添加Data前缀", "default": True}
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base64_str",)
    FUNCTION = "convert_to_base64"
    CATEGORY = "spawner/utils"

    def convert_to_base64(self, image, mask=None, add_data_prefix=True):
        img_np = image[0].cpu().numpy()
        img_array = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        if mask is not None:
            mask_np = mask[0, :, :, 0].cpu().numpy()
            mask_array = (mask_np * 255).astype(np.uint8)
            alpha_img = Image.fromarray(mask_array, mode="L")
            pil_img = pil_img.convert("RGBA")
            pil_img.putalpha(alpha_img)

        img_buffer = BytesIO()
        pil_img.save(img_buffer, format="png", optimize=True)

        img_buffer.seek(0)
        base64_bytes = base64.b64encode(img_buffer.read())
        base64_str = base64_bytes.decode("utf-8")

        if add_data_prefix:
            base64_str = f"data:image/png;base64,{base64_str}"

        return (base64_str,)
