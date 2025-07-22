import torch
import torch.nn as nn
import json

class ConditioningInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("TENSOR", "TENSOR", "*") 
    RETURN_NAMES = ("cond_tensor", "pooled_tensor", "details_dict")
    FUNCTION = "inspect"
    CATEGORY = "spawner/conditioning"

    def inspect(self, conditioning):
        if not isinstance(conditioning, list) or len(conditioning) == 0:
            raise ValueError("输入的 Conditioning 无效或为空。")
        
        cond_tensor = conditioning[0][0]
        details_dict = conditioning[0][1].copy()

        pooled_tensor = details_dict.get("pooled_output", None)

        if pooled_tensor is None:
            if len(cond_tensor.shape) > 2:
                pooled_tensor = torch.zeros_like(cond_tensor[:, 0])
            else:
                pooled_tensor = torch.zeros_like(cond_tensor)
        
        num_tokens = details_dict.get("num_tokens", 0)

        return (cond_tensor, pooled_tensor, details_dict)

class TensorInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tensor": ("TENSOR",)}}
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("shape", "dtype", "device", "min_value", "max_value", "mean_value")
    FUNCTION = "inspect"
    CATEGORY = "spawner/conditioning"
    def inspect(self, tensor):
        shape = str(list(tensor.shape))
        dtype = str(tensor.dtype)
        device = str(tensor.device)
        if tensor.numel() == 0: return (shape, dtype, device, 0.0, 0.0, 0.0)
        sample = tensor.flatten().float()
        if sample.numel() > 1000000: sample = sample[torch.randperm(sample.numel())[:1000000]]
        return (shape, dtype, device, torch.min(sample).item(), torch.max(sample).item(), torch.mean(sample).item())

class ConditioningPacker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emb": ("TENSOR",),
            },
            "optional": {
                "pooled_emb": ("TENSOR",),
                "details_dict": ("*",),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "pack"
    CATEGORY = "spawner/conditioning"

    def pack(self, emb, details_dict=None, pooled_emb=None):
        if details_dict:
            new_details = details_dict.copy()
        else:
            new_details = {}
        if pooled_emb is not None:
            new_details["pooled_output"] = pooled_emb
        elif "pooled_output" not in new_details:
             if len(emb.shape) > 2:
                new_details["pooled_output"] = torch.zeros_like(emb[:, 0])
             else:
                new_details["pooled_output"] = torch.zeros_like(emb)

        return ([(emb, new_details)],)

class ConditioningCrossAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning_q": ("CONDITIONING",), "conditioning_kv": ("CONDITIONING",), "n_heads": ("INT", {"default": 8, "min": 1}),"add_residual": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_cross_attention"
    CATEGORY = "spawner/conditioning"
    def apply_cross_attention(self, conditioning_q, conditioning_kv, n_heads, add_residual):
        if not conditioning_q or not conditioning_kv: raise ValueError("输入的 Conditioning 不能为空。")
        cond_q, q_details_dict, cond_kv = conditioning_q[0][0], conditioning_q[0][1], conditioning_kv[0][0]
        embedding_dim = cond_q.shape[-1]
        if embedding_dim != cond_kv.shape[-1]: raise ValueError(f"输入Q和KV的嵌入维度不匹配! Q: {embedding_dim}, KV: {cond_kv.shape[-1]}")
        if embedding_dim % n_heads != 0: raise ValueError(f"嵌入维度 ({embedding_dim}) 必须能被注意力头数量 ({n_heads}) 整除。")
        device, dtype = cond_q.device, cond_q.dtype
        attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, batch_first=True).to(device, dtype=dtype)
        attn_output, _ = attention_layer(query=cond_q, key=cond_kv, value=cond_kv)
        final_cond = (cond_q + attn_output) if add_residual else attn_output
        return ([(final_cond, q_details_dict)],)

class TensorShapeAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tensor_to_align": ("TENSOR",),"tensor_reference": ("TENSOR",)}}
    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "shape_adapt"
    CATEGORY = "spawner/conditioning"
    def shape_adapt(self, tensor_to_align, tensor_reference):
        source, reference = tensor_to_align, tensor_reference
        source_dim, target_dim = source.shape[-1], reference.shape[-1]
        if source_dim == target_dim: return (source,)
        linear_layer = nn.Linear(source_dim, target_dim).to(source.device, dtype=source.dtype)
        aligned_tensor = linear_layer(source)
        return (aligned_tensor,)
