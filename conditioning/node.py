import torch
import torch.nn as nn
import json

class ConditioningInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("TENSOR", "TENSOR", "DICT", "STRING", "TENSOR") 
    RETURN_NAMES = ("cond_tensor", "pooled_tensor", "details_dict", "nested_keys", "attention_mask")
    FUNCTION = "inspect"
    CATEGORY = "spawner/conditioning"

    def _get_nested_keys(self, data, parent_key='', sep='.'):
        keys = []
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    keys.extend(self._get_nested_keys(v, new_key, sep))
                else:
                    keys.append(new_key)
        return keys

    def inspect(self, conditioning):
        if not isinstance(conditioning, list) or len(conditioning) == 0:
            raise ValueError("输入的 Conditioning 无效或为空。")
        
        cond_tensor = conditioning[0][0]
        details_dict = conditioning[0][1].copy() if isinstance(conditioning[0][1], dict) else {}

        nested_keys_list = self._get_nested_keys(details_dict)
        nested_keys_str = "\n".join(nested_keys_list) if nested_keys_list else "无可用键"

        attention_mask = details_dict.get("attention_mask", None)
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            attention_mask = None

        pooled_tensor = details_dict.get("pooled_output", None)
        if pooled_tensor is None or not isinstance(pooled_tensor, torch.Tensor):
            if len(cond_tensor.shape) > 2:
                pooled_tensor = torch.zeros_like(cond_tensor[:, 0])
            else:
                pooled_tensor = torch.zeros_like(cond_tensor)
        
        return (cond_tensor, pooled_tensor, details_dict, nested_keys_str, attention_mask)

class TensorInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tensor": ("TENSOR",)}}
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("shape", "dtype", "device", "min_value", "max_value", "mean_value")
    FUNCTION = "inspect"
    CATEGORY = "spawner/tensor"
    def inspect(self, tensor):
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return ("None", "None", "None", 0.0, 0.0, 0.0)
            
        shape = str(list(tensor.shape))
        dtype = str(tensor.dtype)
        device = str(tensor.device)
        if tensor.numel() == 0: 
            return (shape, dtype, device, 0.0, 0.0, 0.0)
            
        sample = tensor.flatten().float()
        if sample.numel() > 1000000: 
            sample = sample[torch.randperm(sample.numel(), device=sample.device)[:1000000]]
            
        return (shape, dtype, device, 
                torch.min(sample).item(), 
                torch.max(sample).item(), 
                torch.mean(sample).item())

class ConditioningPacker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emb": ("TENSOR",),
            },
            "optional": {
                "pooled_emb": ("TENSOR",),
                "attention_mask": ("TENSOR",),
                "details_dict": ("*",),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "pack"
    CATEGORY = "spawner/conditioning"

    def pack(self, emb, details_dict=None, pooled_emb=None, attention_mask=None):
        if not isinstance(emb, torch.Tensor):
            raise ValueError("emb 必须是张量类型")
        emb = emb.view([int(dim) for dim in emb.shape])
        
        if isinstance(details_dict, dict):
            new_details = details_dict.copy()
        else:
            new_details = {}

        if len(emb.shape) >= 2:
            seq_len = int(emb.shape[1])
            new_details["seq_len"] = seq_len
            new_details["max_seq_len"] = seq_len

        if isinstance(pooled_emb, torch.Tensor):
            pooled_emb = pooled_emb.view([int(dim) for dim in pooled_emb.shape])
            if pooled_emb.shape[0] != emb.shape[0]:
                raise ValueError(f"pooled_emb与emb的batch_size不匹配！"
                                 f"emb: {emb.shape[0]}, pooled_emb: {pooled_emb.shape[0]}")
            if pooled_emb.shape[-1] != emb.shape[-1]:
                raise ValueError(f"pooled_emb与emb的特征维度不匹配！"
                                 f"emb: {emb.shape[-1]}, pooled_emb: {pooled_emb.shape[-1]}")
            new_details["pooled_output"] = pooled_emb
        elif "pooled_output" not in new_details:
            if len(emb.shape) > 2:
                new_details["pooled_output"] = torch.zeros_like(emb[:, 0])
            else:
                new_details["pooled_output"] = torch.zeros_like(emb)

        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.view([int(dim) for dim in attention_mask.shape])
            new_details["attention_mask"] = attention_mask

        return ([(emb, new_details)],)

class ConditioningCrossAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning_q": ("CONDITIONING",), "conditioning_kv": ("CONDITIONING",), "n_heads": ("INT", {"default": 8, "min": 1}),"add_residual": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_cross_attention"
    CATEGORY = "spawner/conditioning"
    def apply_cross_attention(self, conditioning_q, conditioning_kv, n_heads, add_residual):
        def extract_cond_info(conditioning, name):
            if not (isinstance(conditioning, list) and len(conditioning) > 0 and isinstance(conditioning[0], tuple)):
                raise ValueError(f"{name} 是无效的CONDITIONING对象")
            cond_tensor = conditioning[0][0]
            cond_details = conditioning[0][1] if len(conditioning[0]) > 1 else {}
            if not isinstance(cond_tensor, torch.Tensor):
                raise ValueError(f"{name} 中无有效张量")
            return cond_tensor, cond_details

        cond_q, q_details = extract_cond_info(conditioning_q, "conditioning_q")
        cond_kv, kv_details = extract_cond_info(conditioning_kv, "conditioning_kv")

        embedding_dim = cond_q.shape[-1]
        if embedding_dim != cond_kv.shape[-1]: 
            raise ValueError(f"Q和KV的嵌入维度不匹配! Q: {embedding_dim}, KV: {cond_kv.shape[-1]}")
        if embedding_dim % n_heads != 0: 
            raise ValueError(f"嵌入维度 ({embedding_dim}) 必须能被注意力头数 ({n_heads}) 整除。")
        
        kv_mask = kv_details.get("attention_mask")
        if kv_mask is not None and isinstance(kv_mask, torch.Tensor):
            kv_mask = kv_mask.view([int(dim) for dim in kv_mask.shape])
            if len(kv_mask.shape) != 2 or kv_mask.shape[0] != cond_kv.shape[0] or kv_mask.shape[1] != cond_kv.shape[1]:
                kv_mask = None

        device, dtype = cond_q.device, cond_q.dtype
        attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, batch_first=True).to(device, dtype=dtype)
        key_padding_mask = ~kv_mask.bool() if kv_mask is not None else None
        attn_output, _ = attention_layer(
            query=cond_q, 
            key=cond_kv, 
            value=cond_kv,
            key_padding_mask=key_padding_mask
        )

        final_cond = (cond_q + attn_output) if add_residual else attn_output
        return ([(final_cond, q_details)],)

class TensorShapeAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tensor_to_align": ("TENSOR",),"tensor_reference": ("TENSOR",)}}
    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "shape_adapt"
    CATEGORY = "spawner/tensor"
    def shape_adapt(self, tensor_to_align, tensor_reference):
        if not isinstance(tensor_to_align, torch.Tensor) or not isinstance(tensor_reference, torch.Tensor):
            raise ValueError("输入必须是张量类型")

        tensor_to_align = tensor_to_align.view([int(dim) for dim in tensor_to_align.shape])
        tensor_reference = tensor_reference.view([int(dim) for dim in tensor_reference.shape])
        
        source, reference = tensor_to_align, tensor_reference
        source_dim, target_dim = source.shape[-1], reference.shape[-1]
        if source_dim == target_dim: 
            return (source,)
            
        linear_layer = nn.Linear(source_dim, target_dim).to(source.device, dtype=source.dtype)
        aligned_tensor = linear_layer(source)
        aligned_tensor = aligned_tensor.view([int(dim) for dim in aligned_tensor.shape])
        return (aligned_tensor,)

class ConditioningConcatenation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning1": ("CONDITIONING",),
                "conditioning2": ("CONDITIONING",)
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("concatenated_conditioning",)
    FUNCTION = "concatenate"
    CATEGORY = "spawner/conditioning"

    def concatenate(self, conditioning1, conditioning2):
        cond1, details1 = conditioning1[0][0], conditioning1[0][1].copy()
        cond2, details2 = conditioning2[0][0], conditioning2[0][1].copy()
        
        if cond1.shape[-1] != cond2.shape[-1]:
            adapter = TensorShapeAdapter()
            cond2 = adapter.shape_adapt(cond2, cond1)[0]
        
        concatenated_cond = torch.cat([cond1, cond2], dim=1)
        
        pooled1 = details1.get("pooled_output", None)
        pooled2 = details2.get("pooled_output", None)
        pooled_list = []
        if isinstance(pooled1, torch.Tensor):
            pooled_list.append(pooled1)
        if isinstance(pooled2, torch.Tensor):
            if pooled_list and pooled2.shape[-1] != pooled_list[0].shape[-1]:
                adapter = TensorShapeAdapter()
                pooled2 = adapter.shape_adapt(pooled2, pooled_list[0])[0]
            pooled_list.append(pooled2)
        
        mask1 = details1.get("attention_mask", None)
        mask2 = details2.get("attention_mask", None)
        masks = []
        if isinstance(mask1, torch.Tensor):
            masks.append(mask1)
        if isinstance(mask2, torch.Tensor):
            masks.append(mask2)
        
        new_details = {}
        if pooled_list:
            new_details["pooled_output"] = torch.mean(torch.stack(pooled_list), dim=0)
        if masks and len(masks) == 2 and masks[0].shape[0] == masks[1].shape[0]:
            new_details["attention_mask"] = torch.cat(masks, dim=1)
        elif masks:
            new_details["attention_mask"] = masks[0]
        
        return ([(concatenated_cond, new_details)],)


class ConditioningPooledMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning1": ("CONDITIONING",),
                "conditioning2": ("CONDITIONING",),
                "merge_strategy": (["concat", "add", "mean", "max"], {"default": "mean"})
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("merged_conditioning",)
    FUNCTION = "merge"
    CATEGORY = "spawner/conditioning"

    def _masked_pool(self, embed, mask=None):
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.view([int(dim) for dim in mask.shape])
            mask = mask.unsqueeze(-1)
            return (embed * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        return torch.mean(embed, dim=1)

    def merge(self, conditioning1, conditioning2, merge_strategy):
        cond1, details1 = conditioning1[0][0], conditioning1[0][1].copy()
        cond2, details2 = conditioning2[0][0], conditioning2[0][1].copy()

        cond1 = cond1.view([int(dim) for dim in cond1.shape])
        cond2 = cond2.view([int(dim) for dim in cond2.shape])

        mask1 = details1.get("attention_mask", None)
        mask2 = details2.get("attention_mask", None)
        if mask1 is not None and isinstance(mask1, torch.Tensor):
            mask1 = mask1.view([int(dim) for dim in mask1.shape])
        if mask2 is not None and isinstance(mask2, torch.Tensor):
            mask2 = mask2.view([int(dim) for dim in mask2.shape])

        pooled1 = self._masked_pool(cond1, mask1)
        pooled2 = self._masked_pool(cond2, mask2)

        if pooled1.shape[-1] != pooled2.shape[-1]:
            adapter = TensorShapeAdapter()
            pooled2 = adapter.shape_adapt(pooled2, pooled1)[0]
            pooled2 = pooled2.view([int(dim) for dim in pooled2.shape])

        if merge_strategy == "concat":
            merged_pooled = torch.cat([pooled1, pooled2], dim=-1)
            merged_cond = merged_pooled.unsqueeze(1)
        elif merge_strategy == "add":
            merged_pooled = pooled1 + pooled2
            merged_cond = merged_pooled.unsqueeze(1)
        elif merge_strategy == "mean":
            merged_pooled = (pooled1 + pooled2) / 2
            merged_cond = merged_pooled.unsqueeze(1)
        elif merge_strategy == "max":
            merged_pooled = torch.max(pooled1, pooled2)
            merged_cond = merged_pooled.unsqueeze(1)
        else:
            raise ValueError(f"不支持的合并策略: {merge_strategy}")

        merged_cond = merged_cond.view([int(dim) for dim in merged_cond.shape])
        new_details = {
            "pooled_output": merged_pooled,
            "seq_len": int(merged_cond.shape[1]),
            "max_seq_len": int(merged_cond.shape[1])
        }

        if mask1 is not None:
            new_details["attention_mask"] = mask1
        elif mask2 is not None:
            new_details["attention_mask"] = mask2

        return ([(merged_cond, new_details)],)

class TensorConcatenation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor1": ("TENSOR",),
                "tensor2": ("TENSOR",),
                "dim": ("INT", {"default": 1, "min": -10, "max": 10}),
            },
            "optional": {
                "enable_debug": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("concatenated_tensor", "debug_info")
    FUNCTION = "concatenate"
    CATEGORY = "spawner/tensor"

    def concatenate(self, tensor1, tensor2, dim, enable_debug=False):
        def normalize_dim(dim, tensor):
            if dim < 0:
                return len(tensor.shape) + dim
            return dim

        tensor1 = tensor1.contiguous()
        tensor2 = tensor2.contiguous()

        normalized_dim = normalize_dim(dim, tensor1)
        if normalized_dim < 0 or normalized_dim >= len(tensor1.shape):
            raise ValueError(f"维度 {dim}（标准化后 {normalized_dim}）对于形状 {tensor1.shape} 无效")
            
        if len(tensor1.shape) != len(tensor2.shape):
            raise ValueError(f"张量维度数量不匹配: {len(tensor1.shape)} vs {len(tensor2.shape)}")

        for i in range(len(tensor1.shape)):
            if i != normalized_dim and tensor1.shape[i] != tensor2.shape[i]:
                raise ValueError(f"维度 {i} 大小不匹配: {tensor1.shape[i]} vs {tensor2.shape[i]}")
                
        if tensor1.device != tensor2.device:
            tensor2 = tensor2.to(tensor1.device, non_blocking=True)
            
        concatenated = torch.cat([tensor1, tensor2], dim=normalized_dim)
        
        debug_info = ""
        if enable_debug:
            reversed_concat = torch.cat([tensor2, tensor1], dim=normalized_dim)
            order_sensitivity = torch.norm(concatenated - reversed_concat).item()
            
            debug_info = (
                f"拼接详情:\n"
                f"- 拼接维度: {dim}（标准化后: {normalized_dim}\n"
                f"- tensor1形状: {tensor1.shape}（设备: {tensor1.device}\n"
                f"- tensor2形状: {tensor2.shape}（设备: {tensor2.device}\n"
                f"- 拼接后形状: {concatenated.shape}\n"
                f"- 顺序敏感性（差异范数）: {order_sensitivity:.6f}"
            )
        
        return (concatenated, debug_info)

class TensorPooledMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor1": ("TENSOR",),
                "pool_dim": ("INT", {"default": 1, "min": -4, "max": 4}),
                "merge_strategy": (["concat", "add", "mean", "max"], {"default": "mean"})
            },
            "optional": {
                "tensor2": ("TENSOR",),
                "mask1": ("TENSOR",),
                "mask2": ("TENSOR",)
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("merged_tensor",)
    FUNCTION = "merge"
    CATEGORY = "spawner/tensor"

    def _masked_pool(self, tensor, mask=None, pool_dim=1):
        pool_dim = int(pool_dim)
        tensor = tensor.view([int(d) for d in tensor.shape])
        
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.view([int(d) for d in mask.shape])
            mask_dims = [1] * len(tensor.shape)
            mask_dims[pool_dim] = -1
            mask = mask.view(mask_dims)
            pooled = (tensor * mask).sum(dim=pool_dim) / mask.sum(dim=pool_dim).clamp(min=1e-8)
            return pooled.view([int(d) for d in pooled.shape])
        pooled = torch.mean(tensor, dim=pool_dim)
        return pooled.view([int(d) for d in pooled.shape])

    def merge(self, tensor1, pool_dim, merge_strategy, tensor2=None, mask1=None, mask2=None):
        def normalize_dim(dim, tensor):
            dim = int(dim)
            if dim < 0:
                dim = len(tensor.shape) + dim
            if dim < 0 or dim >= len(tensor.shape):
                raise ValueError(f"无效维度 {dim}（原始维度 {pool_dim}），张量形状 {tensor.shape}")
            return dim
        
        tensor1 = tensor1.view([int(d) for d in tensor1.shape])
        normalized_pool_dim = normalize_dim(pool_dim, tensor1)

        if tensor2 is None:
            pooled = self._masked_pool(tensor1, mask1, normalized_pool_dim)
            return (pooled,)

        tensor2 = tensor2.view([int(d) for d in tensor2.shape])
        if tensor1.device != tensor2.device:
            tensor2 = tensor2.to(tensor1.device)

        pooled1 = self._masked_pool(tensor1, mask1, normalized_pool_dim)
        pooled2 = self._masked_pool(tensor2, mask2, normalized_pool_dim)

        if pooled1.shape[-1] != pooled2.shape[-1]:
            adapter = nn.Linear(pooled2.shape[-1], pooled1.shape[-1], 
                               device=pooled1.device, dtype=pooled1.dtype)
            pooled2 = adapter(pooled2)
            pooled2 = pooled2.view([int(d) for d in pooled2.shape])

        if merge_strategy == "concat":
            merged = torch.cat([pooled1, pooled2], dim=-1)
        elif merge_strategy == "add":
            merged = pooled1 + pooled2
        elif merge_strategy == "mean":
            merged = (pooled1 + pooled2) / 2
        elif merge_strategy == "max":
            merged = torch.max(pooled1, pooled2)
        else:
            raise ValueError(f"不支持的合并策略: {merge_strategy}")

        merged = merged.view([int(d) for d in merged.shape])
        return (merged,)


class TensorAttentionFusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor1": ("TENSOR",),
                "tensor2": ("TENSOR",),
                "n_heads": ("INT", {"default": 8, "min": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1})
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("fused_tensor",)
    FUNCTION = "fuse"
    CATEGORY = "spawner/tensor"

    def fuse(self, tensor1, tensor2, n_heads, temperature):
        if len(tensor1.shape) != 3 or len(tensor2.shape) != 3:
            raise ValueError("tensor1和tensor2必须是3维张量 [batch_size, seq_len, hidden_dim]")
        if tensor1.shape[0] != tensor2.shape[0]:
            raise ValueError(f"tensor1和tensor2的batch_size不匹配: {tensor1.shape[0]} vs {tensor2.shape[0]}")
        if tensor1.shape[1] != tensor2.shape[1]:
            print(f"提示：tensor1的seq_len={tensor1.shape[1]}，tensor2的seq_len={tensor2.shape[1]}，融合后seq_len={tensor1.shape[1]+tensor2.shape[1]}")

        if tensor1.shape[-1] != tensor2.shape[-1]:
            adapter = TensorShapeAdapter()
            tensor2 = adapter.shape_adapt(tensor2, tensor1)[0]

        embedding_dim = tensor1.shape[-1]
        if embedding_dim % n_heads != 0:
            raise ValueError(f"嵌入维度 ({embedding_dim}) 必须能被注意力头数量 ({n_heads}) 整除")

        device = tensor1.device
        dtype = tensor1.dtype
        attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            batch_first=True
        ).to(device, dtype=dtype)

        combined = torch.cat([tensor1, tensor2], dim=1)
        attn_output, _ = attention(query=combined, key=combined, value=combined)
        fused_tensor = attn_output / temperature

        fused_tensor = fused_tensor.view([int(dim) for dim in fused_tensor.shape])
        return (fused_tensor,)

class TensorCrossAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query_tensor": ("TENSOR",),
                "key_tensor": ("TENSOR",),
                "value_tensor": ("TENSOR",),
                "n_heads": ("INT", {"default": 8, "min": 1}),
                "add_residual": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "query_mask": ("TENSOR",),
                "key_mask": ("TENSOR",)
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output_tensor",)
    FUNCTION = "apply_cross_attention"
    CATEGORY = "spawner/tensor"

    def apply_cross_attention(self, query_tensor, key_tensor, value_tensor, n_heads, add_residual, query_mask=None, key_mask=None):
        if not (isinstance(query_tensor, torch.Tensor) and isinstance(key_tensor, torch.Tensor) and isinstance(value_tensor, torch.Tensor)):
            raise ValueError("query/key/value必须是张量类型")
        if key_tensor.shape != value_tensor.shape:
            raise ValueError(f"key和value形状必须一致！当前key: {key_tensor.shape}, value: {value_tensor.shape}")
        if len(query_tensor.shape) != 3 or len(key_tensor.shape) != 3:
            raise ValueError("query/key/value必须是3维张量 [batch_size, seq_len, hidden_dim]")
        
        hidden_dim_q = query_tensor.shape[-1]
        hidden_dim_k = key_tensor.shape[-1]
        if hidden_dim_q != hidden_dim_k:
            adapter = TensorShapeAdapter()
            key_tensor = adapter.shape_adapt(key_tensor, query_tensor)[0]
            value_tensor = adapter.shape_adapt(value_tensor, query_tensor)[0]
        
        if hidden_dim_q % n_heads != 0:
            raise ValueError(f"特征维度 {hidden_dim_q} 必须能被注意力头数 {n_heads} 整除（建议调整n_heads为4/8/16）")
        
        def process_mask(mask, tensor, is_query=True):
            if mask is None:
                return None
            if len(mask.shape) != 2:
                raise ValueError(f"{('query' if is_query else 'key')} mask必须是2维张量 [batch_size, seq_len]")
            if mask.shape[0] != tensor.shape[0]:
                raise ValueError(f"{('query' if is_query else 'key')} mask的batch_size {mask.shape[0]} 与张量的batch_size {tensor.shape[0]} 不匹配")
            tensor_seq_len = tensor.shape[1]
            mask_seq_len = mask.shape[1]
            if mask_seq_len != tensor_seq_len:
                raise ValueError(f"{('query' if is_query else 'key')} mask序列长度 {mask_seq_len} 与张量序列长度 {tensor_seq_len} 不匹配")
            return mask.to(tensor.device, tensor.dtype)
        
        query_mask_processed = process_mask(query_mask, query_tensor, is_query=True)
        key_mask_processed = process_mask(key_mask, key_tensor, is_query=False)

        device = query_tensor.device
        dtype = query_tensor.dtype
        attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_dim_q,
            num_heads=n_heads,
            batch_first=True
        ).to(device, dtype)

        key_padding_mask = ~key_mask_processed.bool() if key_mask_processed is not None else None
        attn_output, _ = attention_layer(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        if add_residual:
            output_tensor = query_tensor + attn_output
        else:
            output_tensor = attn_output
        return (output_tensor.view([int(dim) for dim in output_tensor.shape]),)

class AllOnesMaskGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("TENSOR",),
                "seq_dim": ("INT", {"default": 1, "min": -4, "max": 4}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("all_ones_mask",)
    FUNCTION = "generate_mask"
    CATEGORY = "spawner/tensor"

    def _normalize_dim(self, dim, tensor):
        dim = int(dim)
        if dim < 0:
            return len(tensor.shape) + dim
        return dim

    def generate_mask(self, tensor, seq_dim=1):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("输入必须是张量类型")
            
        tensor = tensor.view([int(d) for d in tensor.shape])
        normalized_seq_dim = self._normalize_dim(seq_dim, tensor)
            
        if normalized_seq_dim < 0 or normalized_seq_dim >= len(tensor.shape):
            raise ValueError(f"序列维度 {seq_dim}（标准化后为 {normalized_seq_dim}）对于形状 {tensor.shape} 无效")
        
        batch_size = int(tensor.shape[0]) if len(tensor.shape) > 0 else 1
        seq_len = int(tensor.shape[normalized_seq_dim])
        mask = torch.ones((batch_size, seq_len), device=tensor.device, dtype=torch.int32)
        
        return (mask,)

class TensorUnsqueeze:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("TENSOR",),
                "dim": ("INT", {"default": 1, "min": -4, "max": 4}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("expanded_tensor",)
    FUNCTION = "unsqueeze"
    CATEGORY = "spawner/tensor"

    def _normalize_dim(self, dim, tensor):
        dim = int(dim)
        if dim < 0:
            return len(tensor.shape) + dim
        return dim

    def unsqueeze(self, tensor, dim=1):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("输入必须是张量类型")
            
        tensor = tensor.view([int(d) for d in tensor.shape])
            
        normalized_dim = self._normalize_dim(dim, tensor)
        
        if normalized_dim < 0 or normalized_dim > len(tensor.shape):
            raise ValueError(f"无法在维度 {dim} 扩展，张量当前维度为 {len(tensor.shape)}")
            
        expanded = torch.unsqueeze(tensor, dim=normalized_dim)
        
        expanded = expanded.view([int(d) for d in expanded.shape])
        
        return (expanded,)
